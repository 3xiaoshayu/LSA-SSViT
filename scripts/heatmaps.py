#!/usr/bin/env python
"""Token-level class activation maps, weakly supervised localization and faithfulness checks.

Each head is a linear layer applied to the mean of the final (layer-normalized) patch tokens, so its logit
decomposes exactly into per-token contributions:

    logit_k = b_k + (1 / N) * sum_j  w_k . z_j

The contributions w_k . z_j on the 64 x 64 token grid form the class activation map (CAM) of head k. No gradient
is needed, and the mean of the map plus the bias reproduces the logit (checked for every image).

The comparison networks (COMPARATOR_MODELS, trained with comparators/train_comparator.py; the map figures show those
of BLOCKS in scripts/plot_heatmaps.py) are mapped with Grad-CAM (Selvaraju et al., 2017) on their last spatial feature
map A, without the final ReLU:

    M_k = sum_c alpha_kc A_c,   alpha_kc = spatial mean of d logit_k / d A_c

For a linear head on average-pooled features alpha_kc = w_kc / N, so M_k is the exact CAM divided by N: both methods
give the same maps up to a positive scale (gradcam-check), and the display of scripts/plot_heatmaps.py does not depend
on that scale.
Feature maps (at 1024 px; 1022 px for the patch-14 models) and whether the map is exact (LINEAR_POOLED):
    ResNet-50           input of the global pooling, the output of the last bottleneck (2048 x 32 x 32); exact
    EfficientNet-B3     input of the global pooling, the 1 x 1 head convolution with BatchNorm and SiLU
                        (1536 x 32 x 32); exact
    ConvNeXt-T          input of the global pooling, the output of the last stage (768 x 32 x 32); the pretrained head
                        LayerNorm follows the pooling, so the map is not exact
    MobileNetV4-Conv-M  input of the global pooling (960 x 32 x 32); a 1 x 1 convolution, BatchNorm and ReLU follow the
                        pooling, so the map is not exact
    RoPE-ViT-B/16       patch tokens at the input of the last block (64 x 64); the head reads the class token, so the
                        map is not exact
    MambaVision-T       input of the average pooling, the output of the final BatchNorm (640 x 32 x 32); exact
    DINOv2-B/14 and     final layer-normalized patch tokens (73 x 73; class and register tokens dropped), averaged by
    RAD-DINO            the encoder as in our models; exact
    Mammo-CLIP          input of the average pooling (1408 x 32 x 32); exact

    python scripts/heatmaps.py cam --cache-root outputs/cache/vindr_1024
    python scripts/heatmaps.py cam --inbreast-manifest outputs/inbreast/manifest.csv --inbreast-root /path/to/AllDICOMs
    python scripts/heatmaps.py cam --cache-root outputs/cache/vindr_1024 --comparator-root outputs/comparators \\
        --models resnet50 efficientnet_b3 convnext_tiny mobilenetv4_conv_medium rope_vit_base mambavision_tiny \\
        dinov2_base rad_dino mammo_clip_b2
    python scripts/heatmaps.py localization --cache-root outputs/cache/vindr_1024
    python scripts/heatmaps.py faithfulness --cache-root outputs/cache/vindr_1024
    python scripts/heatmaps.py gradcam-check --cache-root outputs/cache/vindr_1024

cam           writes outputs/heatmaps/cam_<vindr|inbreast>_<model>.npz (cam, logits, image_id, method) for the four
              models, or for the networks given with --models. The comparison networks are read from
              --comparator-root/<model>/best.pt, and on VinDr-Mammo their logits are checked against the test
              predictions recorded by the training run of each checkpoint
              (--comparator-results/<model>/predictions_test.csv) before the maps are written. INbreast images are
              read from a preprocessing cache (--inbreast-cache-root) or preprocessed from DICOM (--inbreast-root).
localization  scores the maps of the three heads against the radiologists' lesion annotations of the official test
              split (results/heatmaps/lesion_masks_test.npz, rasterized in the 1024 x 1024 preprocessed frame; the
              models never saw them) with the pointing game and the top-1% precision.
              With --inbreast-manifest/--inbreast-root/--inbreast-xml it scores the screening head on INbreast instead,
              against the radiologists' contours (AllXML) of masses and calcification clusters.
faithfulness  deletion test of the calcification head and cascading model-parameter randomization.
gradcam-check  Grad-CAM versus the exact CAM of the Baseline and LSA-SSViT on the first --max-cases test images
              (Pearson r and scale; gradcam_check.json).
Results are written to results/heatmaps/; scripts/plot_heatmaps.py draws the figures.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import CLASS_NAMES, parse_birads_ge3  # noqa: E402

MODELS = ["baseline", "grain", "cusp", "LSA-SSViT"]
LABELS = {"baseline": "Baseline", "grain": "+GRAIN", "cusp": "+CUSP", "LSA-SSViT": "LSA-SSViT"}
# Comparison networks of comparators/zoo.py, mapped with Grad-CAM (in the order of the comparison table).
COMPARATOR_MODELS = ["resnet50", "efficientnet_b3", "convnext_tiny", "mobilenetv4_conv_medium", "rope_vit_base",
                     "mambavision_tiny", "dinov2_base", "rad_dino", "mammo_clip_b2"]
# Networks whose head is linear on average-pooled features: N * mean(Grad-CAM) + b reproduces the logit.
LINEAR_POOLED = {*MODELS, "resnet50", "efficientnet_b3", "mambavision_tiny", "dinov2_base", "rad_dino", "mammo_clip_b2"}
HEAD_TITLE = {"birads": "Screening", "mass": "Mass", "calcification": "Calcification"}
MASK_CHANNEL = {"mass": 0, "calcification": 1}
IMAGE_SIZE = 1024
GRID = 64
TOLERANCE_PX = 15
DELETION_FRACTIONS = np.array([0, .02, .04, .06, .08, .10, .15, .20, .30, .40, .50, .75, 1.0])
RANDOMIZATION_STAGES = ["original", "head", "block12", "block11", "block10", "block9"]
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAMs, weakly supervised localization and faithfulness checks.")
    parser.add_argument("command", choices=["cam", "localization", "faithfulness", "gradcam-check"])
    parser.add_argument("--cache-root", type=Path, default=None, help="VinDr-Mammo preprocessing cache")
    parser.add_argument("--image-root", type=Path, default=None, help="VinDr-Mammo DICOM root, used without a cache")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "splits" / "vindr" / "test.csv")
    parser.add_argument("--inbreast-manifest", type=Path, default=None,
                        help="cam and localization: work on INbreast (manifest of scripts/prepare_inbreast.py)")
    parser.add_argument("--inbreast-root", type=Path, default=None, help="INbreast DICOM folder")
    parser.add_argument("--inbreast-cache-root", type=Path, default=None,
                        help="cam: INbreast preprocessing cache (scripts/build_cache.py), used instead of --inbreast-root")
    parser.add_argument("--inbreast-xml", type=Path, default=None, help="INbreast AllXML folder (lesion contours)")
    parser.add_argument("--models", nargs="+", default=None, choices=MODELS + COMPARATOR_MODELS,
                        help="cam and gradcam-check: networks (default: the four models for cam; baseline and "
                             "LSA-SSViT for gradcam-check, which accepts only the four models)")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--comparator-root", type=Path, default=PROJECT_ROOT / "outputs" / "comparators",
                        help="checkpoints of the comparison networks (<root>/<model>/best.pt)")
    parser.add_argument("--comparator-results", type=Path, default=PROJECT_ROOT / "results" / "comparators",
                        help="test predictions recorded by the training run of each comparison-network checkpoint "
                             "(<root>/<model>/predictions_test.csv), for the logit check")
    parser.add_argument("--logit-tolerance", type=float, default=1e-2,
                        help="cam: largest accepted MEAN |logit - recorded logit| of a comparison network (its test "
                             "AUCs must also match the recorded ones within 1e-3)")
    parser.add_argument("--cam-dir", type=Path, default=PROJECT_ROOT / "outputs" / "heatmaps")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "heatmaps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None, help="gradcam-check: images (default 16)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# CAM
# -----------------------------------------------------------------------------


class CamExtractor:
    """Captures the final normalized tokens with a forward hook and returns the per-head token contributions."""

    def __init__(self, model) -> None:
        self.model = model
        self.prefix = int(model.features.num_prefix_tokens)
        self.tokens = None
        self.handle = model.features.backbone.norm.register_forward_hook(self._hook)
        linear = model.classifier[1]
        self.weight = linear.weight.detach().float()
        self.bias = linear.bias.detach().float()

    def _hook(self, module, inputs, output) -> None:
        self.tokens = output

    def __call__(self, x):
        import torch

        with torch.no_grad():
            logits = self.model(x).float()
            tokens = self.tokens[:, self.prefix:, :].float()
            cam = torch.einsum("bnc,kc->bkn", tokens, self.weight)
            error = float((cam.mean(dim=-1) + self.bias - logits).abs().max())
            side = int(round(math.sqrt(cam.shape[-1])))
            return logits, cam.reshape(cam.shape[0], cam.shape[1], side, side), error


def gradcam_site(model, name: str):
    """Module whose input ("input") or output ("output") is the feature map of Grad-CAM, and the number of prefix
    tokens to drop from a token map."""
    if name in MODELS:  # final normalized tokens: Grad-CAM equals the exact CAM divided by N
        return model.features.backbone.norm, "output", int(model.features.num_prefix_tokens)
    encoder = model.encoder
    net = encoder.net
    if name == "resnet50":  # timm ResNet: layer4 (ReLU after the residual sum) -> global_pool -> linear
        return net.global_pool, "input", 0
    if name == "efficientnet_b3":  # timm EfficientNet: bn2(conv_head) with SiLU -> global_pool -> linear
        return net.global_pool, "input", 0
    if name == "convnext_tiny":  # timm ConvNeXt: stages -> head: global_pool -> LayerNorm -> linear
        return net.head.global_pool, "input", 0
    if name == "mobilenetv4_conv_medium":  # timm MobileNetV3 class: global_pool -> conv_head -> BN/ReLU -> linear
        return net.global_pool, "input", 0
    if name == "rope_vit_base":  # timm Eva, class-token pooling: input of the last block = output of the one before
        return net.blocks[-2], "output", int(net.num_prefix_tokens)
    if name == "mambavision_tiny":  # MambaVision (Hugging Face code): levels -> norm (BatchNorm2d) -> avgpool -> linear
        return net.model.avgpool, "input", 0
    if name in ("dinov2_base", "rad_dino"):  # zoo.DinoLastBlocks: layernorm -> mean of the patch tokens -> linear
        return net.layernorm, "output", int(encoder.num_prefix_tokens)
    if name == "mammo_clip_b2":  # efficientnet_pytorch: swish(bn1(conv_head)) -> _avg_pooling -> linear
        return net._avg_pooling, "input", 0
    raise KeyError(f"No Grad-CAM feature map defined for {name!r}")


class GradCamExtractor:
    """Grad-CAM without the final ReLU, one map per head, on the feature map of gradcam_site.

    The feature map is replaced in the forward pass by a leaf tensor that requires grad, and every parameter is
    frozen, so the graph runs only from the feature map to the logits; the gradients of the three heads come from one
    forward pass. Batch items are independent in eval mode, so the gradient of the summed logits is per image.
    """

    def __init__(self, model, name: str) -> None:
        self.model = model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        module, where, self.prefix = gradcam_site(model, name)
        self.features = None
        if where == "input":
            self.handle = module.register_forward_pre_hook(self._pre_hook)
        else:
            self.handle = module.register_forward_hook(self._hook)
        # N * mean(map) + b reproduces the logit for a linear head on average-pooled features (checked).
        self.bias = model.classifier[1].bias.detach().float() if name in LINEAR_POOLED else None

    def _leaf(self, tensor):
        self.features = tensor.detach().requires_grad_(True)
        return self.features

    def _pre_hook(self, module, inputs):
        return (self._leaf(inputs[0]), *inputs[1:])

    def _hook(self, module, inputs, output):
        return self._leaf(output)

    def spatial(self, tensor):
        """B x C x H x W feature map (token maps: prefix tokens dropped, patch tokens on their square grid)."""
        if tensor.ndim == 4:
            return tensor.float()
        tokens = tensor[:, self.prefix:, :].float()
        side = int(round(math.sqrt(tokens.shape[1])))
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def __call__(self, x):
        import torch

        with torch.enable_grad():
            logits = self.model(x).float()
            heads = logits.shape[1]
            grads = [torch.autograd.grad(logits[:, k].sum(), self.features, retain_graph=k < heads - 1)[0]
                     for k in range(heads)]
        features = self.spatial(self.features.detach())
        cam = torch.stack([torch.einsum("bc,bchw->bhw", self.spatial(g).mean(dim=(2, 3)), features) for g in grads], 1)
        self.features = None
        logits = logits.detach()
        error = None
        if self.bias is not None:
            error = float((cam.mean(dim=(2, 3)) * cam[0, 0].numel() + self.bias - logits).abs().max())
        return logits, cam, error


@contextlib.contextmanager
def strict_float32():
    """Float32 matrix products and convolutions on the GPU (no TF32) for Grad-CAM and gradcam-check; restores the
    settings."""
    import torch

    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def load_model(args, name: str, device: str):
    import torch

    if name in COMPARATOR_MODELS:
        from comparators.zoo import load_comparator

        model, _ = load_comparator(args.comparator_root / name / "best.pt", torch.device(device))
        if hasattr(model.encoder.net, "set_grad_checkpointing"):  # timm: checkpointing is not needed here
            model.encoder.net.set_grad_checkpointing(False)
        return model.eval()

    from core import load_model as load_checkpoint

    model, _ = load_checkpoint(args.checkpoint_root / name / "best.pt", torch.device(device))
    return model.eval()


def recorded_logit_difference(args, name: str, ids: list, logits: np.ndarray) -> tuple[float, float, float]:
    """Largest and mean |logit - recorded test logit| of a comparison network
    (--comparator-results/<model>/predictions_test.csv), and the largest |AUC - recorded AUC| over the heads.
    Depthwise-convolution networks differ by up to a few tenths on a few extreme logits between GPU generations,
    so the check uses the mean difference and the AUCs."""
    from sklearn.metrics import roc_auc_score

    recorded = pd.read_csv(args.comparator_results / name / "predictions_test.csv", dtype={"image_id": str})
    recorded = recorded.set_index("image_id").loc[[str(i) for i in ids]]
    reference = recorded[[f"logit_{h}" for h in CLASS_NAMES]].to_numpy(dtype=np.float64)
    difference = np.abs(reference - logits)
    auc_gap = max(abs(roc_auc_score(recorded[f"target_{h}"], logits[:, k])
                      - roc_auc_score(recorded[f"target_{h}"], reference[:, k])) for k, h in enumerate(CLASS_NAMES))
    return float(difference.max()), float(difference.mean()), float(auc_gap)


def compute_cams(args) -> None:
    """Exact CAMs of our models; Grad-CAMs of the comparison networks (float32 without TF32, float32 maps)."""
    import torch
    from torch.utils.data import DataLoader

    from core import MammographyDataset, seed_everything

    seed_everything(args.seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.inbreast_manifest is not None:
        tag, dataset = "inbreast", MammographyDataset(args.inbreast_manifest, train=False,
                                                      cache_root=args.inbreast_cache_root, image_root=args.inbreast_root)
    else:
        tag, dataset = "vindr", MammographyDataset(args.test_csv, train=False, cache_root=args.cache_root,
                                                   image_root=args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    args.cam_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models or MODELS:
        model = load_model(args, name, device)
        gradcam = name in COMPARATOR_MODELS
        extractor = GradCamExtractor(model, name) if gradcam else CamExtractor(model)
        cams, logits, ids, worst = [], [], [], 0.0
        started = time.perf_counter()
        with strict_float32() if gradcam else contextlib.nullcontext():
            for batch in loader:
                batch_logits, cam, error = extractor(batch["image"].to(device))
                worst = max(worst, error or 0.0)
                cams.append(cam.cpu().numpy().astype(np.float32 if gradcam else np.float16))
                logits.append(batch_logits.cpu().numpy())
                ids.extend(batch["image_id"])
        if worst > 1e-3:
            raise RuntimeError(f"{name}: mean(CAM) + bias differs from the logit by {worst:.2e}")
        logits = np.concatenate(logits)
        recorded = None  # comparison networks on VinDr-Mammo: checked before anything is written
        if gradcam and tag == "vindr":
            recorded = recorded_logit_difference(args, name, ids, logits)
            if recorded[1] > args.logit_tolerance or recorded[2] > 1e-3:
                raise RuntimeError(f"{name}: logits differ from the recorded test predictions (mean |difference| "
                                   f"{recorded[1]:.2e}, largest AUC difference {recorded[2]:.1e})")
        np.savez(args.cam_dir / f"cam_{tag}_{name}.npz", cam=np.concatenate(cams), logits=logits,
                 image_id=np.array(ids), method=np.array("gradcam" if gradcam else "exact"))
        if not gradcam:
            print(f"{tag} {name}: {len(ids)} images, max |mean(CAM) + b - logit| = {worst:.1e} "
                  f"({time.perf_counter() - started:.0f} s)")
        else:
            checks = [f"Grad-CAM {cams[0].shape[-2]} x {cams[0].shape[-1]}"]
            if extractor.bias is not None:
                checks.append(f"max |N mean(map) + b - logit| = {worst:.1e}")
            if recorded is not None:
                checks.append(f"max |logit - recorded test logit| = {recorded[0]:.1e} (mean {recorded[1]:.1e}); "
                              f"largest |AUC - recorded AUC| = {recorded[2]:.1e}")
            print(f"{tag} {name}: {len(ids)} images, {', '.join(checks)} ({time.perf_counter() - started:.0f} s)")
        extractor.handle.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def load_cams(cam_dir: Path, tag: str = "vindr") -> dict:
    """{model: {"cam": N x 3 x 64 x 64, "logits": N x 3, "index": {image_id: row}}}."""
    data = {}
    for name in MODELS:
        with np.load(cam_dir / f"cam_{tag}_{name}.npz") as stored:
            data[name] = {"cam": stored["cam"].astype(np.float32), "logits": stored["logits"],
                          "index": {str(i): k for k, i in enumerate(stored["image_id"])}}
    return data


# -----------------------------------------------------------------------------
# Images and lesion masks
# -----------------------------------------------------------------------------


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"study_id": str, "series_id": str, "image_id": str})


def load_image(args, row) -> np.ndarray:
    from core import load_cached_image, prepare_image

    if args.cache_root is not None:
        return load_cached_image(args.cache_root, row)
    return prepare_image(row, args.image_root)


def load_lesion_masks(results_dir: Path):
    """Packed lesion masks of the positive test images, keyed by image_id; decode one with unpack_masks."""
    return np.load(results_dir / "lesion_masks_test.npz")


def unpack_masks(packed: np.ndarray) -> np.ndarray:
    """2 x 1024 x 1024 boolean masks (mass, calcification)."""
    bits = np.unpackbits(packed, axis=1, count=IMAGE_SIZE * IMAGE_SIZE, bitorder="little")
    return bits.reshape(2, IMAGE_SIZE, IMAGE_SIZE).astype(bool)


INBREAST_MASS_ROIS = {"mass", "spiculated region", "espiculated region", "distortion", "asymmetry"}
INBREAST_CLUSTER_ROIS = {"cluster"}


def preprocessing_geometry(row, image_root: Path) -> tuple[int, int, int, int, int, int, int, int]:
    """Crop box and resizing of core.prepare_image for one DICOM: (x1, y1, x2, y2, new_w, new_h, left, top)."""
    from core import (
        _largest_component_bbox,
        clean_non_primary_breast_regions,
        crop_breast_region,
        estimate_breast_mask,
        read_dicom_image,
        resolve_image_path,
    )

    image = read_dicom_image(resolve_image_path(image_root, row))
    mask = estimate_breast_mask(image)
    image, mask = clean_non_primary_breast_regions(image, mask)
    height, width = image.shape
    x1, y1, x2, y2 = 0, 0, width, height
    bbox = _largest_component_bbox(mask > 0)  # same rule as core.crop_breast_region
    if bbox is not None and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(height * width, 1) >= 0.08:
        mx, my = int(round(width * 0.02)), int(round(height * 0.02))
        box = max(0, bbox[0] - mx), max(0, bbox[1] - my), min(width, bbox[2] + mx), min(height, bbox[3] + my)
        if box[2] - box[0] >= 32 and box[3] - box[1] >= 32:
            x1, y1, x2, y2 = box
    if crop_breast_region(image, mask).shape != (y2 - y1, x2 - x1):
        raise RuntimeError("preprocessing_geometry is out of sync with core.crop_breast_region")
    scale = IMAGE_SIZE / max(y2 - y1, x2 - x1)
    new_h, new_w = max(1, int(round((y2 - y1) * scale))), max(1, int(round((x2 - x1) * scale)))
    return x1, y1, x2, y2, new_w, new_h, (IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2


def inbreast_lesion_masks(xml_path: Path, geometry) -> np.ndarray:
    """2 x 1024 x 1024 masks (masses and other focal findings; calcification clusters) from an INbreast OsiriX XML file,
    in the preprocessed frame. Individual calcifications (annotated as points or small outlines) are left out."""
    import plistlib

    import cv2

    masks = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    if not Path(xml_path).is_file():
        return masks.astype(bool)
    x1, y1, x2, y2, new_w, new_h, left, top = geometry
    sx, sy = new_w / (x2 - x1), new_h / (y2 - y1)
    with open(xml_path, "rb") as handle:
        rois = plistlib.load(handle)["Images"][0]["ROIs"]
    for roi in rois:
        name = str(roi.get("Name", "")).strip().lower()
        channel = 0 if name in INBREAST_MASS_ROIS else 1 if name in INBREAST_CLUSTER_ROIS else None
        points = [tuple(float(v) for v in p.strip("()").split(",")[:2]) for p in roi.get("Point_px", [])]
        if channel is None or len(points) < 3:
            continue
        polygon = np.array([[(x - x1) * sx + left, (y - y1) * sy + top] for x, y in points])
        cv2.fillPoly(masks[channel], [np.round(polygon).astype(np.int32)], 1)
    return masks.astype(bool)


def breast_mask(image: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mask = ndimage.binary_closing(image > 1e-4, structure=np.ones((5, 5)), iterations=1)
    return ndimage.binary_fill_holes(mask)


def breast_fraction(image: np.ndarray) -> np.ndarray:
    """Fraction of breast pixels in each of the 64 x 64 tokens (flattened)."""
    tok = IMAGE_SIZE // GRID
    return breast_mask(image).reshape(GRID, tok, GRID, tok).mean(axis=(1, 3)).ravel()


def upsample(cam: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()


# -----------------------------------------------------------------------------
# Weakly supervised localization
# -----------------------------------------------------------------------------


def lesion_geometry(lesion: np.ndarray, breast: np.ndarray, tolerance_px: int = TOLERANCE_PX):
    from scipy import ndimage

    if lesion.sum() == 0 or breast.sum() == 0:
        return None
    return {"inside": breast, "lesion": lesion & breast,
            "dilated": (ndimage.distance_transform_edt(~lesion) <= tolerance_px) & breast}


def localization_one(cam: np.ndarray, geometry) -> dict | None:
    """Pointing game (maximum within 15 px of a lesion) and top-1% precision, with their chance levels."""
    if geometry is None or geometry["lesion"].sum() == 0:
        return None
    full = upsample(cam)
    inside, lesion, dilated = geometry["inside"], geometry["lesion"], geometry["dilated"]
    y, x = np.unravel_index(int(np.argmax(np.where(inside, full, -np.inf))), full.shape)
    top = (full >= np.percentile(full[inside], 99)) & inside
    return {
        "hit": float(dilated[y, x]),
        "top1": float((top & lesion).sum() / max(top.sum(), 1)),
        "chance_hit": float(dilated.sum() / inside.sum()),
        "chance_top1": float(lesion.sum() / inside.sum()),
    }


def localization_tasks(row, lesions) -> list[str]:
    """Heads scored on one test image: mass and calcification where annotated; screening for BI-RADS 3-5 images
    whose findings are all annotated as a mass or a suspicious calcification."""
    heads = [h for h in ("mass", "calcification") if int(row[h]) == 1]
    if not heads or row["image_id"] not in lesions.files:
        return []
    other = set(json.loads(row["finding_categories_json"])) - {"Mass", "Suspicious Calcification", "No Finding"}
    if parse_birads_ge3(row["breast_birads"]) and not other:
        heads.append("birads")
    return heads


def bootstrap_mean(values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> list[float]:
    """Mean with a percentile 95% CI; exams are resampled with replacement."""
    rng = np.random.default_rng(seed)
    unique = pd.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    stats = [np.nanmean(values[np.concatenate([index[g] for g in rng.choice(unique, len(unique), replace=True)])])
             for _ in range(n_boot)]
    return [float(np.nanmean(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def summarize_localization(per_image: pd.DataFrame, n_boot: int, seed: int, unit: str = "exams") -> dict:
    summary = {"protocol": {"pointing_game_tolerance_px": TOLERANCE_PX, "top_percent": 1,
                            "bootstrap": f"{n_boot} replicates over {unit}, seed {seed}"}}
    for head in [h for h in ("birads", "mass", "calcification") if h in set(per_image["head"])]:
        part = per_image[per_image["head"] == head]
        summary[head] = {"n_images": int(part["image_id"].nunique())}
        baseline = part[part["model"] == "baseline"].set_index("image_id")
        for name in MODELS:
            rows = part[part["model"] == name].reset_index(drop=True)
            groups = rows["study_id"].to_numpy()
            cell = {metric: bootstrap_mean(rows[metric].to_numpy(dtype=float), groups, n_boot, seed) for metric in ("hit", "top1")}
            cell["chance_hit"] = float(rows["chance_hit"].mean())
            cell["chance_top1"] = float(rows["chance_top1"].mean())
            if name != "baseline":
                joined = rows.set_index("image_id").join(baseline, rsuffix="_base", how="inner").reset_index()
                for metric in ("hit", "top1"):
                    difference = (joined[metric] - joined[f"{metric}_base"]).to_numpy(dtype=float)
                    cell[f"delta_vs_baseline_{metric}"] = bootstrap_mean(difference, joined["study_id"].to_numpy(), n_boot, seed)
            summary[head][name] = cell
    return summary


def localization_table(summary: dict, unit: str = "exams") -> str:
    def fmt(t):
        return f"{t[0]:.3f} ({t[1]:.3f}–{t[2]:.3f})"

    lines = ["| Head | Model | Pointing game | Top-1% precision |", "| --- | --- | --- | --- |"]
    for head in [h for h in ("birads", "mass", "calcification") if h in summary]:
        cell = summary[head]
        for name in MODELS:
            marks = {m: "*" if name != "baseline" and cell[name][f"delta_vs_baseline_{m}"][1] > 0
                     and cell[name][m][0] > cell["baseline"][f"chance_{m}"] else "" for m in ("hit", "top1")}
            lines.append(f"| {HEAD_TITLE[head]} (n = {cell['n_images']}) | {LABELS[name]} | {fmt(cell[name]['hit'])}{marks['hit']} | "
                         f"{fmt(cell[name]['top1'])}{marks['top1']} |")
        lines.append(f"| {HEAD_TITLE[head]} | chance | {cell['baseline']['chance_hit']:.3f} | {cell['baseline']['chance_top1']:.3f} |")
    lines += ["", "Pointing game: the map's maximum within the breast lies within 15 px of an annotated lesion. "
              "Top-1% precision: share of the 1% highest-scoring breast pixels inside a lesion. "
              f"95% CI from 2,000 bootstrap replicates over {unit}; * above chance level, and the 95% CI of the paired "
              "difference versus Baseline excludes 0."]
    return "\n".join(lines) + "\n"


def inbreast_localization(args) -> None:
    """Screening head on the screening-positive INbreast images with annotated masses or calcification clusters."""
    from core import prepare_image

    if args.inbreast_root is None or args.inbreast_xml is None:
        raise SystemExit("INbreast localization needs --inbreast-root and --inbreast-xml")
    frame = pd.read_csv(args.inbreast_manifest, dtype={"study_id": str, "series_id": str, "image_id": str})
    cams = load_cams(args.cam_dir, "inbreast")
    h = CLASS_NAMES.index("birads")
    rows = []
    for _, row in frame.iterrows():
        image_id = str(row["image_id"])
        if not parse_birads_ge3(row["breast_birads"]) or image_id not in cams["baseline"]["index"]:
            continue
        masks = inbreast_lesion_masks(args.inbreast_xml / f"{image_id}.xml", preprocessing_geometry(row, args.inbreast_root))
        if not masks.any():
            continue
        geometry = lesion_geometry(masks[0] | masks[1], breast_mask(prepare_image(row, args.inbreast_root)))
        for name in MODELS:
            record = localization_one(cams[name]["cam"][cams[name]["index"][image_id], h], geometry)
            if record is not None:
                rows.append({"model": name, "head": "birads", "image_id": image_id, "study_id": row["study_id"], **record})
    per_image = pd.DataFrame(rows)
    summary = summarize_localization(per_image, args.n_boot, args.seed, unit="patients")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.results_dir / "inbreast_localization_per_image.csv", index=False, lineterminator="\n")
    (args.results_dir / "inbreast_localization_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                         encoding="utf-8", newline="\n")
    table = localization_table(summary, unit="patients")
    (args.results_dir / "inbreast_localization_table.md").write_text(table, encoding="utf-8", newline="\n")
    print(table)


def localization(args) -> None:
    if args.inbreast_manifest is not None:
        inbreast_localization(args)
        return
    test = read_split(args.test_csv)
    cams = load_cams(args.cam_dir)
    lesions = load_lesion_masks(args.results_dir)
    rows = []
    for _, row in test.iterrows():
        heads = localization_tasks(row, lesions)
        if not heads:
            continue
        masks = unpack_masks(lesions[row["image_id"]])
        breast = breast_mask(load_image(args, row))
        for head in heads:
            lesion = masks[0] | masks[1] if head == "birads" else masks[MASK_CHANNEL[head]]
            geometry = lesion_geometry(lesion, breast)
            for name in MODELS:
                k = cams[name]["index"][str(row["image_id"])]
                record = localization_one(cams[name]["cam"][k, CLASS_NAMES.index(head)], geometry)
                if record is not None:
                    rows.append({"model": name, "head": head, "image_id": row["image_id"], "study_id": row["study_id"], **record})
    per_image = pd.DataFrame(rows)
    summary = summarize_localization(per_image, args.n_boot, args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.results_dir / "localization_per_image.csv", index=False, lineterminator="\n")
    (args.results_dir / "localization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    table = localization_table(summary)
    (args.results_dir / "localization_table.md").write_text(table, encoding="utf-8", newline="\n")
    print(table)


# -----------------------------------------------------------------------------
# Faithfulness
# -----------------------------------------------------------------------------


def faithfulness_subset(test: pd.DataFrame, cams: dict, n_per_head: int) -> list:
    """Positive test images of the mass and calcification heads with the highest mean logit over the four models."""
    chosen = []
    for head in ("mass", "calcification"):
        positives = test[test[head].astype(int) == 1].copy()
        h = CLASS_NAMES.index(head)
        positives["score"] = [np.mean([cams[m]["logits"][cams[m]["index"][i], h] for m in MODELS]) for i in positives["image_id"]]
        chosen += [(head, row) for _, row in positives.sort_values("score", ascending=False).head(n_per_head).iterrows()]
    return chosen


def to_input(image: np.ndarray):
    import torch

    return ((torch.from_numpy(image).float() - 0.5) / 0.25)[None, None]


def deletion(args, test, cams, device) -> pd.DataFrame:
    """Remove breast tokens in decreasing CAM order or in random order and follow the calcification probability."""
    import torch

    h = CLASS_NAMES.index("calcification")
    tok = IMAGE_SIZE // GRID
    rows = []
    for name in MODELS:
        model = load_model(args, name, device)
        for n_case, (head, row) in enumerate(faithfulness_subset(test, cams, 40)):
            if head != "calcification":
                continue
            image = load_image(args, row)
            cam = cams[name]["cam"][cams[name]["index"][str(row["image_id"])], h]
            tokens = np.flatnonzero(breast_fraction(image) > 0)
            orders = {"cam": tokens[np.argsort(-cam.ravel()[tokens])],
                      "random": np.random.default_rng([args.seed, n_case]).permutation(tokens)}
            for order_name, order in orders.items():
                images = []
                for fraction in DELETION_FRACTIONS:
                    removed = np.zeros(GRID * GRID, bool)
                    removed[order[: int(round(fraction * len(order)))]] = True
                    removed = np.kron(removed.reshape(GRID, GRID), np.ones((tok, tok), bool))
                    images.append(np.where(removed, 0.0, image).astype(np.float32))
                with torch.no_grad():
                    x = torch.cat([to_input(img) for img in images]).to(device)
                    curve = torch.sigmoid(model(x)[:, h]).float().cpu().numpy()
                rows.append({"model": name, "image_id": row["image_id"], "order": order_name,
                             "auc": float(trapezoid(curve, DELETION_FRACTIONS)),
                             "curve": json.dumps([round(float(c), 5) for c in curve])})
        del model
    return pd.DataFrame(rows)


def gradcam_check(args) -> None:
    """Grad-CAM versus the exact CAM on the first --max-cases test images (default 16): for our models the two maps
    must agree up to the positive scale 1 / N (N = 64 x 64 tokens)."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from core import MammographyDataset, seed_everything

    seed_everything(args.seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MammographyDataset(args.test_csv, train=False, cache_root=args.cache_root, image_root=args.image_root)
    dataset = Subset(dataset, range(min(args.max_cases or 16, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    report = {}
    for name in args.models or ["baseline", "LSA-SSViT"]:
        if name not in MODELS:
            raise SystemExit(f"gradcam-check compares with the exact CAM of our models; {name} has none")
        model = load_model(args, name, device)
        maps = {}
        with strict_float32():  # both methods at the same precision, so that only the method differs
            for method in ("exact", "gradcam"):
                extractor = CamExtractor(model) if method == "exact" else GradCamExtractor(model, name)
                cams, logits = [], []
                for batch in loader:
                    batch_logits, cam, _ = extractor(batch["image"].to(device))
                    cams.append(cam.float().cpu().numpy())
                    logits.append(batch_logits.cpu().numpy())
                extractor.handle.remove()
                maps[method] = (np.concatenate(cams), np.concatenate(logits))
        exact, gradcam = (maps[m][0].reshape(len(dataset), len(CLASS_NAMES), -1).astype(np.float64) for m in maps)
        r = np.array([[np.corrcoef(exact[i, k], gradcam[i, k])[0, 1] for k in range(exact.shape[1])]
                      for i in range(exact.shape[0])])
        scale = (exact * gradcam).sum(-1) / (exact * exact).sum(-1)  # least squares through the origin
        residual = np.abs(gradcam - scale[..., None] * exact).max(-1) / np.abs(gradcam).max(-1)
        report[name] = {"n_images": len(dataset), "pearson_r_min": float(r.min()), "pearson_r_mean": float(r.mean()),
                        "scale_mean": float(scale.mean()), "scale_min": float(scale.min()), "scale_max": float(scale.max()),
                        "scale_expected": 1.0 / exact.shape[-1], "max_relative_residual": float(residual.max()),
                        "max_abs_logit_difference": float(np.abs(maps["exact"][1] - maps["gradcam"][1]).max())}
        print(f"{name}: Pearson r min {r.min():.6f}, mean {r.mean():.6f}; scale {scale.mean():.4e} "
              f"(range {scale.min():.4e}-{scale.max():.4e}, 1 / N = {1.0 / exact.shape[-1]:.4e}); "
              f"max |residual| / max |Grad-CAM| = {residual.max():.1e}")
        if r.min() < 0.999 or np.abs(scale * exact.shape[-1] - 1).max() > 1e-2:
            raise RuntimeError(f"{name}: Grad-CAM does not reproduce the exact CAM up to scale")
        del model
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "gradcam_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


def randomization(args, test, cams, device) -> pd.DataFrame:
    """Cascading model-parameter randomization (Adebayo et al., 2018): the head, then blocks 12, 11, 10 and 9."""
    import torch
    from scipy import stats

    from core import backbone_layers

    subset = faithfulness_subset(test, cams, 10)
    rows = []
    for name in ("baseline", "LSA-SSViT"):
        model = load_model(args, name, device)
        torch.manual_seed(args.seed)
        layers = backbone_layers(model.features.backbone)
        reference = {}
        for stage in RANDOMIZATION_STAGES:
            with torch.no_grad():
                if stage == "head":
                    torch.nn.init.normal_(model.classifier[1].weight, std=0.02)
                    torch.nn.init.zeros_(model.classifier[1].bias)
                elif stage.startswith("block"):
                    block = layers[int(stage[5:]) - 1]
                    for module in block.modules():
                        if isinstance(module, torch.nn.Linear):
                            torch.nn.init.normal_(module.weight, std=0.02)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
                        elif isinstance(module, torch.nn.LayerNorm):
                            torch.nn.init.ones_(module.weight)
                            torch.nn.init.zeros_(module.bias)
                    for parameter_name, parameter in block.named_parameters():
                        if parameter_name.endswith("lambda1"):
                            parameter.fill_(float(getattr(model.features.backbone.config, "layerscale_value", 1.0)))
                    if model.grain is not None:
                        index = int(stage[5:]) - 1 - (len(layers) - len(model.grain))
                        if index >= 0:
                            for parameter in model.grain[index].parameters():
                                torch.nn.init.normal_(parameter, std=0.02)
            extractor = CamExtractor(model)
            for head, row in subset:
                image = load_image(args, row)
                _, cam, _ = extractor(to_input(image).to(device))
                values = cam[0, CLASS_NAMES.index(head)].float().cpu().numpy().ravel()[breast_fraction(image) >= 0.5]
                key = (head, row["image_id"])
                if stage == "original":
                    reference[key] = values
                else:
                    rows.append({"model": name, "stage": stage, "head": head, "image_id": row["image_id"],
                                 "spearman_abs": float(stats.spearmanr(np.abs(reference[key]), np.abs(values)).statistic)})
            extractor.handle.remove()
        del model
    return pd.DataFrame(rows)


def summarize_faithfulness(deleted: pd.DataFrame, randomized: pd.DataFrame) -> dict:
    return {
        "deletion_auc_calcification": {f"{m}/{o}": round(float(v), 4)
                                       for (m, o), v in deleted.groupby(["model", "order"])["auc"].mean().items()},
        "randomization_abs_spearman": {f"{m}/{s}": round(float(v), 4)
                                       for (m, s), v in randomized.groupby(["model", "stage"])["spearman_abs"].mean().items()},
        "n_images": {"deletion": int(deleted["image_id"].nunique()), "randomization": int(randomized["image_id"].nunique())},
    }


def faithfulness(args) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = read_split(args.test_csv)
    cams = load_cams(args.cam_dir)
    deleted = deletion(args, test, cams, device)
    randomized = randomization(args, test, cams, device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    deleted.to_csv(args.results_dir / "faithfulness_deletion.csv", index=False, lineterminator="\n")
    randomized.to_csv(args.results_dir / "faithfulness_randomization.csv", index=False, lineterminator="\n")
    summary = summarize_faithfulness(deleted, randomized)
    (args.results_dir / "faithfulness_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.models and args.command not in ("cam", "gradcam-check"):
        raise SystemExit("--models applies to cam and gradcam-check; localization and faithfulness use the four models")
    if args.command == "gradcam-check" and set(args.models or []) - set(MODELS):
        raise SystemExit("gradcam-check compares Grad-CAM with the exact maps of the four models only")
    if args.command == "cam":
        compute_cams(args)
    elif args.command == "localization":
        localization(args)
    elif args.command == "gradcam-check":
        gradcam_check(args)
    else:
        faithfulness(args)


if __name__ == "__main__":
    main()
