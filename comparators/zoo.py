"""Comparator networks of the whole-model comparison.

Every comparator is re-trained with exactly the protocol of train.py: the same splits, the
same 1024 x 1024 preprocessing and augmentation, a linear three-head classifier (dropout
0.25) on the pooled features, class-balanced BCE, AdamW (weight decay 1e-4), batch size 8,
the same learning-rate schedule, early stopping on the validation screening AUC and
validation-selected thresholds. Only the network, its pretrained weights, its input
normalisation, the backbone learning rate and, for MambaVision-T, the autocast precision change:

    regime "full"   every parameter is trained;
    regime "last4"  foundation models with the recipe of LSA-SSViT: the last four
                    transformer blocks and the final LayerNorm are trained and the final
                    patch tokens are averaged.

The backbone learning rate of each network (COMPARATORS below) was selected on the
validation split; the head learning rate is 2e-4 for every model, as for LSA-SSViT. The
grey-scale input is repeated to three channels and standardised with the statistics of the
pretrained weights. Patch-14 transformers see the central 1022 x 1022 pixels (one pixel is
cropped on each side, so that the image is not resampled).

Pooled features are the network's own pre-logits: global average pooling (followed by the
pretrained head LayerNorm for ConvNeXt-T and by the pretrained 1 x 1 head convolution,
960 -> 1280, with BatchNorm and ReLU for MobileNetV4-Conv-M), the class token for
RoPE-ViT-B/16, and the mean of the final patch tokens for the "last4" models. Each network
keeps its default stochastic depth (MambaVision-T and Mammo-CLIP 0.2, as in their own
fine-tuning recipes; none for the others and for LSA-SSViT).
Mixed precision is float16 autocast with a GradScaler, as for LSA-SSViT; MambaVision-T
uses bfloat16 autocast, because its activations exceed the float16 range. The rotary angles of
RoPE-ViT-B/16 are computed in float32, as in LSA-SSViT.
Gradient checkpointing (needed to fit 1024 px into 24 GB) recomputes each block in the
backward pass; the momentum of the BatchNorm layers inside those blocks is set to
1 - sqrt(1 - m), so that their two updates per step equal one update with momentum m.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from core import IMAGE_SIZE, UNFREEZE_LAST_BLOCKS, preprocessing_config
from metrics import CLASS_NAMES

HEAD_LR = 2e-4
COMPARATOR_CHECKPOINT_FORMAT = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MAMMO_CLIP_REPO = "shawn24/Mammo-CLIP"


@dataclass(frozen=True)
class MammoClipVariant:
    """Released image-text checkpoint of the Mammo-CLIP code base whose image encoder is a comparator."""

    repository: str    # Hugging Face repository
    file: str          # checkpoint in the repository
    encoder_name: str  # config["model"]["image_encoder"]["name"] stored in the checkpoint
    architecture: str  # efficientnet_pytorch model that this encoder name stands for in the Mammo-CLIP code
    weights: str       # extracted image encoder in --weights-dir (comparators/prepare_weights.py)

    @property
    def source(self) -> str:
        return f"{self.repository}/{self.file}"


# Keyed by comparator. 'tf_efficientnetv2-detect' is EfficientNet-B2 in Mammo-CLIP's
# breastclip/model/modules/__init__.py, despite its name.
MAMMO_CLIP_VARIANTS: dict[str, MammoClipVariant] = {
    "mammo_clip_b2": MammoClipVariant(MAMMO_CLIP_REPO, "Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
                                      "tf_efficientnetv2-detect", "efficientnet-b2",
                                      "mammo_clip_b2_image_encoder.pt"),
}


@dataclass(frozen=True)
class Comparator:
    key: str
    title: str
    family: str
    reference: str
    loader: str  # "timm", "hf_dino", "hf_mambavision" or "mammo_clip"
    source: str  # timm model name, Hugging Face repository or (mammo_clip) repository/checkpoint
    backbone_lr: float
    regime: str = "full"
    input_size: int = IMAGE_SIZE
    grad_checkpointing: bool = False
    channels_last: bool = False
    timm_kwargs: tuple[tuple[str, Any], ...] = ()
    normalization: tuple[float, float] | None = None  # (mean, std) for every channel; None: from the weights
    amp_dtype: str = "float16"  # autocast precision during training


COMPARATORS: dict[str, Comparator] = {spec.key: spec for spec in [
    Comparator("resnet50", "ResNet-50", "CNN", "He et al., CVPR 2016",
               "timm", "resnet50.a1_in1k", 1e-4, channels_last=True),
    Comparator("efficientnet_b3", "EfficientNet-B3", "CNN", "Tan and Le, ICML 2019",
               "timm", "efficientnet_b3.ra2_in1k", 1e-4, grad_checkpointing=True, channels_last=True),
    Comparator("convnext_tiny", "ConvNeXt-T", "CNN", "Liu et al., CVPR 2022",
               "timm", "convnext_tiny.fb_in22k_ft_in1k", 3e-6, grad_checkpointing=True, channels_last=True),
    Comparator("mobilenetv4_conv_medium", "MobileNetV4-Conv-M", "CNN", "Qin et al., ECCV 2024",
               "timm", "mobilenetv4_conv_medium.e500_r256_in1k", 1e-4, grad_checkpointing=True, channels_last=True),
    # Rotary position embedding only (no absolute one): the 64 x 64 grid at 1024 px needs no resampling.
    Comparator("rope_vit_base", "RoPE-ViT-B/16", "Transformer", "Heo et al., ECCV 2024",
               "timm", "vit_base_patch16_rope_mixed_224.naver_in1k", 1e-5, grad_checkpointing=True,
               timm_kwargs=(("img_size", IMAGE_SIZE),)),
    Comparator("mambavision_tiny", "MambaVision-T", "Hybrid", "Hatamizadeh and Kautz, CVPR 2025",
               "hf_mambavision", "nvidia/MambaVision-T-1K", 1e-5, grad_checkpointing=True, amp_dtype="bfloat16"),
    Comparator("dinov2_base", "DINOv2-B/14", "Foundation model", "Oquab et al., TMLR 2024; Darcet et al., ICLR 2024",
               "hf_dino", "facebook/dinov2-with-registers-base", 1e-4, regime="last4", input_size=1022),
    Comparator("rad_dino", "RAD-DINO", "Medical foundation model", "Perez-Garcia et al., Nat. Mach. Intell. 2025",
               "hf_dino", "microsoft/rad-dino", 1e-5, regime="last4", input_size=1022),
    Comparator("mammo_clip_b2", "Mammo-CLIP (EfficientNet-B2)", "Mammography foundation model",
               "Ghosh et al., MICCAI 2024", "mammo_clip", MAMMO_CLIP_VARIANTS["mammo_clip_b2"].source, 5e-5,
               grad_checkpointing=True, channels_last=True, normalization=(0.3089279, 0.25053555408335154)),
]}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def checkpoint_blocks(blocks: Iterable[nn.Module]) -> int:
    """Recompute the activations of each block in the backward pass (non-reentrant checkpointing).

    The forward method of each block is wrapped in place, so the parameter names do not change.
    """
    count = 0
    for block in blocks:
        def forward(*args, _forward=block.forward, _block=block, **kwargs):
            if _block.training and torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        block.forward = forward
        count += 1
    return count


def compensate_batchnorm_momentum(modules: Iterable[nn.Module]) -> int:
    """BatchNorm layers inside checkpointed blocks run twice per step (forward and recomputation),
    each time updating their running statistics with the same batch. Momentum 1 - sqrt(1 - m) makes
    the two updates equal one update with the original momentum m."""
    count = 0
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.momentum:
                layer.momentum = 1.0 - (1.0 - float(layer.momentum)) ** 0.5
                count += 1
    return count


def block_output(value: Any) -> torch.Tensor:
    """Hidden states returned by a Hugging Face transformer block (tensor, tuple or output object)."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    hidden = getattr(value, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    raise RuntimeError(f"Unexpected output type from a transformer block: {type(value)}")


def hf_image_statistics(repository: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean and standard deviation of the image processor of a Hugging Face model."""
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(repository, "preprocessor_config.json")).read_text(encoding="utf-8"))
    return tuple(float(v) for v in config["image_mean"]), tuple(float(v) for v in config["image_std"])


def _plain(value: Any) -> Any:
    """OmegaConf containers -> plain Python objects (for the Mammo-CLIP checkpoint configuration)."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=False)
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def _config_entries(config: Any, keys: tuple[str, ...], path: str = "") -> dict[str, Any]:
    """Every entry of a nested configuration whose key is in keys, by dotted path (e.g. {"base.mean": 0.31})."""
    found: dict[str, Any] = {}
    items = (config.items() if isinstance(config, dict)
             else enumerate(config) if isinstance(config, list) else ())
    for key, value in items:
        name = f"{path}.{key}" if path else str(key)
        if str(key).lower() in keys:
            found[name] = value
        found.update(_config_entries(value, keys, name))
    return found


def extract_mammo_clip_encoder(tar_path: Path, output_path: Path, key: str) -> dict[str, Any]:
    """Keep only the image encoder of a released Mammo-CLIP-type checkpoint (MAMMO_CLIP_VARIANTS[key]),
    with its training-data record and the input statistics stored in its configuration."""
    variant = MAMMO_CLIP_VARIANTS[key]
    try:
        checkpoint_data = torch.load(tar_path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # not a zip-format file: load it fully
        checkpoint_data = torch.load(tar_path, map_location="cpu", weights_only=False)
    config = _plain(checkpoint_data["config"])
    encoder = (config.get("model") or {}).get("image_encoder") or {}
    record = {
        "source": variant.source,
        "architecture": variant.architecture,
        "image_encoder": encoder,
        "data_train": config.get("data_train"),
        # Input statistics and size wherever the configuration stores them (dotted path -> value).
        "mean": _config_entries(config, ("mean",)),
        "std": _config_entries(config, ("std",)),
        "image_size": _config_entries(config, ("image_size", "img_size", "image_size_h", "image_size_w")),
        "epoch": _plain(checkpoint_data.get("epoch")),
    }
    summary = {name: record[name] for name in ("image_encoder", "data_train", "mean", "std", "image_size")}
    if str(encoder.get("name")).lower() != variant.encoder_name:
        raise ValueError(f"{key}: image encoder {encoder.get('name')!r}, expected {variant.encoder_name!r}; "
                         f"checkpoint configuration: {summary}")
    if record["data_train"] is None:
        raise ValueError(f"{key}: no data_train in the checkpoint configuration (keys {sorted(config)}), "
                         "so the pre-training data cannot be checked")
    if "vindr" in str(record["data_train"]).lower():  # VinDr-Mammo is our training and test data
        raise ValueError(f"{key}: pre-trained on VinDr-Mammo: data_train = {record['data_train']}")
    state = {name[len("image_encoder."):]: value.clone() for name, value in checkpoint_data["model"].items()
             if name.startswith("image_encoder.")}
    if not state:
        raise ValueError(f"{key}: no image_encoder.* weights in {tar_path}")
    digest = hashlib.sha256()
    with tar_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    record["source_sha256"] = digest.hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, **record}, output_path)
    return record


# -----------------------------------------------------------------------------
# Encoders: forward(x) returns pooled features (B x out_dim)
# -----------------------------------------------------------------------------


class TimmEncoder(nn.Module):
    def __init__(self, spec: Comparator, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.net = timm.create_model(spec.source, pretrained=pretrained, num_classes=0, **dict(spec.timm_kwargs))
        config = self.net.pretrained_cfg
        self.mean, self.std = tuple(config["mean"]), tuple(config["std"])
        self.out_dim = int(getattr(self.net, "head_hidden_size", None) or self.net.num_features)
        rope = getattr(self.net, "rope", None)
        if rope is not None and hasattr(rope, "get_embed"):
            # Rotary position embedding (RoPE-ViT-B/16): the angles (grid coordinate, up to 63 at 1024 px,
            # times the learned frequency) reach tens of radians, and timm computes them with a matmul that
            # autocast would run in half precision. They are computed in float32, as in the DINOv3 backbone;
            # parameter names do not change.
            def get_embed(*args, _get_embed=rope.get_embed, _rope=rope, **kwargs):
                device = next(iter([*_rope.parameters(), *_rope.buffers()])).device
                with torch.autocast(device_type=device.type, enabled=False):
                    return _get_embed(*args, **kwargs)

            rope.get_embed = get_embed
        if spec.grad_checkpointing:
            self.net.set_grad_checkpointing(True)
            # The containers that timm runs through checkpoint_seq (blocks, stages, layers or layer1-4).
            names = ("blocks", "stages", "layers", "layer1", "layer2", "layer3", "layer4")
            compensate_batchnorm_momentum(getattr(self.net, name) for name in names
                                          if isinstance(getattr(self.net, name, None), nn.Module))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoLastBlocks(nn.Module):
    """Hugging Face DINOv2-type model; the last blocks and the final LayerNorm are trained, patch tokens averaged."""

    def __init__(self, spec: Comparator, pretrained: bool, trainable_blocks: int = UNFREEZE_LAST_BLOCKS) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.net = (AutoModel.from_pretrained(spec.source) if pretrained
                    else AutoModel.from_config(AutoConfig.from_pretrained(spec.source)))
        config = self.net.config
        self.num_prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0) or 0)
        self.out_dim = int(config.hidden_size)
        self.mean, self.std = hf_image_statistics(spec.source)
        layers = self.net.encoder.layer
        self.cut = len(layers) - int(trainable_blocks)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        for module in [*layers[self.cut:], self.net.layernorm]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = self.net.encoder.layer
        with torch.no_grad():  # frozen part: no activations are kept
            hidden = self.net.embeddings(x)
            for layer in layers[: self.cut]:
                hidden = block_output(layer(hidden))
        for layer in layers[self.cut:]:
            hidden = block_output(layer(hidden))
        hidden = self.net.layernorm(hidden)
        return hidden[:, self.num_prefix_tokens:, :].mean(dim=1)


class MambaVisionEncoder(nn.Module):
    def __init__(self, spec: Comparator, pretrained: bool) -> None:
        super().__init__()
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # The repository's model class predates transformers 5, whose from_pretrained() it does not
        # support; the class is built directly and the weights are loaded strictly.
        config = AutoConfig.from_pretrained(spec.source, trust_remote_code=True)
        model_class = get_class_from_dynamic_module("modeling_mambavision.MambaVisionModel", spec.source)
        self.net = model_class(config)
        if pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            self.net.load_state_dict(load_file(hf_hub_download(spec.source, "model.safetensors")), strict=True)
        body = self.net.model
        self.out_dim = int(body.norm.num_features)  # width of the final BatchNorm before pooling
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        if spec.grad_checkpointing:
            blocks = [block for level in body.levels for block in level.blocks]
            checkpoint_blocks(blocks)
            compensate_batchnorm_momentum(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        pooled = output[0] if isinstance(output, (tuple, list)) else output
        return pooled.flatten(1)


class MammoClipEncoder(nn.Module):
    """Image encoder of Mammo-CLIP: an EfficientNet (efficientnet_pytorch layout, the
    architecture of MAMMO_CLIP_VARIANTS) with global average pooling."""

    def __init__(self, spec: Comparator, pretrained: bool, weights_dir: Path | None) -> None:
        super().__init__()
        if spec.normalization is None:  # these encoders take their input statistics from the spec only
            raise ValueError(f"{spec.key}: normalization is not set in comparators/zoo.py; enter the mean and "
                             "std printed by comparators/prepare_weights.py")
        from efficientnet_pytorch import EfficientNet

        variant = MAMMO_CLIP_VARIANTS[spec.key]
        self.net = EfficientNet.from_name(variant.architecture, num_classes=1)
        self.net._fc = nn.Identity()  # the released encoder has no final linear layer
        if pretrained:
            path = Path(weights_dir or "") / variant.weights
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found; run comparators/prepare_weights.py first")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.net.load_state_dict(payload["state_dict"], strict=True)
        self.out_dim = int(self.net._conv_head.out_channels)
        self.mean = self.std = None  # set from spec.normalization
        self.checkpointing = bool(spec.grad_checkpointing)
        if self.checkpointing:
            compensate_batchnorm_momentum(self.net._blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        x = net._swish(net._bn0(net._conv_stem(x)))
        for index, block in enumerate(net._blocks):
            rate = net._global_params.drop_connect_rate
            if rate:
                rate *= float(index) / len(net._blocks)
            if self.checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, rate, use_reentrant=False)
            else:
                x = block(x, rate)
        x = net._swish(net._bn1(net._conv_head(x)))
        return net._avg_pooling(x).flatten(1)


def build_encoder(spec: Comparator, pretrained: bool, weights_dir: Path | None) -> nn.Module:
    if spec.loader == "timm":
        return TimmEncoder(spec, pretrained)
    if spec.loader == "hf_dino":
        return DinoLastBlocks(spec, pretrained)
    if spec.loader == "hf_mambavision":
        return MambaVisionEncoder(spec, pretrained)
    if spec.loader == "mammo_clip":
        return MammoClipEncoder(spec, pretrained, weights_dir)
    raise ValueError(f"Unknown loader {spec.loader!r}")


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------


class ComparatorClassifier(nn.Module):
    """Comparator encoder + the linear three-head classifier of LSA-SSViT (dropout, linear layer).

    Takes the 1 x H x W tensor of MammographyDataset, like BreastClassifier.
    """

    def __init__(self, key: str, pretrained: bool = True, dropout: float = 0.25,
                 num_classes: int = len(CLASS_NAMES), weights_dir: Path | None = None) -> None:
        super().__init__()
        if key not in COMPARATORS:
            raise KeyError(f"Unknown comparator {key!r}; choose from {sorted(COMPARATORS)}")
        self.spec = COMPARATORS[key]
        self.encoder = build_encoder(self.spec, pretrained, weights_dir)
        if self.spec.normalization is not None:
            mean, std = (self.spec.normalization[0],) * 3, (self.spec.normalization[1],) * 3
        else:
            mean, std = self.encoder.mean, self.encoder.std
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        # Construction order (encoder, classifier) fixes the random initialisation of the head.
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, num_classes))
        if self.spec.channels_last:
            self.encoder.to(memory_format=torch.channels_last)

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected a B x 1 x H x W tensor, got {tuple(x.shape)}")
        raw = (x.float() * 0.25 + 0.5).clamp(0.0, 1.0)
        size = self.spec.input_size
        top, left = (raw.shape[-2] - size) // 2, (raw.shape[-1] - size) // 2
        if top < 0 or left < 0:
            raise ValueError(f"Input {tuple(raw.shape[-2:])} is smaller than {size} x {size}")
        if top or left:
            raw = raw[..., top: top + size, left: left + size]
        rgb = (raw.expand(-1, 3, -1, -1) - self.input_mean) / self.input_std
        return rgb.contiguous(memory_format=torch.channels_last) if self.spec.channels_last else rgb

    def forward(self, x: torch.Tensor, return_features: bool = False):
        pooled = self.encoder(self.prepare_input(x))
        logits = self.classifier(pooled)
        return (logits, pooled) if return_features else logits

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Trainable parameters of the encoder and of the classifier."""
        backbone = [p for p in self.encoder.parameters() if p.requires_grad]
        head = [p for p in self.classifier.parameters() if p.requires_grad]
        return backbone, head


def comparator_config(model: ComparatorClassifier, **extra: Any) -> dict[str, Any]:
    """Configuration stored in comparator checkpoints."""
    return {
        "comparator": model.spec.key,
        "spec": asdict(model.spec),
        "dropout": float(model.classifier[0].p),
        "class_names": list(CLASS_NAMES),
        "preprocessing": preprocessing_config(),
        **extra,
    }


def load_comparator(checkpoint_path: Path, device: torch.device,
                    weights_dir: Path | None = None) -> tuple[ComparatorClassifier, dict[str, Any]]:
    """Rebuild a trained comparator from a checkpoint written by comparators/train_comparator.py."""
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint_data.get("format_version") != COMPARATOR_CHECKPOINT_FORMAT:
        raise ValueError(f"{checkpoint_path} is not a comparator checkpoint.")
    config = checkpoint_data["config"]
    model = ComparatorClassifier(config["comparator"], pretrained=False, dropout=float(config["dropout"]),
                                 weights_dir=weights_dir)
    model.load_state_dict(checkpoint_data["model_state"], strict=True)
    return model.to(device).eval(), checkpoint_data
