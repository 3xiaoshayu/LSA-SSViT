#!/usr/bin/env python
"""Parameters and compute of the model with and without GRAIN.

    python scripts/model_complexity.py --checkpoint checkpoints/LSA-SSViT/best.pt
    python scripts/model_complexity.py            # reads the DINOv3 config from Hugging Face instead

Parameters are counted exactly. Multiply-accumulate operations (MACs) for one
1024 x 1024 image are measured with torch.utils.flop_counter on the CPU (eager attention,
no gradients) and cross-checked against the closed form of GRAIN:
    parameters per operator  49 C + C^2 + 1
    MACs per operator        (49 C + C^2) * H * W   with H = W = 64 patch positions.
CUSP changes only the training loss and adds nothing at inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import DEFAULT_BACKBONE, IMAGE_SIZE, BreastClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count parameters and MACs.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="take the backbone configuration from here")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    return parser.parse_args()


def build(grain: bool, args: argparse.Namespace) -> BreastClassifier:
    if args.checkpoint is not None:
        config = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]["backbone_config"]
        model = BreastClassifier(grain=grain, pretrained=False, backbone_config=config)
    else:
        model = BreastClassifier(grain=grain, backbone=args.backbone, pretrained=False)
    # Eager attention makes the attention matrix products visible to the FLOP counter.
    model.features.backbone.config._attn_implementation = "eager"
    if getattr(model.features.backbone.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Could not switch the backbone to eager attention; MACs would be undercounted.")
    return model.eval()


def closed_form_backbone_macs(model: BreastClassifier) -> float:
    """Patch embedding, attention (QKV, scores, weighted sum, output) and MLP of all blocks."""
    config = model.features.backbone.config
    dim, hidden = int(config.hidden_size), int(config.intermediate_size)
    patch = int(config.patch_size)
    patches = (IMAGE_SIZE // patch) ** 2
    tokens = patches + model.features.num_prefix_tokens
    mlp = (3 if getattr(config, "use_gated_mlp", False) else 2) * tokens * dim * hidden
    block = 4 * tokens * dim * dim + 2 * tokens * tokens * dim + mlp
    return patches * 3 * patch * patch * dim + int(config.num_hidden_layers) * block


def measure_macs(model: BreastClassifier) -> float:
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE))
    return counter.get_total_flops() / 2.0  # FlopCounterMode counts 2 FLOPs per multiply-accumulate


def main() -> None:
    args = parse_args()
    results = {}
    for grain in (False, True):
        model = build(grain, args)
        backbone = sum(p.numel() for p in model.features.parameters())
        grain_params = sum(p.numel() for p in model.grain.parameters()) if model.grain is not None else 0
        results[grain] = {
            "backbone": backbone,
            "grain": grain_params,
            "total": sum(p.numel() for p in model.parameters()),
            "macs": measure_macs(model),
            "closed_form": closed_form_backbone_macs(model),
            "dim": int(model.features.backbone.config.hidden_size),
            "blocks": len(model.grain) if model.grain is not None else 0,
        }
        del model

    base, full = results[False], results[True]
    dim, blocks = full["dim"], full["blocks"]
    grid = (IMAGE_SIZE // 16) ** 2
    closed_params = blocks * (49 * dim + dim * dim + 1)
    closed_macs = blocks * (49 * dim + dim * dim) * grid
    print(f"Backbone parameters:            {base['backbone']:,}")
    print(f"GRAIN parameters:               {full['grain']:,} ({blocks} operators; closed form {closed_params:,})")
    print(f"GRAIN / backbone parameters:    {100 * full['grain'] / base['backbone']:.2f}%")
    print(f"Model parameters without/with:  {base['total']:,} / {full['total']:,}")
    print(f"GMACs per image without GRAIN:  {base['macs'] / 1e9:.1f} "
          f"(closed form of the backbone {base['closed_form'] / 1e9:.1f})")
    if abs(base["macs"] / base["closed_form"] - 1.0) > 0.05:
        print("WARNING: measured and closed-form backbone MACs differ by more than 5%; check the counter.")
    print(f"GMACs per image with GRAIN:     {full['macs'] / 1e9:.1f} "
          f"(+{(full['macs'] - base['macs']) / 1e9:.2f}, +{100 * (full['macs'] / base['macs'] - 1):.2f}%; "
          f"closed form of GRAIN {closed_macs / 1e9:.2f})")


if __name__ == "__main__":
    main()
