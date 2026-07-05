#!/usr/bin/env python
"""Local smoke test: generate one 512x512 image and print its metadata (SPEC decision D11).

This is the single sanctioned local generation — CPU fp32 on an 8 GB M1 takes minutes. It
is a standalone script that pytest never collects, so the Stop-hook gate can never trigger
it. LoRA-free by design (the registry is empty until license verification), which also
proves the base-model path.

Usage (pin CPU to stay off the marginal MPS memory path):
    VOXELCRAFT_DEVICE=cpu venv/bin/python scripts/smoke_test.py --steps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make `src` importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.sd_pipeline import generate  # noqa: E402  (after sys.path bootstrap)
from src.schemas import GenerationRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxelCraft local smoke test")
    parser.add_argument("--prompt", default="a lighthouse on a cliff at sunset, oil painting")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("smoke_output.png"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    request = GenerationRequest(prompt=args.prompt, seed=args.seed, steps=args.steps)
    image, metadata = generate(request)
    image.save(args.out)

    print(f"\nSaved {args.out} ({image.size[0]}x{image.size[1]})")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
