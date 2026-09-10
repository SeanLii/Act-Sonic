#!/usr/bin/env python3
"""Serve an ACT checkpoint to the official GEAR-SONIC VLA inference client."""

from __future__ import annotations

import argparse
from pathlib import Path

from act.server import PolicyServer
from act.sonic_policy import ACTSonicPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = ACTSonicPolicy.from_checkpoint(
        args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        strict=not args.no_strict,
    )
    print(
        f"Loaded ACT checkpoint={args.checkpoint.resolve()} "
        f"horizon={policy.model.chunk_size} device={policy.device}"
    )
    with PolicyServer(policy=policy, host=args.host, port=args.port) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down ACT-Sonic server...")


if __name__ == "__main__":
    main()
