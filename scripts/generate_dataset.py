from __future__ import annotations

import argparse
from pathlib import Path

from data.generate import generate_synthetic_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic invoice dataset.")
    parser.add_argument("--output_dir", required=True, help="Directory to save dataset.")
    parser.add_argument("--n_train", type=int, default=1500)
    parser.add_argument("--n_val", type=int, default=250)
    parser.add_argument("--n_test", type=int, default=250)
    parser.add_argument("--noise_level", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=Path(args.output_dir),
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        noise_level=args.noise_level,
        seed=args.seed,
    )

    print(f"\nDataset generated at: {args.output_dir}")
    print(f"  Train: {args.n_train} examples")
    print(f"  Val:   {args.n_val} examples")
    print(f"  Test:  {args.n_test} examples")
