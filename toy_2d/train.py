from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toy_2d.datasets import build_distribution
from toy_2d.diffusion import DiffusionSchedule
from toy_2d.model import EpsilonMLP
from toy_2d.trainer import DiffusionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D DDPM prior on toy datasets.")
    parser.add_argument("--dataset", type=str, choices=["two_moons", "eight_gaussians"], required=True)
    parser.add_argument("--output_dir", type=str, default="toy_2d_outputs")
    parser.add_argument("--num_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--schedule_type", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    distribution = build_distribution(args.dataset)
    schedule = DiffusionSchedule(
        num_steps=args.diffusion_steps,
        schedule_type=args.schedule_type,
        device=device,
    )
    model = EpsilonMLP(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        num_layers=args.num_layers,
    ).to(device)

    trainer = DiffusionTrainer(
        model=model,
        schedule=schedule,
        distribution=distribution,
        device=device,
        lr=args.lr,
    )
    losses = trainer.train(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )

    run_dir = Path(args.output_dir) / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "ddpm_toy_2d.pt"
    trainer.save_checkpoint(
        checkpoint_path,
        losses=losses,
        extra_config={
            "dataset": args.dataset,
            "model": {
                "hidden_dim": args.hidden_dim,
                "time_dim": args.time_dim,
                "num_layers": args.num_layers,
            },
            "training": {
                "num_steps": args.num_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "device": args.device,
            },
        },
    )

    with open(run_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

