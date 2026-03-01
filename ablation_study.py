import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fen_recognition.model import AblationConfig
from src.fen_recognition.train import train

VARIANTS = {
    "baseline": AblationConfig(),
    "no_transformer": AblationConfig(use_transformer=False),
    "dense_instead": AblationConfig(use_transformer=False, use_dense_instead=True),
    "no_full_img": AblationConfig(use_full_img=False),
    "no_pos_embed": AblationConfig(use_pos_embed=False),
    "no_tiles": AblationConfig(use_tiles=False),
}


def run_ablation_study(game: str, total_steps: int, outdir: str, batch_size: int, lr: float):
    os.makedirs(outdir, exist_ok=True)
    results = {}

    for name, config in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Training variant: {name}")
        print(f"Config: {config}")
        print(f"{'='*60}\n")

        result = train(
            game=game,
            outdir=os.path.join(outdir, name),
            total_steps=total_steps,
            batch_size=batch_size,
            lr=lr,
            ablation_config=config,
        )
        results[name] = result

    # Summary table
    print(f"\n{'='*60}")
    print("Ablation Study Results")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Best Acc':>10}")
    print(f"{'-'*30}")
    for name, result in results.items():
        print(f"{name:<20} {result['best_acc']:>10.3f}")

    # Save results as JSON (without non-serializable fields)
    summary = {name: {"best_acc": r["best_acc"], "test_acc_list": r["test_acc_list"], "test_loss_list": r["test_loss_list"]} for name, r in results.items()}
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, result in results.items():
        axes[0].plot(result["test_loss_list"], label=name)
        axes[1].plot(result["test_acc_list"], label=name)

    axes[0].set_title("Test Loss")
    axes[0].set_xlabel("Evaluation step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Evaluation step")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(outdir, "ablation_comparison.png")
    plt.savefig(plot_path, dpi=250)
    print(f"\nComparison plot saved to {plot_path}")
    print(f"Results JSON saved to {os.path.join(outdir, 'results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for BoardRec model")
    parser.add_argument("--game", type=str, default="chess")
    parser.add_argument("--total_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00004)
    parser.add_argument("--outdir", type=str, default="ablation_results")
    args = parser.parse_args()

    run_ablation_study(
        game=args.game,
        total_steps=args.total_steps,
        outdir=args.outdir,
        batch_size=args.batch_size,
        lr=args.lr,
    )
