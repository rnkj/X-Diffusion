import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.train_utils as train_utils
from dataset_utils.h5_dataset import H5Dataset
from models.xdiffusion.hr_classifier import HumanRobotClassifier
from scripts.train_classifier import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train-set predictions from a trained H5 classifier.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Classifier run directory containing config.yaml and checkpoints/latest.pth")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path. Defaults to run_dir/checkpoints/latest.pth")
    parser.add_argument("--device", default="cpu", help="Device for evaluation")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_plot_samples", type=int, default=0, help="0 plots all evaluated samples")
    parser.add_argument("--output_prefix", default="train_predictions")
    return parser.parse_args()


def load_config(run_dir: Path) -> TrainConfig:
    config_path = run_dir / "config.yaml"
    with config_path.open("r") as f:
        cfg = pyrallis.load(TrainConfig, f)
    return cfg


def main():
    args = parse_args()
    run_dir = args.run_dir
    checkpoint_path = args.checkpoint or run_dir / "checkpoints" / "latest.pth"

    cfg = load_config(run_dir)
    cfg.device = args.device
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0
    cfg.torch_compile = False
    cfg.dataset_cfg.balanced_sampling_weights = None

    dataset = H5Dataset(cfg.dataset_cfg, partition="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg.action_dim = dataset.action_dim
    cfg.state_cond_dim = dataset.state_cond_dim
    classifier = HumanRobotClassifier(
        obs_horizon=cfg.dataset_cfg.obs_horizon,
        state_cond_dim=cfg.state_cond_dim,
        action_dim=cfg.action_dim,
        cfg=cfg.classifier_cfg,
    ).to(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    classifier.eval()

    probs = []
    labels = []
    data_types = []
    losses = []
    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        for batch in loader:
            batch = train_utils.process_namedtuple_batch(batch, args.device)
            timesteps = torch.zeros(batch.label.shape[0], dtype=torch.long, device=args.device)
            logits = classifier.forward(batch, timesteps=timesteps).squeeze(-1)
            prob = torch.sigmoid(logits)
            loss = classifier.bce_loss(logits, batch.label.float())
            probs.append(prob.cpu().numpy())
            labels.append(batch.label.cpu().numpy())
            data_types.append(np.asarray(batch.data_type))
            losses.append(float(loss.item()))

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    data_types = np.concatenate(data_types)
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float((preds == labels).mean())
    mean_loss = float(np.mean(losses))

    csv_path = run_dir / f"{args.output_prefix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "robot_probability", "label", "prediction", "data_type"])
        for i, (prob, label, pred, dtype) in enumerate(zip(probs, labels, preds, data_types)):
            writer.writerow([i, float(prob), int(label), int(pred), int(dtype)])

    plot_count = len(probs) if args.max_plot_samples <= 0 else min(args.max_plot_samples, len(probs))
    x = np.arange(plot_count)
    plt.figure(figsize=(14, 5))
    plt.plot(x, probs[:plot_count], linewidth=1.0, label="Predicted P(robot)")
    plt.plot(x, labels[:plot_count], linewidth=0.8, alpha=0.65, label="Ground truth label")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Decision threshold")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Train sample index")
    plt.ylabel("Probability / label")
    plt.title(f"Train-set classifier predictions | acc={accuracy:.3f}, loss={mean_loss:.3f}, n={len(probs)}")
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path = run_dir / f"{args.output_prefix}.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Evaluated {len(probs)} train samples")
    print(f"accuracy={accuracy:.6f} loss={mean_loss:.6f}")
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
