from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifier P(robot) across diffusion timesteps")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--timestep_start", type=int, default=0)
    parser.add_argument("--timestep_stop", type=int, default=100)
    parser.add_argument("--timestep_step", type=int, default=1)
    parser.add_argument("--output_prefix", default="noise_sweep_p_robot")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict:
    return yaml.safe_load((run_dir / "config_resolved.yaml").read_text())


def instantiate_classifier(cfg: Dict, dataset, device: str):
    from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig

    classifier_cfg = HumanRobotClassifierConfig(
        num_train_timesteps=int(cfg["num_train_timesteps"]),
        add_noise_to_sample=bool(cfg.get("add_noise_to_sample", True)),
        add_noise_to_state_cond=bool(cfg.get("add_noise_to_state_cond", True)),
    )
    classifier = HumanRobotClassifier(
        obs_horizon=int(cfg["obs_horizon"]),
        state_cond_dim=dataset.state_cond_dim,
        action_dim=dataset.action_dim,
        cfg=classifier_cfg,
    ).to(device)
    return classifier


def build_val_dataset(cfg: Dict):
    from dataset_utils.h5_dataset import H5Dataset, H5DatasetConfig

    dataset_cfg = H5DatasetConfig(
        split_file_name=str(cfg["split_file"]),
        use_ee_data=bool(cfg["use_ee_data"]),
        add_grasp_info_to_tracks=True,
        load_images=False,
        obs_horizon=int(cfg["obs_horizon"]),
        pred_horizon=int(cfg["pred_horizon"]),
        action_horizon=int(cfg["action_horizon"]),
        balanced_sampling_weights=None,
    )
    dataset_cfg.physical_data_paths = [Path(cfg["dataset_dirs"]["robot"]), Path(cfg["dataset_dirs"]["human"])]
    dataset_cfg.rgb_seg_data_paths = list(dataset_cfg.physical_data_paths)
    return H5Dataset(dataset_cfg, partition="val")


def evaluate_noise_sweep(
    run_dir: Path,
    checkpoint: Path | None = None,
    device: str = "cpu",
    batch_size: int = 128,
    timestep_start: int = 0,
    timestep_stop: int = 100,
    timestep_step: int = 1,
    output_prefix: str = "noise_sweep_p_robot",
    seed: int = 42,
) -> Dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from models import train_utils

    if timestep_step <= 0:
        raise ValueError("timestep_step must be positive")
    timesteps = list(range(timestep_start, timestep_stop + 1, timestep_step))
    if timesteps != list(range(0, 101)):
        raise ValueError(f"Expected exact timesteps 0..100 inclusive, got {timesteps[:5]} ... {timesteps[-5:]}")

    cfg = load_config(run_dir)
    checkpoint_path = checkpoint or (run_dir / "checkpoints" / "latest.pth")
    dataset = build_val_dataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    classifier = instantiate_classifier(cfg, dataset, device)
    checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    classifier.eval()

    rows: List[Dict] = []
    torch.manual_seed(seed)
    for timestep in timesteps:
        human_vals = []
        robot_vals = []
        all_vals = []
        with torch.no_grad():
            for batch in loader:
                batch = train_utils.process_namedtuple_batch(batch, device)
                ts = torch.full((batch.label.shape[0],), int(timestep), dtype=torch.long, device=device)
                logits = classifier.forward(batch, timesteps=ts).squeeze(-1)
                probs = torch.sigmoid(logits)
                if not torch.isfinite(probs).all():
                    raise RuntimeError(f"Non-finite probabilities at timestep {timestep}")
                if probs.min().item() < 0.0 or probs.max().item() > 1.0:
                    raise RuntimeError(f"Probabilities outside [0,1] at timestep {timestep}")
                human_mask = batch.label == 0
                robot_mask = batch.label == 1
                all_vals.append(probs.cpu().numpy())
                if human_mask.any():
                    human_vals.append(probs[human_mask].cpu().numpy())
                if robot_mask.any():
                    robot_vals.append(probs[robot_mask].cpu().numpy())
        all_np = np.concatenate(all_vals)
        human_np = np.concatenate(human_vals) if human_vals else np.array([], dtype=np.float32)
        robot_np = np.concatenate(robot_vals) if robot_vals else np.array([], dtype=np.float32)
        row = {
            "timestep": int(timestep),
            "mean_p_robot_all": float(all_np.mean()),
            "mean_p_robot_human": float(human_np.mean()) if human_np.size else float("nan"),
            "mean_p_robot_robot": float(robot_np.mean()) if robot_np.size else float("nan"),
            "human_count": int(human_np.size),
            "robot_count": int(robot_np.size),
            "probability_min": float(all_np.min()),
            "probability_max": float(all_np.max()),
        }
        rows.append(row)

    if len(rows) != 101:
        raise RuntimeError(f"Expected 101 timestep rows, got {len(rows)}")

    csv_path = run_dir / f"{output_prefix}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = run_dir / f"{output_prefix}.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    xs = [row["timestep"] for row in rows]
    ys_human = [row["mean_p_robot_human"] for row in rows]
    ys_robot = [row["mean_p_robot_robot"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_human, marker="o", linewidth=1.5, markersize=3, label="Human mean P(robot)")
    plt.plot(xs, ys_robot, marker="o", linewidth=1.5, markersize=3, label="Robot mean P(robot)")
    plt.xlabel("Noise timestep")
    plt.ylabel("Mean predicted P(robot)")
    plt.title("Classifier Logits of Actions over Diffusion Noise Timesteps")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path = run_dir / f"{output_prefix}.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    summary = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "rows": len(rows),
        "timestep_start": rows[0]["timestep"],
        "timestep_stop": rows[-1]["timestep"],
        "all_probabilities_finite": True,
        "all_probabilities_in_unit_interval": True,
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "plot_path": str(plot_path.resolve()),
    }
    (run_dir / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    summary = evaluate_noise_sweep(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        timestep_start=args.timestep_start,
        timestep_stop=args.timestep_stop,
        timestep_step=args.timestep_step,
        output_prefix=args.output_prefix,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
