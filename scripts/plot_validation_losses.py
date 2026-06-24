from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

EXPECTED_RUN_ORDER = [
    "m5_xdiffusion",
    "m6_filtered",
    "m4_naive",
    "m4_robot_only",
]

EXPECTED_RUN_LABELS = {
    "m5_xdiffusion": "X-Diffusion",
    "m6_filtered": "Filtered co-training",
    "m4_naive": "Naive co-training",
    "m4_robot_only": "Robot-only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot minimum validation losses from validated runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "Validated training run directories, for example directories like "
            "m5_xdiffusion or m4_robot_only that contain metrics.jsonl and validation_report.json"
        ),
    )
    parser.add_argument("--output", help="Optional PNG output path")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_metrics(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"metrics.jsonl is empty: {path}")
    return rows


def collect_run_summary(run_dir: Path) -> Dict[str, float | str]:
    validation = load_json(run_dir / "validation_report.json")
    if not validation.get("valid", False):
        raise ValueError(f"Run is not validated: {run_dir}")
    rows = load_metrics(run_dir / "metrics.jsonl")
    min_val = min(float(row["val_loss"]) for row in rows)
    return {
        "run_name": run_dir.name,
        "label": EXPECTED_RUN_LABELS[run_dir.name],
        "min_val_loss": min_val,
    }


def default_output_path(run_dirs: List[Path]) -> Path:
    common_root = Path(os.path.commonpath([str(path.resolve()) for path in run_dirs]))
    output_root = common_root if common_root.is_dir() else common_root.parent
    return output_root / "validation_loss_comparison.png"


def plot_validation_losses(run_dirs: List[Path], output_path: Path) -> List[Dict[str, float | str]]:
    run_names = {path.name for path in run_dirs}
    expected_names = set(EXPECTED_RUN_LABELS)
    if run_names != expected_names:
        missing = sorted(expected_names - run_names)
        extra = sorted(run_names - expected_names)
        raise ValueError(f"Run set mismatch. Missing={missing} Extra={extra}")

    run_by_name = {path.name: path for path in run_dirs}
    summaries = [collect_run_summary(run_by_name[name]) for name in EXPECTED_RUN_ORDER]
    labels = [item["label"] for item in summaries]
    values = [item["min_val_loss"] for item in summaries]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel("Minimum validation loss")
    ax.set_title("Smoke-run validation comparison")
    ax.set_ylim(bottom=0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return summaries


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path).resolve() for path in args.runs]
    output_path = Path(args.output).resolve() if args.output else default_output_path(run_dirs)
    summaries = plot_validation_losses(run_dirs, output_path)
    print(json.dumps({"output": str(output_path), "runs": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
