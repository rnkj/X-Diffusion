#!/usr/bin/env python3
"""Validate the canonical X-Diffusion H5 dataset without modifying it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import h5py


TASKS = (
    "close_drawer",
    "pan_on_plate",
    "push_plate",
    "mug_on_rack",
    "bottle_upright",
)
EXPECTED_IDS = {
    "close_drawer": {
        "robot": set(range(6)),
        "human": set(range(40)),
    },
    "pan_on_plate": {
        "robot": set(range(6)),
        "human": set(range(100)),
        "human_filtered": set(range(50)),
    },
    "push_plate": {
        "robot": set(range(6)),
        "human": set(range(100)),
        "human_filtered": set(range(50)),
    },
    "mug_on_rack": {
        "robot": set(range(6)),
        "human": set(range(100)) - {29},
        "human_filtered": set(range(50)) - {33},
    },
    "bottle_upright": {
        "robot": set(range(6)),
        "human": set(range(100)),
    },
}
PHYSICAL_RE = re.compile(r"^demo(\d{5})\.h5$")
IMAGE_RE = re.compile(r"^demo(\d{5})_imgs\.h5$")


def _issue(code: str, message: str, path: str | None = None) -> dict[str, str]:
    result = {"code": code, "message": message}
    if path is not None:
        result["path"] = path
    return result


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _episode_ids(
    directory: Path,
    pattern: re.Pattern[str],
    root: Path,
    issues: list[dict[str, str]],
) -> dict[int, Path]:
    episodes: dict[int, Path] = {}
    if not directory.is_dir():
        return episodes
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != ".h5":
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            issues.append(
                _issue(
                    "invalid_filename",
                    "H5 episode filename does not match the canonical pattern",
                    _relative(path, root),
                )
            )
            continue
        episode_id = int(match.group(1))
        if episode_id in episodes:
            issues.append(
                _issue(
                    "duplicate_episode_id",
                    f"episode ID {episode_id} occurs more than once",
                    _relative(path, root),
                )
            )
        episodes[episode_id] = path
    return episodes


def _shape(dataset: h5py.Dataset) -> tuple[int, ...]:
    return tuple(int(value) for value in dataset.shape)


def _validate_physical(path: Path, root: Path) -> tuple[int | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    rel = _relative(path, root)
    try:
        with h5py.File(path, "r") as handle:
            required = ("ee_pos", "ee_euler", "gripper_open", "3d_tracks")
            for key in required:
                if key not in handle:
                    errors.append(_issue("missing_key", f"missing required key {key!r}", rel))
            if errors:
                return None, errors

            shapes = {key: _shape(handle[key]) for key in required}
            if len(shapes["ee_pos"]) != 2 or shapes["ee_pos"][1:] != (3,):
                errors.append(_issue("invalid_shape", f"ee_pos has shape {shapes['ee_pos']}, expected (T, 3)", rel))
            if len(shapes["ee_euler"]) != 2 or shapes["ee_euler"][1:] != (3,):
                errors.append(_issue("invalid_shape", f"ee_euler has shape {shapes['ee_euler']}, expected (T, 3)", rel))
            gripper_shape = shapes["gripper_open"]
            if len(gripper_shape) not in (1, 2) or (len(gripper_shape) == 2 and gripper_shape[1:] != (1,)):
                errors.append(_issue("invalid_shape", f"gripper_open has shape {gripper_shape}, expected (T,) or (T, 1)", rel))
            tracks_shape = shapes["3d_tracks"]
            if len(tracks_shape) != 3 or tracks_shape[1] < 1 or tracks_shape[2:] != (3,):
                errors.append(_issue("invalid_shape", f"3d_tracks has shape {tracks_shape}, expected (T, K, 3) with K > 0", rel))

            frame_counts = {key: shape[0] for key, shape in shapes.items() if shape}
            if not frame_counts or min(frame_counts.values()) < 1:
                errors.append(_issue("empty_episode", "episode datasets must contain at least one frame", rel))
                return None, errors
            if len(set(frame_counts.values())) != 1:
                errors.append(_issue("frame_mismatch", f"physical frame counts differ: {frame_counts}", rel))
                return None, errors
            return next(iter(frame_counts.values())), errors
    except (OSError, TypeError, ValueError) as exc:
        return None, [_issue("unreadable_h5", f"cannot read H5 file: {exc}", rel)]


def _validate_images(path: Path, root: Path) -> tuple[int | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    rel = _relative(path, root)
    try:
        with h5py.File(path, "r") as handle:
            required = ("agent1_images", "segmentation")
            for key in required:
                if key not in handle:
                    errors.append(_issue("missing_key", f"missing required key {key!r}", rel))
            if errors:
                return None, errors
            image_shape = _shape(handle["agent1_images"])
            segmentation_shape = _shape(handle["segmentation"])
            if len(image_shape) != 4 or min(image_shape[:3], default=0) < 1 or image_shape[3:] != (3,):
                errors.append(_issue("invalid_shape", f"agent1_images has shape {image_shape}, expected (T, H, W, 3)", rel))
            if len(segmentation_shape) != 3 or min(segmentation_shape, default=0) < 1:
                errors.append(_issue("invalid_shape", f"segmentation has shape {segmentation_shape}, expected (T, H, W)", rel))
            if image_shape and segmentation_shape and image_shape[0] != segmentation_shape[0]:
                errors.append(_issue("frame_mismatch", f"image frame counts differ: agent1_images={image_shape[0]}, segmentation={segmentation_shape[0]}", rel))
                return None, errors
            return image_shape[0] if image_shape else None, errors
    except (OSError, TypeError, ValueError) as exc:
        return None, [_issue("unreadable_h5", f"cannot read H5 file: {exc}", rel)]


def _validate_manifest(path: Path) -> tuple[int, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(_issue("invalid_manifest", f"manifest line {line_number} is invalid JSON: {exc.msg}"))
                    continue
                if not isinstance(record, dict):
                    errors.append(_issue("invalid_manifest", f"manifest line {line_number} is not a JSON object"))
                    continue
                count += 1
    except OSError as exc:
        errors.append(_issue("unreadable_manifest", f"cannot read manifest: {exc}"))
    return count, errors


def _strict_inventory_issues(root: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for task, embodiments in EXPECTED_IDS.items():
        for embodiment, expected in embodiments.items():
            directory = root / "retargeted" / task / embodiment
            actual = set(_episode_ids(directory, PHYSICAL_RE, root, errors))
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                errors.append(
                    _issue(
                        "inventory_mismatch",
                        f"expected {len(expected)} episode IDs; missing={missing}, unexpected={unexpected}",
                        _relative(directory, root),
                    )
                )
        forbidden = {"robot", "human", "human_filtered"} - set(embodiments)
        for embodiment in forbidden:
            directory = root / "retargeted" / task / embodiment
            if directory.exists():
                errors.append(_issue("unexpected_embodiment", "embodiment is not defined for this task", _relative(directory, root)))
    return errors


def _strict_companion_inventory_issues(
    root: Path,
    *,
    tree: str,
    image_files: bool,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    pattern = IMAGE_RE if image_files else PHYSICAL_RE
    for task, embodiments in EXPECTED_IDS.items():
        for embodiment, expected in embodiments.items():
            directory_name = f"{embodiment}_imgs" if image_files else embodiment
            directory = root / tree / task / directory_name
            actual = set(_episode_ids(directory, pattern, root, errors))
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                errors.append(
                    _issue(
                        "inventory_mismatch",
                        f"expected {len(expected)} episode IDs; missing={missing}, unexpected={unexpected}",
                        _relative(directory, root),
                    )
                )
    return errors


def _discover_physical(root: Path) -> Iterable[tuple[str, str, int, Path]]:
    physical_root = root / "retargeted"
    if not physical_root.is_dir():
        return []
    episodes: list[tuple[str, str, int, Path]] = []
    for task_dir in sorted(path for path in physical_root.iterdir() if path.is_dir()):
        for embodiment_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            for path in sorted(embodiment_dir.glob("*.h5")):
                match = PHYSICAL_RE.fullmatch(path.name)
                if match:
                    episodes.append((task_dir.name, embodiment_dir.name, int(match.group(1)), path))
    return episodes


def validate_dataset(
    root: Path,
    *,
    require_images: bool = False,
    allow_partial: bool = False,
    manifest: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    errors: list[dict[str, str]] = []
    counts = {"physical": 0, "images": 0, "manifest_records": 0}

    if not root.is_dir():
        errors.append(_issue("missing_root", "dataset root is not a directory", root.name))
    else:
        if not allow_partial:
            errors.extend(_strict_inventory_issues(root))
            if require_images:
                errors.extend(
                    _strict_companion_inventory_issues(
                        root, tree="rgb_seg", image_files=True
                    )
                )

        episodes = list(_discover_physical(root))
        if not episodes:
            errors.append(_issue("empty_dataset", "no canonical physical episodes found", "retargeted"))

        for task, embodiment, episode_id, physical_path in episodes:
            if task not in EXPECTED_IDS:
                errors.append(_issue("unknown_task", f"unknown task {task!r}", _relative(physical_path, root)))
                continue
            if embodiment not in EXPECTED_IDS[task]:
                errors.append(_issue("unknown_embodiment", f"unknown embodiment {embodiment!r} for task", _relative(physical_path, root)))
                continue

            counts["physical"] += 1
            physical_frames, physical_errors = _validate_physical(physical_path, root)
            errors.extend(physical_errors)

            image_path = root / "rgb_seg" / task / f"{embodiment}_imgs" / f"demo{episode_id:05d}_imgs.h5"
            if image_path.exists():
                counts["images"] += 1
                image_frames, image_errors = _validate_images(image_path, root)
                errors.extend(image_errors)
                if physical_frames is not None and image_frames is not None and physical_frames != image_frames:
                    errors.append(_issue("frame_mismatch", f"physical/image frames differ: {physical_frames} != {image_frames}", _relative(image_path, root)))
            elif require_images:
                errors.append(_issue("missing_modality", "missing corresponding image H5", _relative(image_path, root)))

    if manifest is not None:
        manifest_count, manifest_errors = _validate_manifest(manifest)
        counts["manifest_records"] = manifest_count
        errors.extend(manifest_errors)

    return {
        "valid": not errors,
        "mode": "partial" if allow_partial else "canonical",
        "requirements": {"images": require_images},
        "counts": counts,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="dataset root to inspect read-only")
    parser.add_argument("--manifest", type=Path, help="optional existing private JSONL manifest to parse read-only")
    parser.add_argument("--require-images", action="store_true", help="require an image H5 for every physical episode")
    parser.add_argument("--allow-partial", action="store_true", help="validate a fixture/subset without enforcing canonical episode counts")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(
        args.root,
        require_images=args.require_images,
        allow_partial=args.allow_partial,
        manifest=args.manifest,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
