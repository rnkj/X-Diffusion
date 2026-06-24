# Canonical Dataset Schema

This document defines the public X-Diffusion H5 dataset contract. It describes
the distributable layout only. Source paths, source-to-destination mappings,
feasible/infeasible provenance, and private milestone evidence are not part of
the public dataset schema.

## Distribution Expectation

The dataset is expected to be distributed separately from this repository, for
example through a public Hugging Face dataset page. The Git repository should
contain code, documentation, and small configuration examples only.

## Directory Layout

The dataset root contains two modality trees:

```text
<root>/
  retargeted/<task>/<embodiment>/demoNNNNN.h5
  rgb_seg/<task>/<embodiment>_imgs/demoNNNNN_imgs.h5
```

`NNNNN` is a zero-padded five-digit episode ID. The canonical tasks are:

- `close_drawer`
- `pan_on_plate`
- `push_plate`
- `mug_on_rack`
- `bottle_upright`

The public embodiments are `robot` and `human`. The optional
`human_filtered` embodiment exists only for `pan_on_plate`, `push_plate`, and
`mug_on_rack` and contains feasible-human episodes for filtered-policy
experiments. The combined `human` IDs do not encode feasible/infeasible source
provenance.

## Episode Inventory

| Task | `robot` | `human` | `human_filtered` |
| --- | ---: | ---: | ---: |
| `close_drawer` | 6 | 40 | absent |
| `pan_on_plate` | 6 | 100 | 50 |
| `push_plate` | 6 | 100 | 50 |
| `mug_on_rack` | 6 | 99 | 49 |
| `bottle_upright` | 6 | 100 | absent |

IDs are dense from zero except for two intentional gaps:

- `mug_on_rack/human` omits `demo00029.h5`.
- `mug_on_rack/human_filtered` omits `demo00033.h5`.

All other robot, human, and filtered inventories use IDs from zero through
count minus one.

## H5 Contract

Every dataset listed below must have a nonzero leading frame dimension `T`.
All datasets within an episode and all corresponding modality files must use
the same `T`.

### Physical/action files

Required datasets in `retargeted/.../demoNNNNN.h5`:

| Key | Required shape | Meaning |
| --- | --- | --- |
| `ee_pos` | `(T, 3)` | end-effector position |
| `ee_euler` | `(T, 3)` | end-effector Euler orientation |
| `gripper_open` | `(T,)` or `(T, 1)` | gripper state/action |
| `3d_tracks` | `(T, K, 3)`, `K > 0` | embodiment keypoint tracks |

Additional datasets are allowed.

#### Image files

Required datasets in `rgb_seg/.../demoNNNNN_imgs.h5`:

| Key | Required shape | Meaning |
| --- | --- | --- |
| `agent1_images` | `(T, H, W, 3)` | RGB observations |
| `segmentation` | `(T, H, W)` | segmentation labels/mask |

`H` and `W` must be positive. Depth datasets and any additional camera or
derived arrays are optional. In particular, consumers must not require depth:
the verified close-drawer image files do not contain it.

## Training Modes

The public training surface expects these embodiment selections:

- `robot_only`: `robot` only
- `naive`: `robot` plus combined `human`
- `classifier`: combined `human` plus `robot`
- `xdiffusion`: combined `human` plus `robot` with frozen classifier masking
- `filtered`: `robot` plus `human_filtered` only

## Validation and Privacy

Run strict canonical validation with:

```bash
python scripts/validate_dataset.py   --root /path/to/X-Diffusion-Data   --require-images
```

The validator opens H5 files read-only. It never repairs, converts, renames, or
deletes input. `--manifest` is optional and is limited to checking that an
existing JSONL file parses; validation does not require or create a manifest.
Reports include only aggregate manifest counts, never source paths or manifest
records.

The dataset root's `_private/` directory, source manifests, checksums containing
private paths, feasible/infeasible mappings, checkpoints, and training outputs
are internal evidence. They must not be committed to the public repository or
included in a public dataset release.
