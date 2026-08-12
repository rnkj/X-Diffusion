# X-Diffusion

X-Diffusion is method for training Diffusion Policies on cross-embodiment human demonstrations.

The project website and paper are available at https://portal-cornell.github.io/X-Diffusion/.

## Installation

Create a Python environment with PyTorch, then install the public dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset can be downloaded from Hugging Face at:

```text
https://huggingface.co/datasets/map438/X-Diffusion
```

The dataset contains robot and human demonstrations across 5 tasks:
- `close_drawer`
- `pan_on_plate`
- `push_plate`
- `mug_on_rack`
- `bottle_upright`

The dataset is split by embodiment:
- `robot` - the robot teleoperation demonstrations
- `human` - the human video demonstrations
- `human_filtered` - a manually-curated subset of the human demonstrations, available for `mug_on_rack`, `pan_on_plate`, and `push_plate` only

## Training
### Classifier Training
The X-Diffusion classifier can be configured in `configs/classifier.yaml`.

Then begin training with:
```bash
python scripts/train_classifier.py --config_path configs/classifier.yaml
```

### X-Diffusion
Once the classifier is trained, set the classifier checkpoint path in `configs/policy_xdiffusion.yaml`, and choose the same dataset the classifier was trained on.

Then train the X-Diffusion policy using:
```bash
python scripts/train_policy.py --config_path configs/policy_xdiffusion.yaml
```

### Co-Training Ablation Baselines
You can also train and compare the baselines from Section V.B, including:
- Robot Only: a policy trained only on the robot teleoperation demonstrations
- Naive Co-training: a policy co-trained on the robot and full human datasets without using the classifier
- Filtered: a policy co-trained on the robot dataset and the manually-filtered human dataset

To train any of these, pass in the desired config file:

```bash
python scripts/train_policy.py --config_path configs/<policy_robot_only.yaml | policy_naive_cotraining.yaml | policy_filtered.yaml>
```

## Offline Evaluation

You can inspect the classifier's predictions at different noise levels using:
```bash
python scripts/evaluate_classifier_noise_sweep.py --run_dir <classifier_run_dir>
```
where `<classifier_run_dir>` is the output directory created by
`scripts/train_classifier.py`.

You can also visualize the minimum validation loss across the four public modes.
This reads each run's `validation_report.json`, so validate every run first:
```bash
python scripts/validate_run.py --run_dir <run_dir>
```

Then plot the comparison:
```bash
python scripts/plot_validation_losses.py --runs <xdiffusion_run_dir> <filtered_run_dir> <naive_run_dir> <robot_only_run_dir>
```
Each run directory is matched by the training mode recorded in its validation
report, so the directory names themselves can be anything.