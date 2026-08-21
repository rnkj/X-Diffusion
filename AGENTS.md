# X-Diffusion -- Agents Guide

## Rules

- Rules live in .agents/rules
  Run the following at the start of a session:
  ```
  find .agents/rules -type f -name "*.mdc"
  ```
- Memory is stored under .agents/memory/. At the start of work, read .agents/memory/MEMORY.md to see the summaries and file paths of the memories worth reading, then read whichever memory files are relevant.
- Do not edit anything unless the user requests **even in auto mode**. Before starting any edits, always confirm whether the user has instructed to edit the file (e.g. "edit it", "implement it").
- Do not delete files within the project root. If deletion is necessary, only present the deletion command to be run, and leave the execution to the user.

---

## File Structure

```
.
├── .agents/               # agent rules (.agents/rules), memory (.agents/memory), skills
├── .claude/               # Claude Code config (agents, hooks, settings, skills, workflows)
├── calib_cornell/         # camera calibration data (intrinsics/extrinsics .npy)
├── common_utils/          # shared utilities (checkpointing, logging, data aug, transforms)
├── configs/               # training configs (classifier, policy variants) in YAML
├── dataset_utils/         # dataset loading/processing (h5 dataset, image transforms)
├── docs/                  # documentation (dataset conversion, etc.)
├── interactive_scripts/   # interactive recording/vision utilities
├── local/                 # gitignored local outputs/docs
├── models/                # model definitions (diffusion, encoders, policy nets)
├── scripts/               # training/evaluation entry-point scripts
├── third_party/           # vendored third-party code (e.g. hamer)
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore
```

