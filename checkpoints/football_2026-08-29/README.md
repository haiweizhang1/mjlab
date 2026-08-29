# Football key checkpoints (2026-08-29)

This directory intentionally contains only the checkpoints needed to reproduce
the current Klavier Teacher/Depth Student workflow and compare it with the
previous strong Depth Student. Runtime logs, intermediate checkpoints, W&B
files, TensorBoard events, and videos remain excluded by `.gitignore`.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `walk_init_model_20000.pt` | Klavier walk initialization checkpoint | `a0eb99185d62294eae8ac470a3bde58f2ebe96bc288ab5d6a019a34ef317e391` |
| `teacher_klavier_model_47000.pt` | Klavier football Teacher used for current distillation | `c666d516f7d6dc81bc6f6b8f27399b9865fe5eb7612413f4ca4ef4270ce53fc0` |
| `student_old_strong_model_9999.pt` | Previous strong Depth Student comparison checkpoint | `34fa57c058005826ea1610eeedf1092c58581242cbb79a52ad29acf99f9d97d5` |
| `student_old_strong.onnx` | Deployment export of the previous strong Depth Student | `49e883b82f4bda0cc10ea0f1a49cd8cccbf51d72f023478fae5afeb3c65e72fe` |
| `student_visibility_model_7000.pt` | Current from-zero visibility-supervised Depth Student | `013221161d9267fdc6ed92a94d9d405db04d864cea307dc138dbe5bdd5fa07ad` |
| `student_visibility_model_7000.onnx` | Deployment export corresponding to the current Student run | `078914769b725570579b5917bb02d349b081939269f924744336ad83a14f7e4b` |

The model files are normally ignored repository-wide. They are deliberately
force-added for this curated snapshot; do not add complete training directories.
