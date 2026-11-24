# DMET 901 Computer Vision project: screen privacy through head orientation

ResNet-18 classifies a 64×64 face crop as **forward** or **away** (forward = |yaw| ≤ 20° and |pitch| ≤ 20°). Trained on BIWI with a person-level split (12 train / 4 val / 4 test people). See README.md for the folder layout.

- MS2 baseline: test macro-F1 0.894 ± 0.021 (3 seeds).
- MS3 final model "A + B" (small-input ResNet-18 + 300W-LP training data): 0.908 ± 0.005; real errors 99 → 76.
- `MS3/build_notebook.py` generates `ms3_improvements.ipynb`. Finished runs are cached in `ms3_runs.json`, so re-executing only re-renders the outputs.
- `MS3/export_onnx.py` exports `checkpoints/A_+_B_seed44.pt` to `app/models/head_direction_resnet18.onnx`.
- `app/`: Electron demo (YuNet → crop with 0.4 margin like the BIWI crops → our ONNX model). `npm start`.
- Local Python env: `.venv` (has CUDA torch). `data/` and checkpoints are gitignored.

## Milestone 3 rules (from the TA)
These do NOT count as improvements: changing hyperparameters, metrics or the loss; switching to a different model; data augmentation. Augmentation and hyperparameter runs are in the notebook's section 10 as reference only.

## TODO when building the MS3 presentation
- The TA wants **evidence for everything on the slides**: Google Colab screenshots of each run, the training logs, the tests and the results. Run the final notebook in Colab, then screenshot the cells that back up every number and figure on the slides.
- Include screenshots of the desktop demo app running.
- Say on the slides why we switched away from the augmentation proposed in MS2: the MS3 rules exclude augmentation.
