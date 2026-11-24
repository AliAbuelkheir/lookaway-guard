# LookAway Guard

**Screen privacy through head orientation.** A university computer vision project for DMET 901 Computer Vision at the German University in Cairo (GUC).

The laptop camera watches the person at the screen. While they look at the screen it stays visible; when they look away, it would be blurred so nobody else can read it.

The core is a ResNet-18 that classifies a 64 × 64 face crop as **forward** (|yaw| ≤ 20° and |pitch| ≤ 20°) or **away**. It is trained on BIWI and tested on 4 people it has never seen.

## Results (test set, mean of 3 seeds)

| Model | Macro-F1 | Accuracy | Real errors* |
|---|---|---|---|
| Milestone 2 baseline | 0.894 ± 0.021 | 89.7% | 99 ± 34 |
| **Milestone 3: small-input ResNet-18 + more training people** | **0.908 ± 0.005** | **91.0%** | **76 ± 6** |

\* Errors on heads that are clearly forward (< 17.5°) or clearly away (> 22.5°). Errors between 17.5° and 22.5°, right at the 20° line, are counted separately as "accepted".

## Repository

| Folder | Contents |
|---|---|
| `MS1/` | Milestone 1: dataset check notebook, report, slides |
| `MS2/` | Milestone 2: baseline training notebook, results, slides (`make_slides.py` builds the deck) |
| `MS3/` | Milestone 3: improvement experiments. `build_notebook.py` generates `ms3_improvements.ipynb`; `ms3_local_run.ipynb` is a full run with outputs; `export_onnx.py` exports the final model for the app |
| `app/` | Desktop demo (Electron): live camera → face detection → our model → forward / away |

## Run the notebooks

Open a notebook in Google Colab, pick a T4 GPU, then *Runtime → Run all*. The data is downloaded automatically.

Locally: `pip install -r requirements.txt`, then open the notebook with Jupyter.

## Run the demo app

Needs [Node.js](https://nodejs.org) 20 or newer. Works on Windows and macOS.

```
cd app
npm install
npm start
```

Press **Enter** (or the Capture button) to analyse a photo, or **L** for live mode. The app runs two models:

1. **YuNet** (OpenCV Zoo, pretrained) finds the face.
2. The face is cropped the same way as the training images and resized to 64 × 64.
3. **Our ResNet-18** (`app/models/head_direction_resnet18.onnx`) says forward or away.

## Data and credits

- BIWI Kinect Head Pose (Fanelli et al., 2013), using the face crops prepared by FSA-Net and the corrected angles from HeadPose+.
- 300W-LP (Zhu et al., 2016), face crops prepared by FSA-Net.
- YuNet face detector from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo).

The datasets are not included in this repository. The notebooks download them.
