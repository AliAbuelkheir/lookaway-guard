"""Export the chosen MS3 model (A + B: small-input ResNet-18 trained with 300W-LP) to ONNX for the desktop app.

The ONNX model takes RGB face crops, shape (N, 3, 64, 64), pixel values 0-255,
and returns [P(away), P(forward)]. The ImageNet normalisation is built in.
Run from the MS3 folder:  ..\\.venv\\Scripts\\python export_onnx.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import resnet18

CHECKPOINT = Path("checkpoints/A_+_B_seed44.pt")     # chosen in ms3_improvements.ipynb (best validation F1 among the A + B seeds)
OUT = Path("../app/models/head_direction_resnet18.onnx")


class Exported(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1.stride = (1, 1)          # improvement A: read the 64x64 crop directly
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(512, 2)
        m.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        self.net = m.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return self.net((x - self.mean) / self.std).softmax(1)


model = Exported().eval()
torch.onnx.export(model, torch.zeros(1, 3, 64, 64), OUT, input_names=["crop"], output_names=["probs"],
                  dynamic_axes={"crop": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

# Check: the ONNX model must reproduce the notebook's test predictions for this checkpoint
images = np.load("../data/BIWI_noTrack.npz")["image"][..., ::-1]          # BGR -> RGB
fsa = np.load("../data/headposeplus/data/Biwi_FSA-Net.npz")
seq = np.array([int(n[:2]) for n in fsa["images"]])
test = np.where(np.isin(seq, [1, 2, 17, 20, 21]))[0]
x = images[test].transpose(0, 3, 1, 2).astype(np.float32)
sess = ort.InferenceSession(str(OUT))
pred = np.concatenate([sess.run(None, {"crop": x[i:i + 256]})[0] for i in range(0, len(x), 256)]).argmax(1)
ref = np.array(json.loads(Path("ms3_runs.json").read_text())["A + B | seed 44"]["test_pred"])
print(f"ONNX vs notebook test predictions: {(pred == ref).mean():.2%} identical ({(pred != ref).sum()} differ of {len(ref)})")
