"""Builds the proposed MS3 deck (../MS3_presentation_proposal.pptx). Same design as MS2/make_slides.py.

Numbers come from ms3_runs.json / ms3_results.json (written by ms3_improvements.ipynb) and ../MS2/ms2_results.json.
Pictures come from figures/ (notebook plots) and screenshots/ (notebook cells and the demo app).
Run after the notebook:  python make_slides.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
FIG, SHOTS, MS2_ASSETS = HERE / "figures", HERE / "screenshots", HERE.parent / "MS2" / "assets"
RUNS = json.loads((HERE / "ms3_runs.json").read_text())
SUMMARY = json.loads((HERE / "ms3_results.json").read_text())
MS2 = json.loads((HERE.parent / "MS2" / "ms2_results.json").read_text())
CFG = SUMMARY["config"]
SEEDS = CFG["seeds"]

# ---------- numbers used on the slides ----------
def runs_of(name):
    return [RUNS[f"{name} | seed {s}"] for s in SEEDS]

def test(name, key):
    return np.array([r["test"][key] for r in runs_of(name)], dtype=float)

def gap(name):
    return np.array([r["train_f1"] - r["val_f1"] for r in runs_of(name)])

def val(name):
    return np.array([r["val_f1"] for r in runs_of(name)])

BASE, FINAL = "Baseline (MS2)", SUMMARY["final_model"]["experiment"]
NAMES = list(SUMMARY["experiments"])
f3 = lambda v: f"{v:.3f}"
pm3 = lambda a: f"{a.mean():.3f} ± {a.std():.3f}"
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
n_test = MS2["counts"]["test"]["images"]
minutes = sum(r["minutes"] for r in RUNS.values())
device = SUMMARY["device"].replace("NVIDIA GeForce ", "")
nb_text = (HERE / "ms3_improvements.ipynb").read_text(encoding="utf-8")
onnx_match = re.search(r"(\d+) identical predictions \(([\d.]+)%\)", nb_text)
onnx_same, onnx_pct = (onnx_match.group(1), onnx_match.group(2)) if onnx_match else ("?", "?")
lp_used = re.search(r"([\d,]+) used", nb_text).group(1)
real_b, real_f = test(BASE, "real_errors"), test(FINAL, "real_errors")
ms2_s2 = [h["val_f1"] for h in MS2["history"] if h["stage"] == 2]

# ---------- style (same as Milestone 2) ----------
DARK, BODY, MUTED, LINE = "1B2A33", "4A5560", "7A858C", "C9CED1"
BG, CARD, WHITE = "F6F5F1", "FFFFFF", "FFFFFF"
BLUE, ORANGE, SAND, GREEN = "1F5F84", "B8652A", "E0A56E", "3E7D5A"
FONT = "Calibri"
W, H, M = 13.333, 7.5, 0.6

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
rgb = RGBColor.from_string


def slide(bg=BG):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    return s


def text(s, x, y, w, h, runs, size=15, color=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=6, line=1.1):
    """runs: str, or list of paragraphs; a paragraph is a str or a list of (text, {opts}) pieces."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.line_spacing = align, line
        p.space_after = Pt(space_after)
        pieces = [(para, {})] if isinstance(para, str) else para
        for t, o in pieces:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = FONT, Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(s, x, y, w, h, fill=CARD, line=LINE, radius=0.06):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(0.75)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def title(s, t, n, footer=""):
    text(s, M, 0.45, W - 2 * M, 0.7, t, size=34, color=DARK, bold=True)
    text(s, M, 6.95, W - 2 * M - 0.6, 0.3, footer, size=11, color=MUTED)
    text(s, W - M - 0.5, 6.95, 0.5, 0.3, str(n), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def table(s, x, y, w, rows, col_w, size=13, row_h=0.36, align_right_from=1, highlight=None, center=False):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(DARK if i == 0 else ("EEF3F6" if i in (highlight or []) else CARD))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j < align_right_from else (PP_ALIGN.CENTER if center else PP_ALIGN.RIGHT)
            r = p.add_run()
            r.text = str(value)
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0 or i in (highlight or [])
            r.font.color.rgb = rgb(WHITE if i == 0 else DARK)
    return t


def card(s, x, y, w, h, head, body, head_color=BLUE, size=14):
    box(s, x, y, w, h)
    text(s, x + 0.25, y + 0.22, w - 0.5, 0.4, head, size=17, color=head_color, bold=True)
    text(s, x + 0.25, y + 0.68, w - 0.5, h - 0.85, body, size=size, color=BODY)


def picture(s, path, x, y, w=None, h=None, frame=False):
    """Adds an image. With both w and h, it is scaled to fit inside that box and centred in it."""
    if w and h:
        iw, ih = Image.open(path).size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        x, y, w, h = x + (w - pw) / 2, y + (h - ph) / 2, pw, ph
    pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if frame:
        pic.line.color.rgb = rgb(LINE)
        pic.line.width = Pt(0.75)
    return pic


def evidence(s, path, x, y, w, h, caption):
    """A notebook / app screenshot with a small caption under it (a placeholder box if the screenshot is missing)."""
    if not Path(path).exists():
        box(s, x, y, w, h - 0.32, fill=CARD, line=LINE)
        text(s, x, y + (h - 0.32) / 2 - 0.2, w, 0.4, f"Add screenshot: {Path(path).name}", size=14, color=MUTED, align=PP_ALIGN.CENTER)
        text(s, x, y + h - 0.24, w, 0.3, caption, size=11, color=MUTED)
        return
    pic = picture(s, path, x, y, w, h - 0.32, frame=True)
    text(s, pic.left / 914400, (pic.top + pic.height) / 914400 + 0.08, w, 0.3, caption, size=11, color=MUTED)
    return pic


def steps(s, y, items, h=1.05):
    """A row of boxes with arrows; items are (heading, detail, highlighted)."""
    gap_ = 0.33
    sw = (W - 2 * M - (len(items) - 1) * gap_) / len(items)
    for k, (a, b, ours) in enumerate(items):
        x = M + k * (sw + gap_)
        box(s, x, y, sw, h, fill=DARK if ours else CARD, line=None if ours else LINE)
        text(s, x + 0.2, y + 0.16, sw - 0.4, 0.35, a, size=16, bold=True, color="F3F1EA" if ours else DARK)
        text(s, x + 0.2, y + 0.55, sw - 0.4, 0.45, b, size=12, color="C9D6DC" if ours else MUTED)
        if k < len(items) - 1:
            text(s, x + sw, y + 0.3, gap_, 0.4, "→", size=18, color=MUTED, align=PP_ALIGN.CENTER)


def stat(s, x, y, w, h, big, small):
    box(s, x, y, w, h, fill=DARK, line=None)
    text(s, x + 0.3, y + 0.18, w - 0.6, 0.9, big, size=44, color=SAND, bold=True)
    text(s, x + 0.3, y + 1.05, w - 0.6, h - 1.15, small, size=14, color="E8ECEE")


n = 0
def next_n():
    global n
    n += 1
    return n


# 1 ---------- title ----------
s = slide(DARK); next_n()
text(s, M + 0.3, 0.9, 10, 0.4, "DMET 901 Computer Vision, Milestone 3", size=15, color="9FB7C4")
text(s, M + 0.3, 2.35, 11, 1.8, "Screen Privacy Through Head Orientation", size=50, color="F3F1EA", bold=True, line=1.0)
text(s, M + 0.3, 4.35, 10.5, 0.8, "Improving ResNet-18, and the LookAway Guard demo", size=20, color="C9D6DC")
text(s, M + 0.3, 6.35, 10, 0.4, "German University in Cairo", size=15, color="9FB7C4")

# 2 ---------- recap: the problem (as in Milestone 2) ----------
s = slide()
title(s, "Recap: the problem", next_n(), "Real BIWI crops, enlarged from 64 × 64. Labels come from the recorded head angles.")
text(s, M, 1.5, 5.6, 4.8, [
    [("Blur a laptop screen when its owner looks away.", {"size": 21, "bold": True, "color": DARK})],
    "In shared spaces, people nearby can read your screen. Today the user has to notice this and hide the screen by hand.",
    [("Task: ", {"bold": True, "color": DARK}), ("binary image classification. From one face image, decide ", {}),
     ("forward", {"bold": True, "color": BLUE}), (" or ", {}), ("away", {"bold": True, "color": ORANGE}), (".", {})],
    [("Rule: ", {"bold": True, "color": DARK}), ("forward if |yaw| ≤ 20° and |pitch| ≤ 20°, otherwise away.", {})],
    [("Yaw = turning left/right; pitch = tilting up/down; 0° = facing the camera", {"size": 13, "color": MUTED})],
], size=16, space_after=14)
faces = [("FORWARD", "yaw 0°, pitch 0°", BLUE), ("FORWARD", "yaw +12°, pitch −4°", BLUE), ("AWAY (side)", "yaw −50°, pitch −2°", ORANGE),
         ("AWAY (side)", "yaw +50°, pitch −1°", ORANGE), ("AWAY (up)", "yaw 0°, pitch −45°", ORANGE), ("AWAY (down)", "yaw 0°, pitch +37°", ORANGE)]
fx, fy, fs, gx, gy = 6.85, 1.5, 1.72, 0.2, 0.75
for k, (lab, ang, col) in enumerate(faces):
    x, y = fx + (k % 3) * (fs + gx), fy + (k // 3) * (fs + gy)
    picture(s, MS2_ASSETS / f"image-3-{k + 1}.png", x, y, w=fs, h=fs)
    text(s, x, y + fs + 0.08, fs + 0.1, 0.3, lab, size=13, color=col, bold=True)
    text(s, x, y + fs + 0.36, fs + 0.15, 0.3, ang, size=12, color=MUTED)

# 3 ---------- recap: data, model, pipeline ----------
s = slide()
title(s, "Recap: the data, the model and the app", next_n(),
      "Milestone 1 also planned an owner check (SFace). The demo leaves it out to stay simple.")
cw, ch, cy = 5.9, 2.75, 1.5
card(s, M, cy, cw, ch, "ResNet-18: the CNN we train", [
    "A convolutional network with 18 layers, pretrained on ImageNet.",
    "Its 1,000-class output is replaced by a new layer with 2 outputs: away, forward.",
    "Trained in 2 stages: first only the new layer, then also the last block.",
])
card(s, M + cw + 0.3, cy, cw, ch, "BIWI Kinect Head Pose: the dataset", [
    "13,219 face crops (64 × 64) of 20 people, each with measured head angles.",
    "Split by person: 12 train, 4 validation, 4 test. The test people are never seen in training.",
    "About 1 forward image for every 2 away images.",
])
text(s, M, 4.55, 6, 0.4, "The demo app, step by step", size=17, color=DARK, bold=True)
steps(s, 5.05, [("Camera frame", "laptop webcam", False), ("YuNet", "finds the face (pretrained)", False),
                ("64 × 64 crop", "cut like the training images", False), ("ResNet-18", "forward or away? (we train this)", True)])

# 4 ---------- starting point: Milestone 2 ----------
s = slide()
title(s, "Starting point: Milestone 2", next_n(),
      "Milestone 2: one run, seed 42. Re-run here with 3 seeds (42, 43, 44), same settings.")
stat(s, M, 1.5, 3.6, 2.2, f3(MS2["results"]["test"]["macro_f1"]),
     "test macro-F1 on 4 unseen people (Milestone 2).")
b_seeds = test(BASE, "macro_f1")
items = [
    ("It memorises the training people",
     f"Train macro-F1 {f3(MS2['results']['train']['macro_f1'])}, validation only {f3(MS2['results']['val']['macro_f1'])}. "
     "With 12 people it learns their faces, not just head direction."),
    ("One run can be lucky",
     f"Validation jumped between {min(ms2_s2):.2f} and {max(ms2_s2):.2f} from epoch to epoch. "
     f"Re-running the same model with 3 seeds gives test macro-F1 from {b_seeds.min():.3f} to {b_seeds.max():.3f}."),
    ("Errors sit at the 20° line",
     f"{pct(MS2['errors_within_5deg'] / MS2['errors_total'], 0)} of errors were within 5° of the line, "
     "where forward and away look almost the same."),
]
cw3 = W - 2 * M - 3.6 - 0.3
for k, (h_, b_) in enumerate(items):
    card(s, M + 3.9, 1.5 + k * 1.72, cw3, 1.55, h_, b_, size=13)
text(s, M, 4.05, 3.6, 2.4, [
    [("Our Milestone 2 plan was data augmentation. ", {"bold": True, "color": DARK}),
     ("The Milestone 3 rules do not count it as an improvement, so we changed the plan.", {})]], size=14)

# 5 ---------- the three improvements ----------
s = slide()
title(s, "Three improvements we tried", next_n(),
      "Changing hyperparameters, the loss or the metrics does not count as an improvement in Milestone 3.")
imps = [
    ("A: Small-input ResNet-18",
     ["What: the first layer keeps the full 64 × 64 crop (stride 1, no first max-pool) instead of reading a blurry 224 × 224 enlargement.",
      "Why: small faces lose detail when they are enlarged and then shrunk again inside the network."]),
    ("B: More training people",
     [f"What: add {lp_used} face crops of many different people from a second head-pose dataset (300W-LP) to training only.",
      "Why: with 12 people the model learns faces. More people force it to learn head direction."]),
    ("C: Attention blocks",
     ["What: a small Squeeze-and-Excitation block inside each of ResNet-18's 8 blocks. It gives each feature channel a weight.",
      "Why: lets the network focus on the features that describe head direction."]),
]
cw = (W - 2 * M - 0.6) / 3
for k, (h_, b_) in enumerate(imps):
    card(s, M + k * (cw + 0.3), 1.5, cw, 3.55, h_, b_, size=14)
box(s, M, 5.3, W - 2 * M, 1.2, fill="EEF3F6", line=None)
text(s, M + 0.3, 5.3, W - 2 * M - 0.6, 1.2, [[("The same for every experiment: ", {"bold": True, "color": DARK}),
     ("ImageNet-pretrained ResNet-18, 2-stage training, learning rates, epochs, batch size, loss, metrics, "
      "and the 12 / 4 / 4 person split of BIWI.", {})]], size=15, anchor=MSO_ANCHOR.MIDDLE)

# 6 ---------- how we compare them ----------
s = slide()
title(s, "How we compare them", next_n(), "The best epoch of each run is chosen on the 4 validation people, never on the test people.")
rows = [["Experiment", "A", "B", "C"]]
for name in NAMES:
    o = SUMMARY["experiments"][name]
    rows.append([name.replace(" (MS2)", " (Milestone 2)"), "✓" if o["small_input"] else "", "✓" if o["extra_300wlp"] else "",
                 "✓" if o["se"] else ""])
table(s, M, 1.5, 5.6, rows, [3.5, 0.7, 0.7, 0.7], size=14, row_h=0.46, align_right_from=1,
      highlight=[NAMES.index(FINAL) + 1], center=True)
x0 = M + 6.0
text(s, x0, 1.5, W - M - x0, 3.2, [
    [("Each experiment is trained 3 times ", {"bold": True, "color": DARK}),
     ("(seeds 42, 43, 44). We report the mean ± standard deviation, so one lucky run cannot decide.", {})],
    [("Same training as Milestone 2: ", {"bold": True, "color": DARK}),
     ("2 stages, best epoch chosen on the 4 validation people.", {})],
    [("Same metrics: ", {"bold": True, "color": DARK}), ("macro-F1 and accuracy on the 4 test people.", {})],
], size=15, space_after=12)
box(s, x0, 4.35, W - M - x0, 2.2, fill="EEF3F6", line=None)
text(s, x0 + 0.25, 4.5, W - M - x0 - 0.5, 2.0, [
    [("New: accepted and real errors", {"bold": True, "color": DARK, "size": 16})],
    [("Accepted error: ", {"bold": True, "color": DARK}),
     ("the head is between 17.5° and 22.5°, right at the 20° line. Nobody can see that difference.", {})],
    [("Real error: ", {"bold": True, "color": DARK}),
     ("a clearly forward head (below 17.5°) called away, or a clearly away head (above 22.5°) called forward.", {})],
], size=14, space_after=8)

# 7 ---------- evidence: the code ----------
s = slide()
title(s, "The code: how the model changes", next_n(), "Screenshots from ms3_improvements.ipynb, run on our laptop GPU.")
evidence(s, SHOTS / "code_model.png", M, 1.4, 7.4, 5.4, "Notebook section 4: the SE block (C) and build_model (A and C)")
x0 = M + 7.7
evidence(s, SHOTS / "code_300wlp.png", x0, 1.4, W - M - x0, 1.75, "Section 5: B only changes the training images")
evidence(s, SHOTS / "models_table.png", x0, 3.45, W - M - x0, 2.0, "Section 4 output: the 6 models")
text(s, x0, 5.75, W - M - x0, 1.0, "A changes 2 lines. C adds 8 small blocks (+89 k parameters). B changes no model code.",
     size=13, color=BODY)

# 8 ---------- evidence: data and training runs ----------
s = slide()
title(s, "The data and the training runs", next_n(),
      f"18 training runs in total, {minutes:.0f} minutes on an {device}.")
evidence(s, SHOTS / "data_counts.png", M, 1.4, 5.6, 2.6, "Section 3 output: images per set (BIWI counts match Milestones 1 and 2)")
evidence(s, FIG / "samples.png", M, 4.15, 5.6, 2.65, "Training images: BIWI (top) and 300W-LP (bottom)")
x0 = M + 5.9
evidence(s, SHOTS / "log_baseline.png", x0, 1.4, W - M - x0, 2.65, "Training log: baseline, seed 42")
evidence(s, SHOTS / "log_final.png", x0, 4.15, W - M - x0, 2.65, f"Training log: {FINAL}, seed 42")

# 9 ---------- results: all experiments ----------
s = slide()
title(s, "Results: every experiment", next_n(), "Test set: 2,029 images of 4 unseen people. Mean ± standard deviation over 3 seeds.")
evidence(s, SHOTS / "results_table.png", M, 1.4, W - 2 * M, 3.45, "Notebook section 6 output (real errors are counts out of 2,029)")
singles = [n_ for n_ in NAMES if n_[1:2] == ":"]
cw2 = (W - 2 * M - 0.3) / 2
card(s, M, 5.05, cw2, 1.75, "Alone, none of them helps on test",
     f"A, B and C each score {min(test(n_, 'macro_f1').mean() for n_ in singles):.3f}–"
     f"{max(test(n_, 'macro_f1').mean() for n_ in singles):.3f}, the baseline {test(BASE, 'macro_f1').mean():.3f}. "
     f"A and B do raise validation ({val(BASE).mean():.3f} → {val('A: Small input').mean():.3f} and {val('B: + 300W-LP').mean():.3f}).",
     size=13)
card(s, M + cw2 + 0.3, 5.05, cw2, 1.75, "Together, A + B is best",
     f"Test macro-F1 {pm3(test(FINAL, 'macro_f1'))} and the fewest real errors. "
     f"Adding C on top ({test('A + B + C', 'macro_f1').mean():.3f}) makes it worse.", size=13)

# 10 ---------- final model vs. baseline ----------
s = slide()
title(s, f"Final model ({FINAL}) vs. Milestone 2", next_n(), "Mean of 3 seeds on the 4 unseen test people.")
stat(s, M, 1.4, 4.1, 2.15, f"{real_b.mean():.0f} → {real_f.mean():.0f}",
     f"real errors per run ({(1 - real_f.mean() / real_b.mean()) * 100:.0f}% fewer). Spread between runs: ±{real_b.std():.0f} → ±{real_f.std():.0f}.")
rows = [["", "Baseline", FINAL],
        ["Test macro-F1", pm3(test(BASE, "macro_f1")), pm3(test(FINAL, "macro_f1"))],
        ["Test accuracy", pct(test(BASE, "accuracy").mean()), pct(test(FINAL, "accuracy").mean())],
        ["All errors", f"{test(BASE, 'errors').mean():.0f}", f"{test(FINAL, 'errors').mean():.0f}"],
        ["Accepted (17.5°–22.5°)", f"{test(BASE, 'accepted_errors').mean():.0f}", f"{test(FINAL, 'accepted_errors').mean():.0f}"],
        ["Real errors", f"{real_b.mean():.0f}", f"{real_f.mean():.0f}"],
        ["  forward called away", f"{test(BASE, 'forward_called_away').mean():.0f}", f"{test(FINAL, 'forward_called_away').mean():.0f}"],
        ["  away called forward", f"{test(BASE, 'away_called_forward').mean():.0f}", f"{test(FINAL, 'away_called_forward').mean():.0f}"]]
table(s, M, 3.8, 6.0, rows, [2.6, 1.7, 1.7], size=13, row_h=0.37, highlight=[5])
x0 = M + 6.4
picture(s, FIG / "confusion_matrices.png", x0, 1.4, W - M - x0, 3.5)
fa_b, fa_f = test(BASE, "forward_called_away").mean(), test(FINAL, "forward_called_away").mean()
text(s, x0, 5.05, W - M - x0, 1.7, [
    [(f"Forward heads called away: {fa_b:.0f} → {fa_f:.0f}. ", {"bold": True, "color": DARK}),
     ("Fewer screens blurred while the owner is reading, the costlier mistake.", {})],
    [("Accepted errors barely change. ", {"bold": True, "color": DARK}),
     ("They sit right at the 20° line, where no model can be sure.", {})]], size=14, space_after=8)

# 11 ---------- training curves ----------
s = slide()
title(s, "Training results", next_n(), "One line per seed. Dashed line = start of stage 2. Validation = 4 people not used for training.")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for name, color in [(BASE, "#8C9AA3"), (FINAL, "#3E7D5A")]:
    for k, r in enumerate(runs_of(name)):
        ep_ = range(1, len(r["history"]) + 1)
        label = ("Baseline" if name == BASE else FINAL) if k == 0 else None
        axes[0].plot(ep_, [h["val_f1"] for h in r["history"]], "o-", ms=3, color=color, label=label)
        axes[1].plot(ep_, [h["train_f1"] - h["val_f1"] for h in r["history"]], "o-", ms=3, color=color)
for ax, t_ in zip(axes, ["Validation macro-F1 (higher is better)", "Train − validation macro-F1 (lower is better)"]):
    ax.axvline(CFG["stage1_epochs"] + 0.5, ls="--", color="#7A858C", lw=1)
    ax.set_title(t_, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "slide_training_curves.png", dpi=200)
plt.close(fig)
picture(s, FIG / "slide_training_curves.png", M, 1.35, W - 2 * M, 3.35)
cw2 = (W - 2 * M - 0.3) / 2
card(s, M, 4.9, cw2, 1.85, "Steadier",
     f"Left graph: the green lines ({FINAL}) end higher and close together. "
     f"Best validation macro-F1 {pm3(val(BASE))} → {pm3(val(FINAL))}.", size=13)
card(s, M + cw2 + 0.3, 4.9, cw2, 1.85, "Less memorising",
     f"Right graph: the green lines stay lower. Train minus validation macro-F1 at the best epoch: "
     f"{gap(BASE).mean():.3f} → {gap(FINAL).mean():.3f}.", size=13)

# 12 ---------- errors that remain ----------
s = slide()
title(s, "Discussion: where errors remain", next_n(), "Test error rate by distance from the 20° line, mean of 3 seeds. Shaded = accepted band.")
picture(s, FIG / "error_by_distance.png", M + 0.6, 1.35, W - 2 * M - 1.2, 3.45)
cw3 = (W - 2 * M - 0.6) / 3
items = [("Borderline heads stay hard", f"About {test(FINAL, 'accepted_errors').mean():.0f} errors per run are inside 17.5°–22.5°, "
          f"for both models. Right at the line the label is almost a coin flip."),
         ("Clear cases improve", f"Real errors drop from {real_b.mean():.0f} to {real_f.mean():.0f}: "
          "the bars outside the shaded band are lower for the final model."),
         ("Small test set", "4 test people, about 2,000 images. A gain of 0.01–0.02 macro-F1 is real here, but modest.")]
for k, (h_, b_) in enumerate(items):
    card(s, M + k * (cw3 + 0.3), 4.95, cw3, 1.8, h_, b_, size=13)

# 13 ---------- demo ----------
s = slide()
title(s, "Demo: LookAway Guard", next_n(), "Desktop app (Electron), runs offline on the laptop. Live demo next.")
evidence(s, SHOTS / "app_demo.png", M, 1.4, 7.4, 5.4, "The app: live camera (left), result, face box and the 64 × 64 crop the model sees (right)")
x0 = M + 7.7
text(s, x0, 1.45, W - M - x0, 2.8, [
    [("Press Enter ", {"bold": True, "color": DARK}), ("(or the Capture button) to analyse one photo, or L for live mode.", {})],
    [("Same crop as training: ", {"bold": True, "color": DARK}),
     ("the face box from YuNet plus a 40% margin, resized to 64 × 64.", {})],
    [("Our model inside the app: ", {"bold": True, "color": DARK}),
     (f"exported to ONNX. It gives the same answer as the notebook on {int(onnx_same):,} of {n_test:,} test images.", {})],
], size=14, space_after=10)
evidence(s, SHOTS / "export_check.png", x0, 4.3, W - M - x0, 1.5, "Notebook section 9: export and check")

# 14 ---------- limitations and next steps ----------
s = slide()
title(s, "Limitations and next steps", next_n())
cw2 = (W - 2 * M - 0.3) / 2
acc_share = test(FINAL, "accepted_errors").mean() / test(FINAL, "errors").mean()
card(s, M, 1.5, cw2, 4.6, "Limitations", [
    "Only 4 test people. The score is a rough estimate and differs from person to person.",
    f"Heads near 20° stay hard: {pct(acc_share, 0)} of the final model's errors are in the accepted band.",
    "The model sees head direction, not the eyes. Eyes looking away with the head forward are missed.",
    "BIWI was recorded in one lab. A webcam at home has different light and camera angles.",
], head_color=ORANGE, size=16)
card(s, M + cw2 + 0.3, 1.5, cw2, 4.6, "Next steps", [
    "Collect a small test set with our own webcams to measure the app in real use.",
    "Add the owner check (SFace) so the screen only stays visible for its owner.",
    "Combine head direction with eye direction for heads near the line.",
    "Turn the demo into a background app that actually blurs the screen.",
], head_color=GREEN, size=16)

out = HERE.parent / "MS3_presentation_proposal.pptx"
prs.save(out)
print("wrote", out, "with", len(prs.slides), "slides")
