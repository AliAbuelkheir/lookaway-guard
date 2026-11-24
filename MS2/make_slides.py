"""Builds ms2_slides.pptx from ms2_results.json and figures/ (written by ms2_resnet18_biwi.ipynb).

Run after the notebook:  python make_slides.py
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
R = json.loads((HERE / "ms2_results.json").read_text())
FIG, ASSETS = HERE / "figures", HERE / "assets"

# ---------- numbers used on the slides (all from the run) ----------
cfg, res, pc, cm = R["config"], R["results"], R["per_class"], R["test_confusion_matrix"]
counts, people = R["counts"], R["people"]
s2 = [h for h in R["history"] if h["stage"] == 2]
s1 = [h for h in R["history"] if h["stage"] == 1]
ep = lambda h: R["history"].index(h) + 1                 # epoch number as on the graph (1..15)
s1_best, s2_best = max(s1, key=lambda h: h["val_f1"]), max(s2, key=lambda h: h["val_f1"])
after_best = R["history"][ep(s2_best)] if ep(s2_best) < len(R["history"]) else None
fwd_miss, away_miss = cm[1][0], cm[0][1]                 # rows = true (away, forward), cols = predicted
n_fwd_test, n_away_test = sum(cm[1]), sum(cm[0])
pp = R["per_person"]
p_lo, p_hi = min(pp, key=lambda p: pp[p]["accuracy"]), max(pp, key=lambda p: pp[p]["accuracy"])
dev = R["device"]
hardware = f"Google Colab, {dev} GPU" if "T4" in dev else dev
f3 = lambda v: f"{v:.3f}"
pct = lambda v, d=1: f"{v * 100:.{d}f}%"

# ---------- style ----------
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
         space_after=6, line=1.1, name=None):
    """runs: str, or list of paragraphs; a paragraph is a str or a list of (text, {opts}) pieces."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
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
            f.italic = o.get("italic", False)
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


def title(s, t, n, footer):
    text(s, M, 0.45, W - 2 * M, 0.7, t, size=34, color=DARK, bold=True, name="Title")
    text(s, M, 6.95, W - 2 * M - 0.6, 0.3, footer, size=11, color=MUTED)
    text(s, W - M - 0.5, 6.95, 0.5, 0.3, str(n), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def table(s, x, y, w, rows, col_w, size=13, row_h=0.36, align_right_from=1, highlight=None):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(DARK if i == 0 else ("EEF3F6" if highlight == i else CARD))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j >= align_right_from else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0 or highlight == i
            r.font.color.rgb = rgb(WHITE if i == 0 else DARK)
    return t


def card(s, x, y, w, h, head, body, head_color=DARK, size=14):
    box(s, x, y, w, h)
    text(s, x + 0.25, y + 0.22, w - 0.5, 0.4, head, size=17, color=head_color, bold=True)
    text(s, x + 0.25, y + 0.68, w - 0.5, h - 0.85, body, size=size, color=BODY)


def picture(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def person_icon(s, cx, top, h, female, color):
    """Simple person icon: round head plus a dress (woman) or a rounded body (man)."""
    d = 0.3 * h
    shapes = [s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(top), Inches(d), Inches(d))]
    bt, bh = top + d + 0.05 * h, 0.65 * h
    if female:
        shapes.append(s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(cx - 0.33 * h), Inches(bt), Inches(0.66 * h), Inches(bh)))
    else:
        body = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx - 0.24 * h), Inches(bt), Inches(0.48 * h), Inches(bh))
        body.adjustments[0] = 0.3
        shapes.append(body)
    for shp in shapes:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(color)
        shp.line.fill.background()
        shp.shadow.inherit = False


FOOT = "Milestone 2: Screen privacy through head orientation"

# 1 ---------- title ----------
s = slide(DARK)
text(s, M + 0.3, 0.9, 10, 0.4, "DMET 901 Computer Vision, Milestone 2", size=15, color="9FB7C4")
text(s, M + 0.3, 2.35, 11, 1.8, "Screen Privacy Through Head Orientation", size=50, color="F3F1EA", bold=True, line=1.0)
text(s, M + 0.3, 4.35, 10.5, 0.8, "Fine-tuning ResNet-18 on BIWI",
     size=20, color="C9D6DC")
text(s, M + 0.3, 6.35, 10, 0.4, "German University in Cairo", size=15, color="9FB7C4")

# 2 ---------- recap: the problem ----------
s = slide()
title(s, "Recap: the problem", 2, "Real BIWI crops, enlarged from 64 × 64. Labels come from the recorded head angles.")
text(s, M, 1.5, 5.6, 4.8, [
    [("Blur a laptop screen when its owner looks away.", {"size": 21, "bold": True, "color": DARK})],
    "In shared spaces, people nearby can read your screen. Today the user has to notice this and hide the screen by hand.",
    [("Task: ", {"bold": True, "color": DARK}), ("binary image classification. From one face image, decide ", {}),
     ("forward", {"bold": True, "color": BLUE}), (" or ", {}), ("away", {"bold": True, "color": ORANGE}), (".", {})],
    [("Rule: ", {"bold": True, "color": DARK}),
     ("forward if |yaw| ≤ 20° and |pitch| ≤ 20°, otherwise away.", {})],
    [("Yaw = turning left/right; pitch = tilting up/down; 0° = facing the camera", {"size": 13, "color": MUTED})],
], size=16, space_after=14)
faces = [("FORWARD", "yaw 0°, pitch 0°", BLUE), ("FORWARD", "yaw +12°, pitch −4°", BLUE), ("AWAY (side)", "yaw −50°, pitch −2°", ORANGE),
         ("AWAY (side)", "yaw +50°, pitch −1°", ORANGE), ("AWAY (up)", "yaw 0°, pitch −45°", ORANGE), ("AWAY (down)", "yaw 0°, pitch +37°", ORANGE)]
fx, fy, fs, gx, gy = 6.85, 1.5, 1.72, 0.2, 0.75
for k, (lab, ang, col) in enumerate(faces):
    x, y = fx + (k % 3) * (fs + gx), fy + (k // 3) * (fs + gy)
    picture(s, ASSETS / f"image-3-{k + 1}.png", x, y, w=fs, h=fs)
    text(s, x, y + fs + 0.08, fs + 0.1, 0.3, lab, size=13, color=col, bold=True)
    text(s, x, y + fs + 0.36, fs + 0.15, 0.3, ang, size=12, color=MUTED)

# 3 ---------- recap: the CNN and the dataset ----------
s = slide()
title(s, "Recap: the CNN and the dataset", 3, "")
cw, ch, cy = 5.9, 3.05, 1.5
card(s, M, cy, cw, ch, "ResNet-18: the CNN we train", [
    "A convolutional network with 18 layers.",
    "Pretrained on ImageNet, so it already knows general image features. Useful with only 20 people.",
    "We replace its 1,000-class output with a new layer: 512 features → 2 scores (away, forward).",
    "11.2 million parameters in total.",
], head_color=BLUE)
card(s, M + cw + 0.3, cy, cw, ch, "BIWI Kinect Head Pose: the dataset", [
    "20 people (6 women, 14 men) in 24 recordings; 4 people were recorded twice.",
    "13,219 colour face crops of 64 × 64 pixels (FSA-Net subset).",
    "Every crop has measured head angles (HeadPose+, calibrated). Our 20° rule turns them into labels.",
    "4,270 forward (32%) and 8,949 away (68%): about 2 : 1 imbalance.",
], head_color=BLUE)
# pipeline strip
py, steps = 4.95, [("Camera frame", "webcam image", False), ("YuNet", "finds the face (pretrained)", False),
                   ("SFace", "is it the owner? (pretrained)", False), ("ResNet-18", "forward or away? (we train this)", True)]
sg = 0.33; sw = (W - 2 * M - 3 * sg) / 4
for k, (a, b, ours) in enumerate(steps):
    x = M + k * (sw + sg)
    box(s, x, py, sw, 1.05, fill=DARK if ours else CARD, line=None if ours else LINE)
    text(s, x + 0.2, py + 0.16, sw - 0.4, 0.35, a, size=16, bold=True, color="F3F1EA" if ours else DARK)
    text(s, x + 0.2, py + 0.55, sw - 0.4, 0.35, b, size=12, color="C9D6DC" if ours else MUTED)
    if k < 3:
        text(s, x + sw, py + 0.3, sg, 0.4, "→", size=18, color=MUTED, align=PP_ALIGN.CENTER)
text(s, M, 6.15, W - 2 * M, 0.3, "The full app pipeline from Milestone 1. Milestone 2 trains and evaluates the ResNet-18 step only.",
     size=12, color=MUTED)

# 4 ---------- setup: who is in each split + preprocessing ----------
s = slide()
title(s, "Recap: data split and preprocessing", 4,
      "Split by person, seed 42, same as Milestone 1. Each icon is one person (dress = woman).")
SPLIT_COLOR = {"train": (BLUE, "E8F0F5"), "val": (ORANGE, "F7EDE4"), "test": (GREEN, "E6F1EA")}
SPLIT_NAME = {"train": "Train", "val": "Validation", "test": "Test"}
fs, fg, pad = 0.88, 0.15, 0.22
cols = {"train": 6, "val": 2, "test": 2}
widths = {k: cols[k] * fs + (cols[k] - 1) * fg + 2 * pad for k in cols}
gap = (W - 2 * M - sum(widths.values())) / 2
x, py, ph = M, 1.45, 3.5
for k in ("train", "val", "test"):
    col, tint = SPLIT_COLOR[k]
    box(s, x, py, widths[k], ph, fill=tint, line=None, radius=0.05)
    n_f = sum(p.startswith("F") for p in people[k])
    n_m = len(people[k]) - n_f
    text(s, x + pad, py + 0.18, widths[k] - 2 * pad, 0.35, SPLIT_NAME[k], size=17, color=col, bold=True)
    text(s, x + pad, py + 0.52, widths[k] - 2 * pad, 0.5,
         [f"{len(people[k])} people: {n_f} {'woman' if n_f == 1 else 'women'}, {n_m} {'man' if n_m == 1 else 'men'}",
          f"{counts[k]['images']:,} images, {pct(counts[k]['forward'] / counts[k]['images'], 0)} forward"],
         size=12, color=BODY, space_after=0)
    for j, person in enumerate(people[k]):                      # women first, then men (IDs are sorted F.., M..)
        cx = x + pad + (j % cols[k]) * (fs + fg) + fs / 2
        person_icon(s, cx, py + 1.2 + (j // cols[k]) * (fs + 0.2), fs, person.startswith("F"), col)
    x += widths[k] + gap
text(s, M, 5.32, 6, 0.4, "Preprocessing, applied to every image", size=17, color=DARK, bold=True)
chips = ["BGR → RGB", "Scale pixels to 0–1", "Resize 64 → 224", "ImageNet normalisation"]
chg = 0.33; chw = (W - 2 * M - 3 * chg) / 4
for k, c in enumerate(chips):
    cx = M + k * (chw + chg)
    box(s, cx, 5.8, chw, 0.62, fill="E6EEF3", line=None, radius=0.25)
    text(s, cx, 5.8, chw, 0.62, c, size=15, color=DARK, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if k < 3:
        text(s, cx + chw, 5.91, chg, 0.4, "→", size=18, color=MUTED, align=PP_ALIGN.CENTER)
text(s, M, 6.52, W - 2 * M, 0.3, "Each person is in one split only, so near-identical video frames cannot leak into validation or test. "
     "No data augmentation in this run.", size=12, color=MUTED)

# 5 ---------- run configuration ----------
s = slide()
title(s, "Run configuration", 5, "All settings follow the Milestone 1 plan; none were tuned on the test set")
rows = [["Setting", "Value"],
        ["Model", "ResNet-18, ImageNet weights, new 2-class layer"],
        ["Optimizer, batch size", f"{cfg['optimizer']}, {cfg['batch_size']} images"],
        ["Epochs", f"{cfg['stage1_epochs']} (stage 1) + {cfg['stage2_epochs']} (stage 2)"],
        ["Learning rate", f"{cfg['stage1_lr']:g}, then {cfg['stage2_lr']:g}"],
        ["Loss", f"Weighted cross-entropy: away {R['class_weights']['away']:.2f}, forward {R['class_weights']['forward']:.2f}"],
        ["Model selection", "Best validation macro-F1 after each epoch"],
        ["Seed", f"{cfg['seed']}"],
        ["Hardware, time", f"{hardware}, {R['train_minutes']:.1f} min training"]]
table(s, M, 1.5, 7.2, rows, [2.3, 4.9], size=14, row_h=0.47, align_right_from=9)
x0, w0 = 8.15, W - M - 8.15
card(s, x0, 1.5, w0, 1.6, f"Stage 1: {cfg['stage1_epochs']} epochs, learning rate {cfg['stage1_lr']:g}",
     "Backbone frozen. Only the new 2-class layer learns (1,026 parameters).", head_color=BLUE)
text(s, x0, 3.12, w0, 0.4, "↓", size=20, color=MUTED, align=PP_ALIGN.CENTER)
card(s, x0, 3.55, w0, 1.6, f"Stage 2: {cfg['stage2_epochs']} epochs, learning rate {cfg['stage2_lr']:g}",
     "The last block (layer4) is fine-tuned too, with smaller steps (8.4 of 11.2 M parameters).", head_color=BLUE)
text(s, x0, 5.45, w0, 1.3, [[("Class weights: ", {"bold": True, "color": DARK}),
     ("training images ÷ (2 × images of that class), from training people only. Missing a forward head costs "
      f"{R['class_weights']['forward'] / R['class_weights']['away']:.1f}× more than missing an away head.", {})]], size=13)

# 6 ---------- results ----------
s = slide()
title(s, "Results on 4 unseen people", 6, f"Best checkpoint: epoch {ep(s2_best)} (chosen on validation). One run, seed {cfg['seed']}.")
box(s, M, 1.5, 3.6, 2.15, fill=DARK, line=None)
text(s, M + 0.3, 1.68, 3.1, 0.9, f3(res["test"]["macro_f1"]), size=50, color=SAND, bold=True)
text(s, M + 0.3, 2.65, 3.1, 0.9, f"test macro-F1. Always answering “away” scores {f3(res['always_away_test']['macro_f1'])}.",
     size=14, color="E8ECEE")
rows = [["Set", "Macro-F1", "Accuracy"],
        ["Train (12 people)", f3(res["train"]["macro_f1"]), pct(res["train"]["accuracy"])],
        ["Validation (4)", f3(res["val"]["macro_f1"]), pct(res["val"]["accuracy"])],
        ["Test (4)", f3(res["test"]["macro_f1"]), pct(res["test"]["accuracy"])],
        ["Always away (test)", f3(res["always_away_test"]["macro_f1"]), pct(res["always_away_test"]["accuracy"])]]
table(s, M, 3.95, 5.9, rows, [2.7, 1.6, 1.6], size=14, row_h=0.42, highlight=3)
text(s, M, 6.2, 5.9, 0.6, f"Forward: precision {f3(pc['forward']['precision'])}, recall {f3(pc['forward']['recall'])}. "
     f"Away: precision {f3(pc['away']['precision'])}, recall {f3(pc['away']['recall'])}", size=13, color=MUTED)
picture(s, FIG / "confusion_matrix.png", 6.85, 1.45, h=3.6)
x0 = 6.85
text(s, x0, 5.2, W - M - x0, 1.6, [
    [(f"{fwd_miss} of {n_fwd_test:,} forward frames ({pct(fwd_miss / n_fwd_test)}) ", {"bold": True, "color": DARK}),
     ("would blur the screen while the owner is reading. This is the costlier error: it interrupts work.", {})],
    [(f"{away_miss} of {n_away_test:,} away frames ({pct(away_miss / n_away_test)}) ", {"bold": True, "color": DARK}),
     ("would leave the screen visible for a moment. Cheaper: usually brief, and the next frames catch it.", {})]],
     size=14, space_after=8)

# 7 ---------- discussion: training ----------
s = slide()
title(s, "Training results", 7, "Dashed line = start of stage 2. Validation = 4 people not used for training.")
picture(s, FIG / "training_curves.png", (W - 3.25 * 2200 / 760) / 2, 1.4, h=3.25)
cw = (W - 2 * M - 0.6) / 3
unstable = (f"The orange line on the right zigzags: {s2_best['val_f1']:.2f} at epoch {ep(s2_best)}, "
            f"{after_best['val_f1']:.2f} at epoch {ep(after_best)}. " if after_best else
            "The orange line on the right zigzags from epoch to epoch. ")
items = [("Fine-tuning helps", f"Right graph, orange line: best before the dashed line is {f3(s1_best['val_f1'])} "
          f"(epoch {ep(s1_best)}), best after it is {f3(s2_best['val_f1'])} (epoch {ep(s2_best)})."),
         ("But it is unstable", unstable + f"We kept the best epoch, so {f3(s2_best['val_f1'])} is partly a lucky peak.")]
cw2 = (W - 2 * M - 0.3) / 2
for k, (h_, b_) in enumerate(items):
    card(s, M + k * (cw2 + 0.3), 4.85, cw2, 1.85, h_, b_, head_color=BLUE, size=13)

# 8 ---------- discussion: errors ----------
s = slide()
title(s, "Discussion: errors sit at the 20° line", 8, "Each dot is one test image at its true head angles. The box is the forward region.")
picture(s, FIG / "error_analysis.png", (W - 3.3 * 2400 / 920) / 2, 1.35, h=3.3)
items = [("Most errors are borderline", f"{pct(R['errors_within_5deg'] / R['errors_total'], 0)} of errors are within 5° of the line, "
          f"which holds only {pct(R['near_share'], 0)} of test images. A head at 19° and one at 21° look alike but have opposite labels."),
         ("Clear cases are reliable", f"Error rate is {pct(R['error_rate_near'])} within 5° of the line and "
          f"{pct(R['error_rate_far'])} further away; almost no errors once the head is 10° outside the box."),
         ("It depends on the person", f"Test accuracy ranges from {pct(pp[p_lo]['accuracy'], 0)} ({p_lo}) to "
          f"{pct(pp[p_hi]['accuracy'], 0)} ({p_hi}). With 4 test people the score is a rough estimate.")]
for k, (h_, b_) in enumerate(items):
    card(s, M + k * (cw + 0.3), 4.85, cw, 1.85, h_, b_, head_color=BLUE, size=13)

# 9 ---------- proposed improvement ----------
s = slide()
title(s, "Proposed improvement: data augmentation", 9, "Proposed in Milestone 2, to be run and measured in Milestone 3")
card(s, M, 1.5, 4.6, 2.05, "What", [
    "During training only, each image is randomly mirrored left–right (50%) and made brighter or darker, "
    "with more or less contrast (× 0.8–1.2). Validation and test images stay unchanged."], head_color=ORANGE, size=14)
card(s, M, 3.85, 4.6, 2.45, "Why", [
    f"The main problem is overfitting: train {f3(res['train']['macro_f1'])} vs. validation {f3(res['val']['macro_f1'])}. "
    "Images that change every epoch are harder to memorise. They keep their label: the rule uses |yaw|, and lighting "
    "does not move the head."], head_color=ORANGE, size=14)
x0 = M + 4.9
text(s, x0, 1.5, W - M - x0, 0.4, "What we expect to see in Milestone 3", size=17, color=DARK, bold=True)
rows = [["Milestone 2 result", "Expected with augmentation"],
        [f"Train {f3(res['train']['macro_f1'])} vs. val {f3(res['val']['macro_f1'])}", "Smaller gap"],
        ["Validation score jumps per epoch", "Steadier curve, less luck"],
        [f"Test macro-F1 {f3(res['test']['macro_f1'])}", "Higher on unseen people"],
        [f"Per-person accuracy {pct(pp[p_lo]['accuracy'], 0)}–{pct(pp[p_hi]['accuracy'], 0)}", "Smaller spread between people"],
        [f"{pct(R['errors_within_5deg'] / R['errors_total'], 0)} of errors near the 20° line", "Little change: borderline heads stay hard"]]
table(s, x0, 2.0, W - M - x0, rows, [3.6, W - M - x0 - 3.6], size=14, row_h=0.5, align_right_from=9)
text(s, x0, 5.3, W - M - x0, 1.4, [[("How we will check it: ", {"bold": True, "color": DARK}),
     ("same split, settings and seed as this run, with augmentation as the only change. We compare validation macro-F1 "
      "and use the test people once, at the end.", {})]], size=14)

out = HERE / "ms2_slides.pptx"
prs.save(out)
print("wrote", out, "with", len(prs.slides), "slides; hardware:", hardware)
