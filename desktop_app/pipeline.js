// The two-model pipeline: YuNet finds the face, our ResNet-18 says forward or away.
// Runs in Electron's main process with onnxruntime-node.
const path = require('path');
const ort = require('onnxruntime-node');

const DET_SIZE = 640;        // YuNet input is 640 x 640 (BGR, 0-255)
const DET_MIN_SCORE = 0.6;
const NMS_IOU = 0.3;
const MARGIN = 0.4;          // same face-box enlargement as the BIWI crops we trained on (FSA-Net, ad = 0.4)
const CROP = 64;             // our model reads 64 x 64 RGB crops

let detector, classifier;

async function loadModels(dir) {
  detector = await ort.InferenceSession.create(path.join(dir, 'face_detection_yunet_2023mar.onnx'));
  classifier = await ort.InferenceSession.create(path.join(dir, 'head_direction_resnet18.onnx'));
}

// Bilinear resize of a region of an RGBA frame into a CHW float tensor (like cv2.resize, INTER_LINEAR).
// The result fills the top-left outW x outH of a tensorW x tensorH tensor; the rest stays 0.
function resample(frame, x0, y0, cw, ch, outW, outH, bgr, tensorW = outW, tensorH = outH) {
  const { width: W, height: H, data } = frame;
  const plane = tensorW * tensorH;
  const out = new Float32Array(3 * plane);
  const sx = cw / outW, sy = ch / outH;
  for (let j = 0; j < outH; j++) {
    const fy = Math.min(Math.max(y0 + (j + 0.5) * sy - 0.5, 0), H - 1);
    const ya = Math.floor(fy), yb = Math.min(ya + 1, H - 1), wy = fy - ya;
    for (let i = 0; i < outW; i++) {
      const fx = Math.min(Math.max(x0 + (i + 0.5) * sx - 0.5, 0), W - 1);
      const xa = Math.floor(fx), xb = Math.min(xa + 1, W - 1), wx = fx - xa;
      const p00 = (ya * W + xa) * 4, p01 = (ya * W + xb) * 4, p10 = (yb * W + xa) * 4, p11 = (yb * W + xb) * 4;
      for (let c = 0; c < 3; c++) {
        const s = bgr ? 2 - c : c;
        out[c * plane + j * tensorW + i] =
          (data[p00 + s] * (1 - wx) + data[p01 + s] * wx) * (1 - wy) + (data[p10 + s] * (1 - wx) + data[p11 + s] * wx) * wy;
      }
    }
  }
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// Step 1: YuNet face detection. Returns faces in frame pixels, best first.
async function detectFaces(frame) {
  const scale = DET_SIZE / Math.max(frame.width, frame.height);
  const dw = Math.round(frame.width * scale), dh = Math.round(frame.height * scale);
  const input = resample(frame, 0, 0, frame.width, frame.height, dw, dh, true, DET_SIZE, DET_SIZE);
  const out = await detector.run({ input: new ort.Tensor('float32', input, [1, 3, DET_SIZE, DET_SIZE]) });
  const found = [];
  for (const stride of [8, 16, 32]) {
    const cls = out[`cls_${stride}`].data, obj = out[`obj_${stride}`].data, box = out[`bbox_${stride}`].data;
    const cols = DET_SIZE / stride;
    for (let k = 0; k < cls.length; k++) {
      const score = Math.sqrt(Math.min(Math.max(cls[k], 0), 1) * Math.min(Math.max(obj[k], 0), 1));
      if (score < DET_MIN_SCORE) continue;
      const row = Math.floor(k / cols), col = k % cols;
      const cx = (col + box[4 * k]) * stride, cy = (row + box[4 * k + 1]) * stride;
      const w = Math.exp(box[4 * k + 2]) * stride, h = Math.exp(box[4 * k + 3]) * stride;
      found.push({ x: (cx - w / 2) / scale, y: (cy - h / 2) / scale, w: w / scale, h: h / scale, score });
    }
  }
  found.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of found) if (kept.every((k) => iou(f, k) < NMS_IOU)) kept.push(f);
  return kept;
}

// Step 2: crop like the training data (face box + 40% margin, clipped to the frame), resize to 64 x 64.
function cropFace(frame, face) {
  const x1 = Math.max(Math.trunc(face.x - MARGIN * face.w), 0);
  const y1 = Math.max(Math.trunc(face.y - MARGIN * face.h), 0);
  const x2 = Math.min(Math.trunc(face.x + face.w + MARGIN * face.w), frame.width - 1);
  const y2 = Math.min(Math.trunc(face.y + face.h + MARGIN * face.h), frame.height - 1);
  const rect = { x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 };
  return { rect, tensor: resample(frame, rect.x, rect.y, rect.w, rect.h, CROP, CROP, false) };
}

// Full pipeline for one RGBA frame {width, height, data}.
async function analyze(frame) {
  const t0 = performance.now();
  const faces = await detectFaces(frame);
  if (faces.length === 0) return { face: null, ms: performance.now() - t0 };
  const face = faces.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a));   // the biggest face = the person at the screen
  const { rect, tensor } = cropFace(frame, face);
  const out = await classifier.run({ crop: new ort.Tensor('float32', tensor, [1, 3, CROP, CROP]) });
  const pForward = out.probs.data[1];
  const cropPixels = new Uint8ClampedArray(CROP * CROP * 4);              // RGBA copy of the crop, for the preview
  for (let p = 0; p < CROP * CROP; p++) {
    for (let c = 0; c < 3; c++) cropPixels[p * 4 + c] = tensor[c * CROP * CROP + p];
    cropPixels[p * 4 + 3] = 255;
  }
  return { face, faces: faces.length, crop: rect, cropPixels, pForward, label: pForward >= 0.5 ? 'forward' : 'away', ms: performance.now() - t0 };
}

module.exports = { loadModels, analyze, detectFaces, cropFace };
