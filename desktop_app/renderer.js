const MAX_WIDTH = 640;     // frames are scaled to 640 px wide, close to the 640 x 480 BIWI camera

const video = document.getElementById('video');
const statusEl = document.getElementById('status');
const captureBtn = document.getElementById('capture');
const liveBtn = document.getElementById('live');
const verdict = document.getElementById('verdict');
const verdictLabel = document.getElementById('verdict-label');
const verdictSub = document.getElementById('verdict-sub');
const barFill = document.getElementById('bar-fill');
const snapshot = document.getElementById('snapshot');
const cropCanvas = document.getElementById('crop');
const detInfo = document.getElementById('det-info');
const clsInfo = document.getElementById('cls-info');
const flash = document.getElementById('flash');

let live = false;
let busy = false;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    video.srcObject = stream;
    await video.play();
    statusEl.textContent = `Camera on: ${video.videoWidth} × ${video.videoHeight}`;
  } catch (err) {
    statusEl.textContent = `Camera not available: ${err.message}`;
    captureBtn.disabled = liveBtn.disabled = true;
  }
}

function grabFrame() {
  const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
  const w = Math.round(video.videoWidth * scale), h = Math.round(video.videoHeight * scale);
  snapshot.width = w; snapshot.height = h;
  const ctx = snapshot.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function drawBoxes(result) {
  const ctx = snapshot.getContext('2d');
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.setLineDash([5, 4]);
  const c = result.crop;
  ctx.strokeRect(c.x + 0.5, c.y + 0.5, c.w - 1, c.h - 1);          // the crop the model sees
  ctx.setLineDash([]);
  ctx.strokeStyle = result.label === 'forward' ? '#4fb3a9' : '#e0a56e';
  ctx.lineWidth = 3;
  const f = result.face;
  ctx.strokeRect(f.x, f.y, f.w, f.h);                               // the face YuNet found
}

function show(result) {
  const cctx = cropCanvas.getContext('2d');
  if (!result.face) {
    verdict.className = 'verdict none';
    verdictLabel.textContent = 'No face found';
    verdictSub.textContent = 'Move into the camera view and try again';
    barFill.style.width = '0%';
    detInfo.textContent = ', no face';
    clsInfo.textContent = '';
    cctx.clearRect(0, 0, 64, 64);
    return;
  }
  drawBoxes(result);
  cctx.putImageData(new ImageData(new Uint8ClampedArray(result.cropPixels), 64, 64), 0, 0);
  const pct = Math.round(result.pForward * 100);
  verdict.className = `verdict ${result.label}`;
  verdictLabel.textContent = result.label === 'forward' ? 'FORWARD' : 'AWAY';
  verdictSub.textContent = result.label === 'forward'
    ? `Looking at the screen (${pct}% forward)`
    : `Looking away: the screen would be blurred (${100 - pct}% away)`;
  barFill.style.width = `${pct}%`;
  barFill.className = result.label;
  detInfo.textContent = `, face score ${result.face.score.toFixed(2)}${result.faces > 1 ? `, ${result.faces} faces, biggest used` : ''}`;
  clsInfo.textContent = `, P(forward) = ${result.pForward.toFixed(3)}, ${Math.round(result.ms)} ms`;
}

async function analyzeOnce() {
  if (busy || !video.videoWidth) return;
  busy = true;
  try {
    const frame = grabFrame();
    const result = await window.pipeline.analyze({ width: frame.width, height: frame.height, data: frame.data });
    show(result);
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    busy = false;
  }
}

function capture() {
  flash.classList.remove('on'); void flash.offsetWidth; flash.classList.add('on');
  analyzeOnce();
}

async function liveLoop() {
  while (live) {
    await analyzeOnce();
    await new Promise((r) => setTimeout(r, 120));
  }
}

function setLive(on) {
  live = on;
  liveBtn.setAttribute('aria-pressed', String(on));
  liveBtn.firstChild.textContent = `Live mode: ${on ? 'on' : 'off'} `;
  if (on) liveLoop();
}

captureBtn.addEventListener('click', capture);
liveBtn.addEventListener('click', () => setLive(!live));
document.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  if (e.key === 'Enter') { e.preventDefault(); capture(); }
  if (e.key === 'l' || e.key === 'L') setLive(!live);
});

startCamera();
