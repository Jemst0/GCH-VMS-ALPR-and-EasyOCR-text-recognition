const cameraVideo = document.getElementById("cameraVideo");
const previewImage = document.getElementById("previewImage");
const overlayCanvas = document.getElementById("overlayCanvas");
const emptyState = document.getElementById("emptyState");
const stage = document.getElementById("stage");
const cameraStatus = document.getElementById("cameraStatus");
const sourceLabel = document.getElementById("sourceLabel");
const confidenceLabel = document.getElementById("confidenceLabel");
const detectionCount = document.getElementById("detectionCount");
const plateText = document.getElementById("plateText");
const inferenceMode = document.getElementById("inferenceMode");
const resultsList = document.getElementById("resultsList");
const thumbnailStrip = document.getElementById("thumbnailStrip");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const demoBtn = document.getElementById("demoBtn");
const imageUpload = document.getElementById("imageUpload");

const detectionApiBaseUrl = "http://localhost:8000";

let cameraStream = null;
let currentSource = null;
let currentObjectUrl = null;
let currentDetections = [];
let liveDetectionTimer = null;
let liveDetectionBusy = false;

const liveDetectionIntervalMs = 900;

const demoDetections = [
  { plate: "ABC-1234", confidence: 0.97, lane: "Front plate", box: [0.36, 0.58, 0.28, 0.16] },
  { plate: "ABC-1234", confidence: 0.92, lane: "OCR readback", box: [0.37, 0.60, 0.26, 0.13] },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getActiveMediaElement() {
  if (cameraVideo.classList.contains("active") && cameraVideo.videoWidth && cameraVideo.videoHeight) {
    return cameraVideo;
  }

  if (previewImage.classList.contains("active") && previewImage.naturalWidth && previewImage.naturalHeight) {
    return previewImage;
  }

  return null;
}

function getDisplayedMediaRect() {
  const media = getActiveMediaElement();
  const stageRect = stage.getBoundingClientRect();

  if (!media) {
    return {
      x: 0,
      y: 0,
      width: stageRect.width,
      height: stageRect.height,
    };
  }

  const mediaWidth = media instanceof HTMLVideoElement ? media.videoWidth : media.naturalWidth;
  const mediaHeight = media instanceof HTMLVideoElement ? media.videoHeight : media.naturalHeight;
  const stageRatio = stageRect.width / stageRect.height;
  const mediaRatio = mediaWidth / mediaHeight;

  if (mediaRatio > stageRatio) {
    const height = stageRect.height;
    const width = height * mediaRatio;
    return {
      x: (stageRect.width - width) / 2,
      y: 0,
      width,
      height,
    };
  }

  const width = stageRect.width;
  const height = width / mediaRatio;
  return {
    x: 0,
    y: (stageRect.height - height) / 2,
    width,
    height,
  };
}

function normalizeDetectionResponse(payload) {
  const detections = Array.isArray(payload?.detections) ? payload.detections : [];

  return detections.map((detection, index) => {
    const box = detection.box ?? detection.bbox ?? detection.xywh ?? detection.coordinates ?? [0.36, 0.58, 0.28, 0.16];
    const [x = 0, y = 0, width = 0, height = 0] = Array.isArray(box)
      ? box
      : [box.x ?? box.left ?? 0, box.y ?? box.top ?? 0, box.w ?? box.width ?? 0, box.h ?? box.height ?? 0];

    const label = String(detection.class_name ?? detection.label ?? detection.name ?? "");
    const isPlateLike = /plate|license|licence/i.test(label);
    const plateText = [detection.text, isPlateLike ? payload.plate_text : null]
      .map((value) => (typeof value === "string" ? value.trim() : ""))
      .find((value) => value.length > 0);
    const ocr_source = detection.ocr_source ?? null;

    return {
      plate: plateText ?? `Detection ${index + 1}`,
      confidence: Number(detection.confidence ?? detection.score ?? detection.conf ?? 0),
      lane: detection.class_name ?? detection.label ?? "Detected plate",
      ocr_source: ocr_source,
      label: label || `class_${index + 1}`,
      box: [Number(x), Number(y), Number(width), Number(height)],
    };
  });
}

async function requestDetections(blob) {
  const formData = new FormData();
  formData.append("image", blob, "frame.png");

  const response = await fetch(`${detectionApiBaseUrl}/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Detection API responded with ${response.status}`);
  }

  return normalizeDetectionResponse(await response.json());
}

function stopLiveDetectionLoop() {
  if (liveDetectionTimer) {
    clearInterval(liveDetectionTimer);
    liveDetectionTimer = null;
  }

  liveDetectionBusy = false;
}

async function captureVideoFrameBlob() {
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = cameraVideo.videoWidth || 1280;
  captureCanvas.height = cameraVideo.videoHeight || 720;

  const context = captureCanvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

  return new Promise((resolve) => {
    captureCanvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.88);
  });
}

async function runLiveDetection() {
  if (!cameraStream || liveDetectionBusy || cameraVideo.readyState < 2) {
    return;
  }

  liveDetectionBusy = true;
  updateStatus("Detecting live frame", "live");

  try {
    const frameBlob = await captureVideoFrameBlob();
    if (!frameBlob) {
      return;
    }

    const detections = await requestDetections(frameBlob);
    const activeDetections = detections.length ? detections : [];
    currentDetections = activeDetections;
    drawDetections(activeDetections);
    renderResults(activeDetections, "Live camera");
    setSourceLabel("Live camera");
    updateStatus("Live camera running", "live");
  } catch (error) {
    console.error(error);
    currentDetections = demoDetections;
    drawDetections(currentDetections);
    renderResults(currentDetections, "Live camera · demo fallback");
    updateStatus("Live fallback", "warn");
  } finally {
    liveDetectionBusy = false;
  }
}

async function renderPreviewAndDetect(blob, labelText) {
  const previewUrl = URL.createObjectURL(blob);
  currentObjectUrl = previewUrl;
  currentSource = "image";
  previewImage.src = previewUrl;

  previewImage.onload = async () => {
    showActiveStage("image");
    setSourceLabel(labelText);
    updateStatus("Running model", "live");

    try {
      const detections = await requestDetections(blob);
      const activeDetections = detections.length ? detections : [];
      currentDetections = activeDetections;
      drawDetections(activeDetections);
      renderResults(activeDetections, labelText);
      updateStatus("Model ready", "live");
    } catch (error) {
      console.error(error);
      currentDetections = demoDetections;
      drawDetections(currentDetections);
      renderResults(currentDetections, `${labelText} · demo fallback`);
      updateStatus("Demo fallback", "warn");
    }
  };
}

function updateStatus(message, tone = "idle") {
  cameraStatus.textContent = message;
  cameraStatus.dataset.tone = tone;
  cameraStatus.style.background =
    tone === "live" ? "rgba(73, 216, 197, 0.14)" : tone === "warn" ? "rgba(255, 204, 102, 0.14)" : "rgba(124, 198, 255, 0.14)";
}

function setSourceLabel(label) {
  sourceLabel.textContent = label;
}

function showActiveStage(kind) {
  cameraVideo.classList.toggle("active", kind === "camera");
  previewImage.classList.toggle("active", kind === "image");
  emptyState.style.display = kind ? "none" : "block";
}

function resizeCanvas() {
  const rect = stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  overlayCanvas.width = Math.max(1, Math.round(rect.width * ratio));
  overlayCanvas.height = Math.max(1, Math.round(rect.height * ratio));
}

function clearCanvas() {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function drawDetections(detections) {
  resizeCanvas();
  const ctx = overlayCanvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = overlayCanvas.width / ratio;
  const height = overlayCanvas.height / ratio;
  const mediaRect = getDisplayedMediaRect();

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  detections.forEach((detection, index) => {
    const [x, y, w, h] = detection.box;
    const px = mediaRect.x + x * mediaRect.width;
    const py = mediaRect.y + y * mediaRect.height;
    const pw = w * mediaRect.width;
    const ph = h * mediaRect.height;

    ctx.lineWidth = Math.max(2, width * 0.004);
    ctx.strokeStyle = index === 0 ? "#49d8c5" : "#7cc6ff";
    ctx.fillStyle = index === 0 ? "rgba(73, 216, 197, 0.14)" : "rgba(124, 198, 255, 0.14)";
    ctx.shadowBlur = 16;
    ctx.shadowColor = ctx.strokeStyle;
    ctx.strokeRect(px, py, pw, ph);
    ctx.fillRect(px, py, pw, ph);

    const tagText = detection.plate.startsWith("Detection ") ? detection.label : detection.plate;
    const tag = `${tagText} • ${(detection.confidence * 100).toFixed(1)}%`;
    const tagWidth = ctx.measureText(tag).width + 20;
    const tagHeight = 28;
    const tagX = clamp(px, 10, Math.max(10, width - tagWidth - 10));
    const tagY = Math.max(10, py - tagHeight - 8);

    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(7, 12, 23, 0.92)";
    ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
    ctx.strokeStyle = ctx.shadowColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(tagX, tagY, tagWidth, tagHeight);
    ctx.fillStyle = "#edf2ff";
    ctx.font = "600 13px Bahnschrift, Trebuchet MS, sans-serif";
    ctx.fillText(tag, tagX + 10, tagY + 19);
  });
}

function renderResults(detections, sourceLabelText) {
  currentDetections = detections;
  detectionCount.textContent = String(detections.length);
  plateText.textContent = detections[0]?.plate ?? "--";
  confidenceLabel.textContent = detections.length ? `${Math.round((detections[0]?.confidence ?? 0) * 100)}% confidence` : "--";
  inferenceMode.textContent = sourceLabelText;

  resultsList.innerHTML = detections
    .map(
      (detection, index) => `
        <article>
          <div class="result-top">
            <strong>Plate ${index + 1}: ${detection.plate}</strong>
            <span class="confidence">${Math.round(detection.confidence * 100)}%</span>
          </div>
          <div class="result-meta">${detection.label} | ${detection.lane} | ${sourceLabelText} | OCR: ${detection.ocr_source ?? 'none'}</div>
          <div class="result-boxes">
            <span class="box-tag">Vehicle</span>
            <span class="box-tag">License plate</span>
            <span class="box-tag">OCR text</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderThumbnails(files) {
  thumbnailStrip.innerHTML = files
    .map(
      (file) => `
        <div class="thumb">
          <img src="${file.preview}" alt="${file.name}" />
          <div>
            <strong>${file.name}</strong>
            <div class="thumbnail-meta">${file.size}</div>
          </div>
        </div>
      `,
    )
    .join("");
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function startCamera() {
  stopCamera();

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    showActiveStage("camera");
    resizeCanvas();
    setSourceLabel("Live camera");
    updateStatus("Camera active", "live");
    currentSource = "camera";
    currentDetections = [];
    renderResults([], "Live camera");
    drawDetections([]);

    liveDetectionTimer = window.setInterval(runLiveDetection, liveDetectionIntervalMs);
    runLiveDetection();
  } catch (error) {
    updateStatus("Camera blocked", "warn");
    setSourceLabel("Camera unavailable");
    console.error(error);
  }
}

function stopCamera() {
  stopLiveDetectionLoop();

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  cameraVideo.pause();
  cameraVideo.srcObject = null;

  if (currentSource === "camera") {
    currentSource = null;
    currentDetections = [];
    showActiveStage(null);
    clearCanvas();
    setSourceLabel("No source loaded");
    updateStatus("Idle", "idle");
    confidenceLabel.textContent = "--";
    detectionCount.textContent = "0";
    plateText.textContent = "--";
    inferenceMode.textContent = "Waiting";
    resultsList.innerHTML = "";
  }
}

function captureFrame() {
  if (!cameraStream) {
    updateStatus("Start the camera first", "warn");
    return;
  }

  stopLiveDetectionLoop();

  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = cameraVideo.videoWidth || 1280;
  captureCanvas.height = cameraVideo.videoHeight || 720;
  captureCanvas.getContext("2d").drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

  const snapshotUrl = captureCanvas.toDataURL("image/png");
  showActiveStage("image");
  previewImage.src = snapshotUrl;
  setSourceLabel("Captured frame");
  updateStatus("Frame captured", "live");
  currentSource = "capture";
  currentDetections = demoDetections;
  drawDetections(currentDetections);
  renderResults(currentDetections, "Captured frame");
}

function loadDemoScene() {
  const demoSvg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
      <defs>
        <linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#19263e" />
          <stop offset="100%" stop-color="#060b14" />
        </linearGradient>
        <linearGradient id="body" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#8fa4c7" />
          <stop offset="100%" stop-color="#4b5871" />
        </linearGradient>
      </defs>
      <rect width="1280" height="720" fill="url(#sky)" />
      <circle cx="1020" cy="130" r="54" fill="#ffe7a3" opacity="0.95" />
      <rect x="0" y="470" width="1280" height="250" fill="#0d1320" />
      <rect x="138" y="336" width="992" height="196" rx="58" fill="url(#body)" />
      <rect x="168" y="392" width="730" height="112" rx="28" fill="#202737" />
      <rect x="904" y="388" width="140" height="66" rx="18" fill="#18202d" />
      <rect x="876" y="430" width="236" height="66" rx="20" fill="#fcfdff" />
      <rect x="940" y="454" width="120" height="20" rx="8" fill="#111827" />
      <rect x="238" y="474" width="140" height="42" rx="10" fill="#f6bf4f" />
      <rect x="330" y="238" width="144" height="58" rx="12" fill="#d7e0ee" opacity="0.22" />
      <rect x="488" y="226" width="162" height="62" rx="12" fill="#d7e0ee" opacity="0.22" />
      <rect x="472" y="470" width="128" height="38" rx="18" fill="#0d1119" opacity="0.85" />
      <rect x="690" y="476" width="112" height="38" rx="18" fill="#0d1119" opacity="0.85" />
      <rect x="113" y="422" width="124" height="30" rx="15" fill="#fcfdff" opacity="0.8" />
      <text x="108" y="90" fill="#d5ebff" font-family="Bahnschrift, Trebuchet MS, sans-serif" font-size="42">ALPR Demo Scene</text>
      <text x="108" y="132" fill="#92a4c4" font-family="Bahnschrift, Trebuchet MS, sans-serif" font-size="22">Bounding boxes and plate OCR placeholders</text>
    </svg>
  `;

  renderPreviewAndDetect(new Blob([demoSvg], { type: "image/svg+xml" }), "Demo scene");
}

function handleUpload(event) {
  const files = [...(event.target.files || [])];
  if (!files.length) {
    return;
  }

  const preparedFiles = files.map((file) => ({
    name: file.name,
    size: formatFileSize(file.size),
    preview: URL.createObjectURL(file),
  }));

  renderThumbnails(preparedFiles);
  const firstFile = preparedFiles[0];
  currentSource = "upload";
  renderPreviewAndDetect(files[0], `Uploaded file: ${firstFile.name}`);
}

startCameraBtn.addEventListener("click", async () => {
  await startCamera();
});

stopCameraBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", captureFrame);
demoBtn.addEventListener("click", loadDemoScene);
imageUpload.addEventListener("change", handleUpload);
window.addEventListener("resize", () => {
  resizeCanvas();
  if (currentSource) {
    drawDetections(currentDetections.length ? currentDetections : demoDetections);
  }
});

updateStatus("Idle", "idle");
setSourceLabel("No source loaded");
renderResults([], "Waiting");
resizeCanvas();
showActiveStage(null);