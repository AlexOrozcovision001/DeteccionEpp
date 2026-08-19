await window.__ORT_READY__;

const $ = id => document.getElementById(id);

const DEVICE = {
  isIOS:
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1),
  isAndroid: /Android/i.test(navigator.userAgent),
  isMobile: /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent),
  memoryGB: Number(navigator.deviceMemory || 0),
  cores: Number(navigator.hardwareConcurrency || 0)
};

DEVICE.lowMemory = DEVICE.isMobile && DEVICE.memoryGB > 0 && DEVICE.memoryGB <= 4;
DEVICE.lowCpu = DEVICE.isMobile && DEVICE.cores > 0 && DEVICE.cores <= 4;

function preferredExecutionProvider() {
  if (DEVICE.isIOS) return 'wasm';
  if ('gpu' in navigator) return 'webgpu';
  return 'wasm';
}

function devicePerformanceProfile() {
  const provider = preferredExecutionProvider();

  if (DEVICE.isIOS) {
    return {
      name: 'iPhone / iPad · modo seguro',
      provider: 'wasm', modelKey: 'yolo11s_dataset15', inputSize: 480,
      inferenceMs: 1200, cameraWidth: 640, cameraHeight: 480, cameraFps: 15,
      allowHeavyModel: false
    };
  }

  if (DEVICE.isAndroid) {
    if (provider === 'webgpu' && !DEVICE.lowMemory && !DEVICE.lowCpu) {
      return {
        name: 'Android · WebGPU',
        provider: 'webgpu', modelKey: 'yolo11s_dataset15', inputSize: 480,
        inferenceMs: 200, cameraWidth: 960, cameraHeight: 540, cameraFps: 30,
        allowHeavyModel: false
      };
    }
    return {
      name: 'Android · modo compatible',
      provider: 'wasm', modelKey: 'yolo11s_dataset15', inputSize: 480,
      inferenceMs: 700, cameraWidth: 640, cameraHeight: 480, cameraFps: 20,
      allowHeavyModel: false
    };
  }

  if (provider === 'webgpu') {
    return {
      name: 'Escritorio · WebGPU',
      provider: 'webgpu', modelKey: 'yolo11s_dataset15', inputSize: 480,
      inferenceMs: 0, cameraWidth: 1280, cameraHeight: 720, cameraFps: 30,
      allowHeavyModel: true
    };
  }

  return {
    name: 'Escritorio · WASM',
    provider: 'wasm', modelKey: 'yolo11s_dataset15', inputSize: 480,
    inferenceMs: 400, cameraWidth: 640, cameraHeight: 480, cameraFps: 24,
    allowHeavyModel: false
  };
}

function recommendedInputSize() {
  return devicePerformanceProfile().inputSize;
}
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const roiCanvas = $('roiCanvas');
const roiCtx = roiCanvas.getContext('2d');
const imageCanvas = $('imageCanvas');
const imageCtx = imageCanvas.getContext('2d');

let session = null;
let loadedModelKey = null;
let loadedModelSize = null;
let modelLoadInProgress = false;
let metadata = null;
let stream = null;
let running = false;
let frameNo = 0;
let lastFpsAt = performance.now();
let fpsFrames = 0;
let currentFps = 0;
let lastLatencyMs = 0;
let imageBitmap = null;
let sourceType = 'none';
let lastBoxes = [];
let lastPeople = [];
let tracks = [];
let nextTrackId = 1;
let roiDraft = null;
let roiPointerStart = null;
let iotTimer = null;
let reportStartedAt = new Date();
let lastReportSampleAt = 0;
let inferenceBusy = false;
let renderLoopActive = false;
let inferenceLoopActive = false;
let profileSwitching = false;
let currentProfile = 640;
let currentModelKey = 'yolo11s_dataset15';
let lastProfileSwitchAt = 0;
let latencyHistory = [];
let inferenceTimestamps = [];
let currentIps = 0;
let lastPipelineMs = 0;
let sampleSequence = 0;
let lastEvidenceAt = 0;
let lastEvidenceBlob = null;
let lastEvidenceFileName = '';
let evidenceBusy = false;
const evidenceEvents = [];
let availableCameras = [];
let selectedCameraId = localStorage.getItem('eppSelectedCameraId') || '';
let adminUnlocked = false;
let adminIdleTimer = null;
let deferredInstallPrompt = null;
const ADMIN_IDLE_MS = 5 * 60 * 1000;
const CONFIG_KEY = 'eppFactoryConfigV1';
const ADMIN_KEY = 'eppAdminCredentialV1';
let inferenceWorkspace = {
  size: 0,
  canvas: null,
  ctx: null,
  input: null,
  tensor: null
};

const MODEL_CATALOG = {
  yolo11s_previous: {
    label: 'YOLO11s · Modelo anterior',
    profiles: {
      640: 'models/yolo11s_previous/model_640.onnx',
      512: 'models/yolo11s_previous/model_512.onnx',
      480: 'models/yolo11s_previous/model_480.onnx'
    }
  },
  yolo11s_dataset15: {
    label: 'YOLO11s · Dataset 15',
    profiles: {
      640: 'models/yolo11s_dataset15/model_640.onnx',
      512: 'models/yolo11s_dataset15/model_512.onnx',
      480: 'models/yolo11s_dataset15/model_480.onnx'
    }
  },
  yolo11m_dataset15: {
    label: 'YOLO11m · Dataset 15',
    profiles: {
      640: 'models/yolo11m_dataset15/model_640.onnx',
      512: 'models/yolo11m_dataset15/model_512.onnx',
      480: 'models/yolo11m_dataset15/model_480.onnx'
    }
  }
};

function selectedModelKey() {
  return $('modelSelect')?.value || currentModelKey || 'yolo11s_dataset15';
}

function modelPath(modelKey, size) {
  const model = MODEL_CATALOG[modelKey];
  if (!model) throw new Error(`Modelo ${modelKey} no configurado.`);
  const path = model.profiles[size];
  if (!path) throw new Error(`Resolución ${size} no configurada para ${model.label}.`);
  return path;
}

function modelLabel(modelKey = currentModelKey) {
  return MODEL_CATALOG[modelKey]?.label || modelKey;
}

const sessionLog = [];
const personStats = new Map();
const personEvidence = new Map();

const state = {
  confidence: 0.35,
  iou: 0.45,
  skip: 1,
  inputSize: 640,
  performanceMode: 'manual',
  required: new Set(),
  visible: new Set(),
  personPersistence: 30,
  eppPersistence: 20,
  personHoldMs: 3200,
  eppHoldMs: 1800,
  smoothingPerson: 0.34,
  smoothingEpp: 0.30,
  roi: null, // normalizado: {x1,y1,x2,y2}
  roiSelecting: false,
  eppOnThreshold: 0.36,
  eppOffThreshold: 0.14,
  eppGain: 0.22,
  eppDecay: 0.035,
  reportSampleIntervalMs: 5000,
  inferenceMinIntervalMs: 0,
  evidenceEnabled: false,
  evidenceCooldownMs: 30000,
  evidenceQuality: 0.82,
  evidenceEndpoint: '',
  evidenceApiKey: ''
};

const colors = [
  '#18d4ff', // Person
  '#ffd600', // Hardhat
  '#22d47b', // Safety Vest
  '#b388ff', // Safety Glasses
  '#ff8a3d', // Gloves
  '#ff5d67', // Safety Boots
  '#b0bec5', // Regular Glasses
  '#bcaaa4'  // Regular Shoes
];

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

async function loadMetadata(url = 'models/epp-yolo11/metadata.json') {
  metadata = await fetch(url).then(r => {
    if (!r.ok) throw Error('No se pudo cargar metadata.json');
    return r.json();
  });
  applyMetadata();
}

function applyMetadata() {
  state.required = new Set(metadata.requiredDefaults || []);
  state.visible = new Set(metadata.classes.map((_, i) => i));
  renderClassChecks();
  if (!state.inputSize) state.inputSize = metadata.inputSize || 640;
  $('inputSize').textContent = `${state.inputSize}×${state.inputSize}`;
}

function renderClassChecks() {
  const req = $('requiredClasses');
  const vis = $('visibleClasses');
  req.innerHTML = '';
  vis.innerHTML = '';

  metadata.classes.forEach((name, i) => {
    if (i !== metadata.personClass) {
      const label = document.createElement('label');
      label.innerHTML = `<input type="checkbox" data-required="${i}" ${state.required.has(i) ? 'checked' : ''}> ${name}`;
      req.appendChild(label);
    }

    const visibleLabel = document.createElement('label');
    visibleLabel.innerHTML = `<input type="checkbox" data-visible="${i}" ${state.visible.has(i) ? 'checked' : ''}> ${name}`;
    vis.appendChild(visibleLabel);
  });

  req.onchange = e => {
    const i = Number(e.target.dataset.required);
    e.target.checked ? state.required.add(i) : state.required.delete(i);
    personEvidence.clear();
    render(lastBoxes);
  };

  vis.onchange = e => {
    const i = Number(e.target.dataset.visible);
    e.target.checked ? state.visible.add(i) : state.visible.delete(i);
    render(lastBoxes);
  };
}

async function fetchModelWithProgress(url, onProgress) {
  const response = await fetch(url, { cache: 'no-store' });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} al descargar ${url}`);
  }

  const total = Number(response.headers.get('content-length')) || 0;

  // En iOS y equipos con poca memoria evitamos conservar chunks + copia final.
  // arrayBuffer() mantiene un único bloque principal y reduce el pico de memoria.
  if (DEVICE.isMobile || !response.body || !total) {
    const buffer = await response.arrayBuffer();
    onProgress?.(1, buffer.byteLength, buffer.byteLength);
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    received += value.byteLength;
    onProgress?.(received / total, received, total);
  }

  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  onProgress?.(1, received, total || received);
  return merged;
}

async function createSession(
  model,
  inputSize = state.inputSize || metadata?.inputSize || 480,
  modelKey = currentModelKey
) {
  const modelDescription = `${modelLabel(modelKey)} · ${inputSize}×${inputSize}`;

  if (
    session &&
    loadedModelKey === modelKey &&
    loadedModelSize === inputSize
  ) {
    setStatus(`${modelDescription} ya está activo.`);
    return true;
  }

  if (modelLoadInProgress) {
    setStatus('Espere: ya se está cargando un modelo.');
    return false;
  }

  modelLoadInProgress = true;
  profileSwitching = true;

  let modelBytes = null;
  let newSession = null;

  try {
    setStatus(`Preparando ${modelDescription}...`);

    if (typeof model === 'string') {
      modelBytes = await fetchModelWithProgress(
        model,
        (progress, received, total) => {
          const percent = Math.max(0, Math.min(100, Math.round(progress * 100)));
          const receivedMB = (received / 1024 / 1024).toFixed(1);
          const totalMB = total ? (total / 1024 / 1024).toFixed(1) : '?';
          setStatus(
            `Descargando ${modelDescription}: <b>${percent}%</b> · ` +
            `${receivedMB} / ${totalMB} MB`
          );
        }
      );
    } else if (model instanceof Uint8Array) {
      modelBytes = model;
    } else if (model instanceof ArrayBuffer) {
      modelBytes = new Uint8Array(model);
    } else {
      throw new Error('Formato de modelo no válido.');
    }

    if (session) {
      setStatus(`Liberando modelo anterior antes de cargar ${modelDescription}...`);
      try {
        if (typeof session.release === 'function') {
          await session.release();
        }
      } catch (e) {
        console.warn('No se pudo liberar la sesión anterior:', e);
      }

      session = null;
      loadedModelKey = null;
      loadedModelSize = null;

      await new Promise(resolve => setTimeout(resolve, 250));
    }

    const provider = preferredExecutionProvider();

    setStatus(
      `Inicializando ${modelDescription} con ${provider.toUpperCase()}...`
    );

    newSession = await ort.InferenceSession.create(
      modelBytes,
      {
        executionProviders: [provider],
        graphOptimizationLevel: DEVICE.isIOS ? 'basic' : 'all'
      }
    );

    // Ya no necesitamos conservar los bytes descargados después de crear la sesión.
    // Es especialmente importante en móviles para reducir el pico de memoria.
    modelBytes = null;
    await new Promise(resolve => setTimeout(resolve, 0));

    if (!DEVICE.isIOS) {
      const zero = new Float32Array(3 * inputSize * inputSize);
      const inputName = newSession.inputNames[0];

      setStatus(`Calentando ${modelDescription}...`);

      const warmInput = new ort.Tensor(
        'float32', zero, [1, 3, inputSize, inputSize]
      );
      const warmOutputs = await newSession.run({ [inputName]: warmInput });
      for (const value of Object.values(warmOutputs)) {
        try { value.dispose?.(); } catch (_) {}
      }
      try { warmInput.dispose?.(); } catch (_) {}
    }

    session = newSession;
    newSession = null;

    loadedModelKey = modelKey;
    loadedModelSize = inputSize;
    state.inputSize = inputSize;
    currentProfile = inputSize;
    currentModelKey = modelKey;

    $('provider').textContent = provider.toUpperCase();
    $('engineBadge').textContent = `Motor: ${provider.toUpperCase()}`;
    $('engineBadge').className = 'badge pass';
    $('inputSize').textContent = `${inputSize}×${inputSize}`;
    $('activeModel').textContent = modelLabel(modelKey);
    $('resolutionSelect').value = String(inputSize);
    $('trackerStatus').textContent = 'ACTIVO';

    latencyHistory = [];
    inferenceTimestamps = [];
    currentIps = 0;
    $('ips').textContent = '0.0 IPS';

    resetTracking(`modelo ${modelDescription}`);

    setStatus(`${modelDescription} activo con ${provider.toUpperCase()}.`);
    return true;

  } catch (error) {
    if (newSession) {
      try {
        if (typeof newSession.release === 'function') {
          await newSession.release();
        }
      } catch (_) {}
    }

    session = null;
    loadedModelKey = null;
    loadedModelSize = null;

    $('engineBadge').textContent = 'Motor: sin iniciar';
    $('provider').textContent = '—';

    throw new Error(
      `No se pudo inicializar el modelo: ${error?.message || error}`
    );

  } finally {
    modelBytes = null;
    modelLoadInProgress = false;
    profileSwitching = false;
  }
}

async function warmup() {
  if (!session) return;
  const size = state.inputSize || metadata?.inputSize || 480;
  const zero = new Float32Array(3 * size * size);
  const inputName = session.inputNames[0];
  const inputTensor = new ort.Tensor('float32', zero, [1, 3, size, size]);
  const outputs = await session.run({ [inputName]: inputTensor });
  for (const value of Object.values(outputs)) {
    try { value.dispose?.(); } catch (_) {}
  }
  try { inputTensor.dispose?.(); } catch (_) {}
}

async function loadProfile(size, { fallback = false, reason = 'manual', modelKey = selectedModelKey() } = {}) {
  if (modelLoadInProgress) return false;

  const previousProfile = currentProfile;
  const previousModel = currentModelKey;

  try {
    const path = modelPath(modelKey, size);
    const ok = await createSession(path, size, modelKey);
    if (!ok) return false;

    lastProfileSwitchAt = performance.now();

    const label = reason === 'auto' ? 'Modo automático' : 'Modelo seleccionado';
    setStatus(`${label}: ${modelLabel(modelKey)} · ${size}×${size} activo.`);
    return true;

  } catch (error) {
    currentProfile = previousProfile;
    currentModelKey = previousModel;

    if ($('resolutionSelect')) $('resolutionSelect').value = String(previousProfile);
    if ($('modelSelect')) $('modelSelect').value = previousModel;

    throw error;
  }
}

async function loadDefault() {
  if (DEVICE.isIOS) {
    currentModelKey = 'yolo11s_dataset15';
    state.inputSize = 480;
    if ($('modelSelect')) $('modelSelect').value = 'yolo11s_dataset15';
    if ($('resolutionSelect')) $('resolutionSelect').value = '480';
  }

  if (!metadata) await loadMetadata();
  const requested = Number($('resolutionSelect').value || 512);
  await loadProfile(requested, { fallback: true, reason: 'manual', modelKey: selectedModelKey() });
}

function markInferenceComplete() {
  const now = performance.now();
  inferenceTimestamps.push(now);
  inferenceTimestamps = inferenceTimestamps.filter(t => now - t <= 1000);
  currentIps = inferenceTimestamps.length;
  $('ips').textContent = `${currentIps.toFixed(1)} IPS`;
  updateSystemDiagnostics();
}

function recordLatencyForAutoTune(ms) {
  latencyHistory.push(ms);
  if (latencyHistory.length > 10) latencyHistory.shift();
  if (state.performanceMode !== 'auto' || latencyHistory.length < 6 || profileSwitching) return;
  const now = performance.now();
  if (now - lastProfileSwitchAt < 8000) return;

  const avg = latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length;
  let target = currentProfile;
  if (avg > 180 && currentProfile === 640) target = 512;
  else if (avg > 150 && currentProfile === 512) target = 480;
  else if (avg < 65 && currentProfile === 480) target = 512;
  else if (avg < 55 && currentProfile === 512) target = 640;

  if (target !== currentProfile) {
    loadProfile(target, { fallback: false, reason: 'auto', modelKey: currentModelKey }).catch(error => {
      console.warn('Ajuste automático no disponible:', error);
      state.performanceMode = 'manual';
      $('performanceMode').value = 'manual';
      setStatus(`Modo automático detenido: falta el modelo ${target}×${target}. Genere los perfiles ONNX 512/480.`);
    });
  }
}

function setStatus(text) {
  $('status').innerHTML = text;
}

function resetTracking(reason = 'manual') {
  lastBoxes = [];
  lastPeople = [];
  tracks = [];
  personEvidence.clear();
  nextTrackId = 1;

  inferenceTimestamps = [];
  currentIps = 0;

  if ($('ips')) $('ips').textContent = '0.0 IPS';
  if ($('detectionCount')) $('detectionCount').textContent = '0';

  if ($('persons')) {
    $('persons').innerHTML =
      '<p class="empty">No se han detectado personas.</p>';
  }

  if ($('globalCompliance')) {
    $('globalCompliance').textContent = 'SIN EVALUAR';
    $('globalCompliance').className = 'compliance neutral';
  }

  console.log(`Tracker reiniciado: ${reason}`);
}

function ensureCanvasSize(width, height) {
  let changed = false;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    changed = true;
  }
  if (roiCanvas.width !== width || roiCanvas.height !== height) {
    roiCanvas.width = width;
    roiCanvas.height = height;
    changed = true;
  }
  if (changed) drawRoi(width, height);
}

function showVideoLayer() {
  video.classList.remove('hidden');
  imageCanvas.classList.add('hidden');
}

function showImageLayer() {
  video.classList.add('hidden');
  imageCanvas.classList.remove('hidden');
}

function drawImageLayer() {
  if (!imageBitmap) return;
  imageCanvas.width = imageBitmap.width;
  imageCanvas.height = imageBitmap.height;
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  imageCtx.drawImage(imageBitmap, 0, 0, imageCanvas.width, imageCanvas.height);
}

function syncStageAspect() {
  const [w, h] = sourceDimensions();
  if (w && h) $('stage').style.aspectRatio = `${w} / ${h}`;
}

async function startCamera() {
  stopSource(false);

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Este navegador no permite acceder a la cámara.');
  }

  const profile = devicePerformanceProfile();
  const cameraId = $('cameraSelect')?.value || selectedCameraId || '';
  const videoConstraints = {
    width: { ideal: profile.cameraWidth, max: profile.cameraWidth },
    height: { ideal: profile.cameraHeight, max: profile.cameraHeight },
    frameRate: { ideal: profile.cameraFps, max: profile.cameraFps }
  };

  if (cameraId) {
    videoConstraints.deviceId = { exact: cameraId };
  } else {
    videoConstraints.facingMode = { ideal: 'environment' };
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false
    });
  } catch (error) {
    // Si una cámara guardada dejó de existir, reintentamos con la cámara posterior.
    if (cameraId && (error.name === 'OverconstrainedError' || error.name === 'NotFoundError')) {
      selectedCameraId = '';
      localStorage.removeItem('eppSelectedCameraId');
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: profile.cameraWidth, max: profile.cameraWidth },
          height: { ideal: profile.cameraHeight, max: profile.cameraHeight },
          frameRate: { ideal: profile.cameraFps, max: profile.cameraFps }
        },
        audio: false
      });
    } else {
      throw error;
    }
  }

  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  video.src = '';
  video.controls = false;
  await video.play();

  sourceType = 'video';
  showVideoLayer();
  running = true;
  $('sourceLabel').textContent = 'Cámara';
  syncStageAspect();

  await refreshCameras(false);
  const activeTrack = stream.getVideoTracks()[0];
  const settings = activeTrack?.getSettings?.() || {};
  if (settings.deviceId) {
    selectedCameraId = settings.deviceId;
    localStorage.setItem('eppSelectedCameraId', selectedCameraId);
    if ($('cameraSelect')) $('cameraSelect').value = selectedCameraId;
  }

  startProcessingLoops();
  updateSystemDiagnostics();
}

function openVideo(file) {
  stopSource(false);
  resetTracking('nuevo video');
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.controls = true;
  video.loop = true;
  video.onloadeddata = () => {
    video.play();
    sourceType = 'video';
    showVideoLayer();
    running = true;
    $('sourceLabel').textContent = file.name;
    syncStageAspect();
    startProcessingLoops();
  };
}

async function openImage(file) {
  stopSource(false);
  resetTracking('nueva imagen');
  imageBitmap = await createImageBitmap(file);
  sourceType = 'image';
  showImageLayer();
  drawImageLayer();
  running = true;
  $('sourceLabel').textContent = file.name;
  syncStageAspect();
  await processCurrentFrame();
}

function stopSource(clearRoi = false) {
  running = false;
  resetTracking('fuente detenida');
  if (clearRoi) state.roi = null;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  if (video.src && video.src.startsWith('blob:')) URL.revokeObjectURL(video.src);
  video.pause();
  imageBitmap?.close?.();
  imageBitmap = null;
  sourceType = 'none';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  showVideoLayer();
  $('detectionCount').textContent = '0';
  $('persons').innerHTML = '<p class="empty">No se han detectado personas.</p>';
  $('globalCompliance').textContent = 'SIN EVALUAR';
  $('globalCompliance').className = 'compliance neutral';
  if (video.srcObject && !stream) {
    video.srcObject = null;
  }

  // En iOS se limpian buffers visuales para reducir memoria retenida.
  if (DEVICE.isIOS) {
    try {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
      imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    } catch (_) {}
  }
}

function sourceDimensions() {
  return sourceType === 'image'
    ? [imageBitmap?.width || 0, imageBitmap?.height || 0]
    : [video.videoWidth, video.videoHeight];
}

function sourceObject() {
  return sourceType === 'image' ? imageBitmap : video;
}

function drawSource(width, height) {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);
  const source = sourceObject();
  if (source) ctx.drawImage(source, 0, 0, width, height);
}

function roiPixels(width, height, roi = state.roi) {
  if (!roi) return { x: 0, y: 0, width, height };
  const x1 = Math.round(Math.min(roi.x1, roi.x2) * width);
  const y1 = Math.round(Math.min(roi.y1, roi.y2) * height);
  const x2 = Math.round(Math.max(roi.x1, roi.x2) * width);
  const y2 = Math.round(Math.max(roi.y1, roi.y2) * height);
  return {
    x: Math.max(0, x1),
    y: Math.max(0, y1),
    width: Math.max(1, Math.min(width, x2) - Math.max(0, x1)),
    height: Math.max(1, Math.min(height, y2) - Math.max(0, y1))
  };
}

function drawRoi(width, height) {
  if (!width || !height) return;
  roiCtx.clearRect(0, 0, width, height);
  const roi = roiDraft || state.roi;
  if (!roi) return;
  const r = roiPixels(width, height, roi);
  roiCtx.save();
  roiCtx.fillStyle = 'rgba(0,0,0,.38)';
  roiCtx.fillRect(0, 0, width, r.y);
  roiCtx.fillRect(0, r.y + r.height, width, height - r.y - r.height);
  roiCtx.fillRect(0, r.y, r.x, r.height);
  roiCtx.fillRect(r.x + r.width, r.y, width - r.x - r.width, r.height);
  roiCtx.setLineDash([12, 8]);
  roiCtx.lineWidth = Math.max(3, width / 480);
  roiCtx.strokeStyle = '#18d4ff';
  roiCtx.shadowColor = '#18d4ff';
  roiCtx.shadowBlur = 8;
  roiCtx.strokeRect(r.x, r.y, r.width, r.height);
  roiCtx.shadowBlur = 0;
  roiCtx.setLineDash([]);
  roiCtx.font = `700 ${Math.max(14, width / 65)}px system-ui`;
  const text = 'ROI · Zona de detección';
  const tw = roiCtx.measureText(text).width;
  const ty = Math.max(26, r.y + 26);
  roiCtx.fillStyle = 'rgba(7,17,31,.92)';
  roiCtx.fillRect(r.x + 4, ty - 21, tw + 16, 28);
  roiCtx.fillStyle = '#18d4ff';
  roiCtx.fillText(text, r.x + 12, ty);
  roiCtx.restore();
}
function getInferenceWorkspace(size) {
  if (
    inferenceWorkspace.size !== size ||
    !inferenceWorkspace.canvas ||
    !inferenceWorkspace.ctx ||
    !inferenceWorkspace.input ||
    !inferenceWorkspace.tensor
  ) {
    const workCanvas = new OffscreenCanvas(size, size);
    const workCtx = workCanvas.getContext('2d', {
      willReadFrequently: true,
      alpha: false,
      desynchronized: true
    });

    workCtx.imageSmoothingEnabled = true;
    workCtx.imageSmoothingQuality = 'low';

    const input = new Float32Array(3 * size * size);

    inferenceWorkspace = {
      size,
      canvas: workCanvas,
      ctx: workCtx,
      input,
      tensor: new ort.Tensor('float32', input, [1, 3, size, size])
    };
  }

  return inferenceWorkspace;
}

async function infer() {
  const totalStart = performance.now();
  const [sourceWidth, sourceHeight] = sourceDimensions();
  if (!sourceWidth || !sourceHeight) return [];

  ensureCanvasSize(sourceWidth, sourceHeight);

  const crop = roiPixels(sourceWidth, sourceHeight);
  const size = state.inputSize || metadata.inputSize || 480;
  const workspace = getInferenceWorkspace(size);
  const offCtx = workspace.ctx;
  const input = workspace.input;
  const inputTensor = workspace.tensor;

  offCtx.fillStyle = '#000';
  offCtx.fillRect(0, 0, size, size);

  const scale = Math.min(size / crop.width, size / crop.height);
  const resizedWidth = crop.width * scale;
  const resizedHeight = crop.height * scale;
  const padX = (size - resizedWidth) / 2;
  const padY = (size - resizedHeight) / 2;

  offCtx.drawImage(
    sourceObject(),
    crop.x, crop.y, crop.width, crop.height,
    padX, padY, resizedWidth, resizedHeight
  );

  const rgba = offCtx.getImageData(0, 0, size, size).data;
  const plane = size * size;

  // Reutiliza el mismo Float32Array para evitar asignaciones/GC en cada inferencia.
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    input[i] = rgba[p] * (1 / 255);
    input[plane + i] = rgba[p + 1] * (1 / 255);
    input[2 * plane + i] = rgba[p + 2] * (1 / 255);
  }

  const inferenceStart = performance.now();
  const inputName = session.inputNames[0];
  const output = await session.run({
    [inputName]: inputTensor
  });

  lastLatencyMs = performance.now() - inferenceStart;
  $('latency').textContent = `${lastLatencyMs.toFixed(1)} ms`;
  recordLatencyForAutoTune(lastLatencyMs);

  const tensor = output[session.outputNames[0]];
  const decoded = decode(
    tensor,
    scale,
    padX,
    padY,
    crop,
    sourceWidth,
    sourceHeight
  );

  // Libera la salida de ORT después de decodificarla para evitar acumulación
  // de memoria entre inferencias largas.
  try { tensor.dispose?.(); } catch (_) {}

  lastPipelineMs = performance.now() - totalStart;
  if ($('pipelineLatency')) {
    $('pipelineLatency').textContent = `${lastPipelineMs.toFixed(1)} ms`;
  }

  return decoded;
}

function decode(tensor, scale, padX, padY, crop, sourceWidth, sourceHeight) {
  const data = tensor.data;
  const dims = tensor.dims;
  let rows = [];

  if (dims.at(-1) === 6) {
    const count = dims.at(-2);
    for (let i = 0; i < count; i++) {
      const offset = i * 6;
      rows.push([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        data[offset + 4], Math.round(data[offset + 5])
      ]);
    }
  } else {
    let attributes;
    let count;
    let transposed = false;

    if (dims[1] === 4 + metadata.classes.length) {
      attributes = dims[1];
      count = dims[2];
      transposed = true;
    } else {
      count = dims[1];
      attributes = dims[2];
    }

    for (let i = 0; i < count; i++) {
      const get = attribute => transposed
        ? data[attribute * count + i]
        : data[i * attributes + attribute];

      let bestScore = -1;
      let classId = -1;
      for (let c = 0; c < metadata.classes.length; c++) {
        const score = get(4 + c);
        if (score > bestScore) {
          bestScore = score;
          classId = c;
        }
      }
      if (bestScore < state.confidence) continue;

      const centerX = get(0);
      const centerY = get(1);
      const width = get(2);
      const height = get(3);
      rows.push([
        centerX - width / 2,
        centerY - height / 2,
        centerX + width / 2,
        centerY + height / 2,
        bestScore,
        classId
      ]);
    }
  }

  rows = rows
    .filter(row => row[4] >= state.confidence)
    .map(row => [
      crop.x + (row[0] - padX) / scale,
      crop.y + (row[1] - padY) / scale,
      crop.x + (row[2] - padX) / scale,
      crop.y + (row[3] - padY) / scale,
      row[4], row[5]
    ])
    .map(row => [
      Math.max(crop.x, row[0]),
      Math.max(crop.y, row[1]),
      Math.min(crop.x + crop.width, row[2]),
      Math.min(crop.y + crop.height, row[3]),
      row[4], row[5]
    ])
    .filter(row => row[2] > row[0] && row[3] > row[1] && row[0] < sourceWidth && row[1] < sourceHeight);

  return nms(rows, state.iou);
}

function nms(boxes, iouThreshold) {
  const output = [];
  for (const classId of new Set(boxes.map(box => box[5]))) {
    let queue = boxes
      .filter(box => box[5] === classId)
      .sort((a, b) => b[4] - a[4]);

    while (queue.length) {
      const current = queue.shift();
      output.push(current);
      queue = queue.filter(box => iou(current, box) < iouThreshold);
    }
  }
  return output;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / (areaA + areaB - intersection + 1e-6);
}

function boxCenter(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

function normalizedCenterDistance(a, b) {
  const [ax, ay] = boxCenter(a);
  const [bx, by] = boxCenter(b);
  const scale = Math.max(1, Math.max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]));
  return Math.hypot(ax - bx, ay - by) / scale;
}

function predictTrackBox(track) {
  const box = track.box.slice();
  if (!track.velocity) return box;
  const [vx, vy] = track.velocity;
  box[0] += vx; box[2] += vx;
  box[1] += vy; box[3] += vy;
  return box;
}

function updateTracks(detections) {
  const unmatchedTracks = new Set(tracks.map((_, i) => i));
  const unmatchedDetections = new Set(detections.map((_, i) => i));
  const candidates = [];

  for (let ti = 0; ti < tracks.length; ti++) {
    const predicted = predictTrackBox(tracks[ti]);
    for (let di = 0; di < detections.length; di++) {
      if (tracks[ti].classId !== detections[di][5]) continue;
      const person = detections[di][5] === metadata.personClass;
      const overlap = iou(predicted, detections[di]);
      const distance = normalizedCenterDistance(predicted, detections[di]);
      const maxDistance = person ? 0.85 : 0.46;
      const minOverlap = person ? 0.02 : 0.10;
      if (overlap >= minOverlap || distance <= maxDistance) {
        const score = overlap * 1.8 + Math.max(0, maxDistance - distance);
        candidates.push({ ti, di, score });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  for (const { ti, di } of candidates) {
    if (!unmatchedTracks.has(ti) || !unmatchedDetections.has(di)) continue;
    const track = tracks[ti];
    const det = detections[di];
    const [oldCx, oldCy] = boxCenter(track.box);
    const [newCx, newCy] = boxCenter(det);
    const person = det[5] === metadata.personClass;
    const a = person ? state.smoothingPerson : state.smoothingEpp;

    for (let k = 0; k < 4; k++) track.box[k] = track.box[k] * (1 - a) + det[k] * a;
    track.box[4] = track.box[4] * 0.72 + det[4] * 0.28;
    track.box[5] = det[5];
    const measuredVx = newCx - oldCx;
    const measuredVy = newCy - oldCy;
    track.velocity[0] = track.velocity[0] * 0.65 + measuredVx * 0.35;
    track.velocity[1] = track.velocity[1] * 0.65 + measuredVy * 0.35;
    track.missed = 0;
    track.hits += 1;
    track.lastSeen = performance.now();
    unmatchedTracks.delete(ti);
    unmatchedDetections.delete(di);
  }

  for (const ti of unmatchedTracks) {
    const track = tracks[ti];
    track.missed += 1;
    if (track.classId === metadata.personClass && track.missed <= 8) {
      track.box = predictTrackBox(track);
      track.velocity[0] *= 0.86;
      track.velocity[1] *= 0.86;
    }
    track.box[4] *= 0.985;
  }

  for (const di of unmatchedDetections) {
    const det = detections[di];
    tracks.push({
      id: nextTrackId++,
      classId: det[5],
      box: det.slice(),
      missed: 0,
      hits: 1,
      velocity: [0, 0],
      lastSeen: performance.now()
    });
  }

  const trackNow = performance.now();
  tracks = tracks.filter(track => {
    const holdMs = track.classId === metadata.personClass ? state.personHoldMs : state.eppHoldMs;
    return (trackNow - track.lastSeen) <= holdMs;
  });

  return tracks.map(track => {
    const box = track.box.slice();
    box.trackId = track.id;
    box.missed = track.missed;
    box.hits = track.hits;
    return box;
  });
}

function rawAssociate(boxes) {
  const people = boxes
    .filter(box => box[5] === metadata.personClass)
    .map((box, index) => ({
      id: box.trackId || index + 1,
      box,
      rawEpp: new Map(),
      epp: new Map(),
      ok: false,
      missing: []
    }));

  for (const item of boxes.filter(box => box[5] !== metadata.personClass && (box.missed || 0) <= 4)) {
    const centerX = (item[0] + item[2]) / 2;
    const centerY = (item[1] + item[3]) / 2;
    let bestPerson = null;
    let bestScore = -Infinity;

    for (const person of people) {
      const [x1, y1, x2, y2] = person.box;
      const marginX = (x2 - x1) * 0.08;
      const marginY = (y2 - y1) * 0.04;
      if (centerX < x1 - marginX || centerX > x2 + marginX || centerY < y1 - marginY || centerY > y2 + marginY) continue;

      const relativeY = (centerY - y1) / Math.max(1, y2 - y1);
      const zone = metadata.zones?.[item[5]] || [0, 1];
      if (relativeY < zone[0] - 0.08 || relativeY > zone[1] + 0.08) continue;

      const horizontalDistance = Math.abs(centerX - (x1 + x2) / 2) / Math.max(1, x2 - x1);
      const score = item[4] - horizontalDistance * 0.20 + iou(person.box, item) * 0.10;
      if (score > bestScore) {
        bestPerson = person;
        bestScore = score;
      }
    }

    if (bestPerson) {
      const previous = bestPerson.rawEpp.get(item[5]);
      if (!previous || item[4] > previous[4]) bestPerson.rawEpp.set(item[5], item);
    }
  }

  return people;
}

function stabilizeCompliance(people) {
  const activeIds = new Set(people.map(p => p.id));

  for (const person of people) {
    let memory = personEvidence.get(person.id);
    if (!memory) {
      memory = { evidence: new Map(), present: new Map(), confidence: new Map(), lastSeen: performance.now() };
      personEvidence.set(person.id, memory);
    }
    memory.lastSeen = performance.now();

    for (const classId of state.required) {
      const rawDetected = person.rawEpp.has(classId);

      // La evidencia temporal se usa solo internamente para estabilizar.
      // La confianza mostrada al usuario proviene exclusivamente de YOLO.
      if (rawDetected) {
        const currentDetection = person.rawEpp.get(classId);
        if (currentDetection && Number.isFinite(currentDetection[4])) {
          memory.confidence.set(classId, currentDetection[4]);
        }
      }

      let value = memory.evidence.get(classId) || 0;
      value = rawDetected
        ? Math.min(1, value + state.eppGain)
        : Math.max(0, value - state.eppDecay);
      memory.evidence.set(classId, value);

      const wasPresent = memory.present.get(classId) || false;
      const isPresent = wasPresent ? value >= state.eppOffThreshold : value >= state.eppOnThreshold;
      memory.present.set(classId, isPresent);

      if (isPresent) {
        const current = person.rawEpp.get(classId);
        const synthetic = current || [person.box[0], person.box[1], person.box[0], person.box[1], value, classId];
        person.epp.set(classId, synthetic);
      }
    }

    person.missing = [...state.required].filter(classId => !memory.present.get(classId));
    person.ok = person.missing.length === 0;
    person.evidence = memory.evidence;
    person.yoloConfidence = memory.confidence;
  }

  const now = performance.now();
  for (const [id, memory] of personEvidence) {
    if (!activeIds.has(id) && now - memory.lastSeen > 5000) personEvidence.delete(id);
  }

  return people;
}

function associate(boxes) {
  return stabilizeCompliance(rawAssociate(boxes));
}

function drawDetectionBox(box, color, label, lineWidth = 3, alpha = 1) {
  const [x1, y1, x2, y2] = box;
  const width = x2 - x1;
  const height = y2 - y1;
  const fontSize = Math.max(13, canvas.width / 65);

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  if (!DEVICE.isMobile) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 4;
  }
  ctx.strokeRect(x1, y1, width, height);
  ctx.shadowBlur = 0;

  ctx.font = `700 ${fontSize}px system-ui`;
  const textWidth = ctx.measureText(label).width;
  const labelHeight = fontSize + 10;
  const labelY = y1 >= labelHeight ? y1 - labelHeight : y1;

  ctx.fillStyle = color;
  ctx.fillRect(x1, labelY, textWidth + 14, labelHeight);
  ctx.fillStyle = '#03111e';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x1 + 7, labelY + labelHeight / 2);
  ctx.restore();
}

function render(boxes = lastBoxes, people = lastPeople, updateUi = true) {
  const [width, height] = sourceDimensions();
  if (!width || !height) return;

  ensureCanvasSize(width, height);
  ctx.clearRect(0, 0, width, height);

  const personMap = new Map(people.map(person => [person.box.trackId || person.id, person]));

  for (const box of boxes) {
    if (box[5] === metadata.personClass || !state.visible.has(box[5])) continue;
    const className = metadata.classes[box[5]];
    const label = `${className} ${(box[4] * 100).toFixed(0)}%`;
    drawDetectionBox(box, colors[box[5] % colors.length], label, Math.max(2, width / 520), 1);
  }

  for (const box of boxes) {
    if (box[5] !== metadata.personClass || !state.visible.has(box[5])) continue;
    const person = personMap.get(box.trackId) || people.find(p => p.id === box.trackId);
    const color = person?.ok ? '#22d47b' : '#ff5d67';
    const status = person?.ok ? 'CUMPLE' : 'NO CUMPLE';
    const label = `Persona ${person?.id || box.trackId || ''} · ${status} · ${(box[4] * 100).toFixed(0)}%`;
    drawDetectionBox(box, color, label, Math.max(3, width / 420), 1);
  }

  if (updateUi) {
    renderCompliance(people);
    $('detectionCount').textContent = boxes.length;
  }
}

function renderCompliance(people) {
  const container = $('persons');

  if (!people.length) {
    container.innerHTML = '<p class="empty">No se han detectado personas.</p>';
    $('globalCompliance').textContent = 'SIN PERSONAS';
    $('globalCompliance').className = 'compliance neutral';
    updateMonitorView([]);
    return;
  }

  let allComply = true;
  container.innerHTML = '';

  for (const person of people) {
    allComply &&= person.ok;
    const card = document.createElement('article');
    card.className = 'person';
    card.innerHTML = `
      <div class="person-head">
        <b>Persona ${person.id}</b>
        <span class="badge ${person.ok ? 'pass' : 'fail'}">${person.ok ? 'CUMPLE' : 'NO CUMPLE'}</span>
      </div>
      ${[...state.required].map(classId => {
        const detected = person.epp.has(classId);

        // Mostrar la última confianza REAL de YOLO, no la evidencia temporal.
        const currentRaw = person.rawEpp?.get(classId);
        const confidence = currentRaw?.[4] ?? person.yoloConfidence?.get(classId);

        let statusText = '✗ No detectado';
        if (detected) {
          statusText = Number.isFinite(confidence)
            ? `✓ Detectado · ${(confidence * 100).toFixed(0)}%`
            : '✓ Detectado';
        }

        return `
          <div class="ppe-row">
            <span>${metadata.classes[classId]}</span>
            <b class="${detected ? 'yes' : 'no'}">${statusText}</b>
          </div>`;
      }).join('')}
    `;
    container.appendChild(card);
  }

  $('globalCompliance').textContent = allComply ? 'CUMPLE' : 'NO CUMPLE';
  $('globalCompliance').className = `compliance ${allComply ? 'pass' : 'fail'}`;
  updateMonitorView(people);
}



function base64FromBytes(bytes) {
  let binary = '';
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

function bytesFromBase64(value) {
  const binary = atob(value);
  return Uint8Array.from(binary, c => c.charCodeAt(0));
}

async function derivePinHash(pin, salt) {
  const keyMaterial = await crypto.subtle.importKey(
    'raw', new TextEncoder().encode(pin), 'PBKDF2', false, ['deriveBits']
  );
  const bits = await crypto.subtle.deriveBits({
    name: 'PBKDF2', salt, iterations: 120000, hash: 'SHA-256'
  }, keyMaterial, 256);
  return base64FromBytes(new Uint8Array(bits));
}

function hasAdminCredential() {
  return Boolean(localStorage.getItem(ADMIN_KEY));
}

async function createAdminCredential(pin) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const hash = await derivePinHash(pin, salt);
  localStorage.setItem(ADMIN_KEY, JSON.stringify({
    salt: base64FromBytes(salt), hash, createdAt: new Date().toISOString()
  }));
}

async function verifyAdminPin(pin) {
  try {
    const saved = JSON.parse(localStorage.getItem(ADMIN_KEY) || 'null');
    if (!saved?.salt || !saved?.hash) return false;
    const hash = await derivePinHash(pin, bytesFromBase64(saved.salt));
    return hash === saved.hash;
  } catch (_) {
    return false;
  }
}

function resetAdminIdleTimer() {
  clearTimeout(adminIdleTimer);
  if (!adminUnlocked) return;
  adminIdleTimer = setTimeout(() => {
    lockAdmin();
    setStatus('Administrador bloqueado automáticamente por inactividad.');
  }, ADMIN_IDLE_MS);
}

function showAuthOverlay(mode = 'unlock') {
  const overlay = $('authOverlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');
  $('firstRunPanel')?.classList.toggle('hidden', mode !== 'create');
  $('unlockPanel')?.classList.toggle('hidden', mode !== 'unlock');
  setTimeout(() => {
    (mode === 'create' ? $('setupPin') : $('unlockPin'))?.focus();
  }, 50);
}

function hideAuthOverlay() {
  $('authOverlay')?.classList.add('hidden');
  if ($('setupPin')) $('setupPin').value = '';
  if ($('setupPinConfirm')) $('setupPinConfirm').value = '';
  if ($('unlockPin')) $('unlockPin').value = '';
  if ($('authError')) $('authError').textContent = '';
}

function setAppMode(mode) {
  const admin = mode === 'admin';
  document.body.classList.toggle('admin-mode', admin);
  document.body.classList.toggle('monitor-mode', !admin);
  $('adminView')?.classList.toggle('hidden', !admin);
  $('monitorView')?.classList.toggle('hidden', admin);
  $('adminLiveControls')?.classList.toggle('hidden', !admin);
  $('monitorLiveControls')?.classList.toggle('hidden', admin);

  if (admin) {
    adminUnlocked = true;
    resetAdminIdleTimer();
    updateSystemDiagnostics();
  } else {
    adminUnlocked = false;
    clearTimeout(adminIdleTimer);
  }
}

function lockAdmin() {
  saveFactoryConfig();
  adminUnlocked = false;
  setAppMode('monitor');
}

function factoryConfigFromUi() {
  return {
    version: 1,
    company: $('reportProject')?.value.trim() || '',
    area: $('reportArea')?.value.trim() || '',
    owner: $('reportOwner')?.value.trim() || '',
    cameraId: $('cameraSelect')?.value || selectedCameraId || '',
    required: [...state.required],
    visible: [...state.visible],
    confidence: state.confidence,
    iou: state.iou,
    modelKey: selectedModelKey(),
    inputSize: Number($('resolutionSelect')?.value || state.inputSize || 480),
    performanceMode: $('performanceMode')?.value || state.performanceMode,
    inferenceMinIntervalMs: Number($('frameSkip')?.value || state.inferenceMinIntervalMs || 0),
    reportSampleIntervalSec: Number($('reportSampleInterval')?.value || 5),
    evidenceEnabled: Boolean($('evidenceEnabled')?.checked),
    evidenceCooldownSec: Number($('evidenceCooldown')?.value || 30),
    evidenceQuality: Number($('evidenceQuality')?.value || 0.82),
    evidenceEndpoint: $('evidenceEndpoint')?.value.trim() || '',
    thingSpeakInterval: Number($('thingSpeakInterval')?.value || 0),
    configuredAt: new Date().toISOString()
  };
}

function saveFactoryConfig() {
  try {
    const config = factoryConfigFromUi();
    localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
    selectedCameraId = config.cameraId || '';
    if (selectedCameraId) localStorage.setItem('eppSelectedCameraId', selectedCameraId);
    $('configState') && ($('configState').textContent = 'Configuración guardada');
    updateMonitorIdentity();
    return config;
  } catch (error) {
    console.warn('No se pudo guardar configuración:', error);
    return null;
  }
}

function loadFactoryConfig() {
  try {
    const config = JSON.parse(localStorage.getItem(CONFIG_KEY) || 'null');
    if (!config) return null;

    if ($('reportProject')) $('reportProject').value = config.company || '';
    if ($('reportArea')) $('reportArea').value = config.area || '';
    if ($('reportOwner')) $('reportOwner').value = config.owner || '';
    selectedCameraId = config.cameraId || selectedCameraId;

    if (Array.isArray(config.required)) state.required = new Set(config.required);
    if (Array.isArray(config.visible)) state.visible = new Set(config.visible);
    if (metadata) renderClassChecks();

    if (Number.isFinite(config.confidence)) {
      state.confidence = config.confidence;
      if ($('confRange')) $('confRange').value = String(config.confidence);
      if ($('confOut')) $('confOut').value = Number(config.confidence).toFixed(2);
    }
    if (Number.isFinite(config.iou)) {
      state.iou = config.iou;
      if ($('iouRange')) $('iouRange').value = String(config.iou);
      if ($('iouOut')) $('iouOut').value = Number(config.iou).toFixed(2);
    }

    if (config.modelKey && MODEL_CATALOG[config.modelKey]) {
      currentModelKey = config.modelKey;
      if ($('modelSelect')) $('modelSelect').value = config.modelKey;
    }
    if ([480, 512, 640].includes(Number(config.inputSize))) {
      state.inputSize = Number(config.inputSize);
      if ($('resolutionSelect')) $('resolutionSelect').value = String(config.inputSize);
    }

    state.performanceMode = config.performanceMode || state.performanceMode;
    if ($('performanceMode')) $('performanceMode').value = state.performanceMode;

    if (Number.isFinite(config.inferenceMinIntervalMs)) {
      state.inferenceMinIntervalMs = config.inferenceMinIntervalMs;
      if ($('frameSkip')) $('frameSkip').value = String(config.inferenceMinIntervalMs);
    }

    if (Number.isFinite(config.reportSampleIntervalSec)) {
      state.reportSampleIntervalMs = Math.max(1, config.reportSampleIntervalSec) * 1000;
      if ($('reportSampleInterval')) $('reportSampleInterval').value = String(config.reportSampleIntervalSec);
    }

    state.evidenceEnabled = Boolean(config.evidenceEnabled);
    if ($('evidenceEnabled')) $('evidenceEnabled').checked = state.evidenceEnabled;
    if (config.evidenceCooldownSec && $('evidenceCooldown')) $('evidenceCooldown').value = String(config.evidenceCooldownSec);
    if (config.evidenceQuality && $('evidenceQuality')) $('evidenceQuality').value = String(config.evidenceQuality);
    if ($('evidenceEndpoint')) $('evidenceEndpoint').value = config.evidenceEndpoint || '';
    if ($('thingSpeakInterval')) $('thingSpeakInterval').value = String(config.thingSpeakInterval || 0);

    $('configState') && ($('configState').textContent = 'Configuración instalada en este dispositivo');
    updateMonitorIdentity();
    return config;
  } catch (error) {
    console.warn('Configuración local inválida:', error);
    return null;
  }
}

function updateMonitorIdentity() {
  if ($('monitorCompany')) $('monitorCompany').textContent = $('reportProject')?.value.trim() || 'Sistema de seguridad EPP';
  if ($('monitorArea')) $('monitorArea').textContent = $('reportArea')?.value.trim() || 'Área sin configurar';
}

function updateMonitorView(people) {
  const status = $('monitorCompliance');
  const detail = $('monitorDetail');
  const chips = $('monitorRequired');
  if (!status || !detail || !chips || !metadata) return;

  chips.innerHTML = [...state.required].map(classId => {
    const everyoneHas = people.length > 0 && people.every(p => p.epp.has(classId));
    return `<span class="monitor-chip ${everyoneHas ? 'ok' : 'missing'}">${metadata.classes[classId]} ${everyoneHas ? '✓' : '—'}</span>`;
  }).join('');

  if (!people.length) {
    status.textContent = 'EN ESPERA';
    status.className = 'monitor-compliance neutral';
    detail.textContent = 'Ubíquese dentro del área de detección.';
    return;
  }

  const nonCompliant = people.filter(p => !p.ok);
  if (!nonCompliant.length) {
    status.textContent = '✓ CUMPLE';
    status.className = 'monitor-compliance pass';
    detail.textContent = `${people.length} persona${people.length === 1 ? '' : 's'} · EPP completo`;
    return;
  }

  const missing = [...new Set(nonCompliant.flatMap(p => p.missing.map(id => metadata.classes[id])))];
  status.textContent = '⚠ NO CUMPLE';
  status.className = 'monitor-compliance fail';
  detail.textContent = `Falta: ${missing.join(', ') || 'EPP requerido'}`;
}

async function refreshCameras(requestPermission = false) {
  if (!navigator.mediaDevices?.enumerateDevices) return [];

  if (requestPermission && !stream) {
    try {
      const temp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      temp.getTracks().forEach(track => track.stop());
    } catch (error) {
      setStatus(`No se pudo obtener permiso de cámara: ${error.message}`);
    }
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  availableCameras = devices.filter(device => device.kind === 'videoinput');
  const select = $('cameraSelect');
  if (select) {
    const current = selectedCameraId || select.value;
    select.innerHTML = '';
    availableCameras.forEach((camera, index) => {
      const option = document.createElement('option');
      option.value = camera.deviceId;
      option.textContent = camera.label || `Cámara ${index + 1}`;
      select.appendChild(option);
    });
    if (current && availableCameras.some(c => c.deviceId === current)) {
      select.value = current;
    } else if (availableCameras.length) {
      const back = availableCameras.find(c => /back|rear|environment|trasera/i.test(c.label));
      select.value = back?.deviceId || availableCameras[0].deviceId;
    }
    selectedCameraId = select.value || '';
  }
  if ($('cameraCount')) $('cameraCount').textContent = `${availableCameras.length} cámara${availableCameras.length === 1 ? '' : 's'}`;
  return availableCameras;
}

async function switchToNextCamera() {
  await refreshCameras(true);
  if (availableCameras.length < 2) {
    setStatus('Solo se detectó una cámara disponible.');
    return;
  }
  const currentIndex = Math.max(0, availableCameras.findIndex(c => c.deviceId === selectedCameraId));
  const next = availableCameras[(currentIndex + 1) % availableCameras.length];
  selectedCameraId = next.deviceId;
  localStorage.setItem('eppSelectedCameraId', selectedCameraId);
  if ($('cameraSelect')) $('cameraSelect').value = selectedCameraId;
  if (running && sourceType === 'video') await startCamera();
  setStatus(`Cámara activa: ${next.label || 'Cámara seleccionada'}.`);
}

function updateSystemDiagnostics() {
  const profile = devicePerformanceProfile();
  const parts = [
    profile.name,
    `Backend ${profile.provider.toUpperCase()}`,
    DEVICE.cores ? `${DEVICE.cores} hilos lógicos` : 'CPU no reportada',
    DEVICE.memoryGB ? `~${DEVICE.memoryGB} GB RAM reportada` : 'RAM no reportada'
  ];
  if ($('deviceProfile')) $('deviceProfile').textContent = parts.join(' · ');
  if ($('diagFps')) $('diagFps').textContent = `${currentFps.toFixed(1)} FPS`;
  if ($('diagIps')) $('diagIps').textContent = `${currentIps.toFixed(1)} IPS`;
  if ($('diagLatency')) $('diagLatency').textContent = lastLatencyMs ? `${lastLatencyMs.toFixed(1)} ms` : '—';
  if ($('diagPipeline')) $('diagPipeline').textContent = lastPipelineMs ? `${lastPipelineMs.toFixed(1)} ms` : '—';
  if ($('diagProvider')) $('diagProvider').textContent = $('provider')?.textContent || profile.provider.toUpperCase();
  if ($('diagModel')) $('diagModel').textContent = session ? `${modelLabel(currentModelKey)} · ${state.inputSize}×${state.inputSize}` : 'Sin cargar';
}

async function startInstalledMonitoring() {
  saveFactoryConfig();
  setAppMode('monitor');
  try {
    if (!session) await loadDefault();
    await startCamera();
    setStatus('Monitoreo activo.');
  } catch (error) {
    setStatus(`No se pudo iniciar el monitoreo: ${error.message}`);
  }
}

function safeFilePart(value) {
  return String(value || '')
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 50);
}

function evidenceMetadata(people, now, fileName) {
  return {
    timestamp: now.toISOString(),
    localTime: now.toLocaleString(),
    fileName,
    project: $('reportProject')?.value.trim() || '',
    area: $('reportArea')?.value.trim() || '',
    owner: $('reportOwner')?.value.trim() || '',
    source: $('sourceLabel')?.textContent || '',
    model: modelLabel(currentModelKey),
    inputSize: state.inputSize,
    provider: $('provider')?.textContent || '',
    fps: Number(currentFps.toFixed(1)),
    ips: Number(currentIps.toFixed(1)),
    latencyMs: Number(lastLatencyMs.toFixed(1)),
    people: people.map(person => ({
      id: person.id,
      compliance: person.ok ? 'CUMPLE' : 'NO CUMPLE',
      missing: person.missing.map(id => metadata.classes[id]),
      detected: [...state.required]
        .filter(id => person.epp.has(id))
        .map(id => metadata.classes[id])
    })),
    roi: state.roi ? { ...state.roi } : null
  };
}

async function buildEvidenceBlob(people, now, fileName) {
  const [sourceWidth, sourceHeight] = sourceDimensions();
  const source = sourceObject();
  if (!source || !sourceWidth || !sourceHeight) {
    throw new Error('No hay una fuente activa para capturar.');
  }

  const maxWidth = DEVICE.isIOS ? 960 : 1280;
  const scale = Math.min(1, maxWidth / sourceWidth);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));

  const evidenceCanvas = document.createElement('canvas');
  evidenceCanvas.width = width;
  evidenceCanvas.height = height;
  const ectx = evidenceCanvas.getContext('2d', { alpha: false });

  ectx.drawImage(source, 0, 0, width, height);

  if (canvas.width && canvas.height) {
    ectx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, width, height);
  }

  if (roiCanvas.width && roiCanvas.height) {
    ectx.drawImage(roiCanvas, 0, 0, roiCanvas.width, roiCanvas.height, 0, 0, width, height);
  }

  const nonCompliant = people.filter(person => !person.ok);
  const missing = [...new Set(
    nonCompliant.flatMap(person => person.missing.map(id => metadata.classes[id]))
  )];

  const fontSize = Math.max(16, Math.round(width / 55));
  const lineHeight = fontSize + 8;
  const lines = [
    `NO CUMPLE · ${now.toLocaleString()}`,
    `Personas: ${nonCompliant.map(p => p.id).join(', ') || '—'}`,
    `Faltante: ${missing.join(', ') || '—'}`
  ];

  const boxHeight = lineHeight * lines.length + 16;
  ectx.fillStyle = 'rgba(120, 0, 15, 0.86)';
  ectx.fillRect(0, height - boxHeight, width, boxHeight);
  ectx.fillStyle = '#ffffff';
  ectx.font = `700 ${fontSize}px system-ui`;
  ectx.textBaseline = 'top';

  lines.forEach((line, index) => {
    ectx.fillText(line, 12, height - boxHeight + 8 + index * lineHeight);
  });

  const quality = Math.min(0.95, Math.max(0.55, Number(state.evidenceQuality) || 0.82));

  const blob = await new Promise((resolve, reject) => {
    evidenceCanvas.toBlob(
      value => value ? resolve(value) : reject(new Error('No se pudo generar la fotografía JPEG.')),
      'image/jpeg',
      quality
    );
  });

  return blob;
}

async function uploadEvidence(blob, meta) {
  const endpoint = String(state.evidenceEndpoint || '').trim();
  if (!endpoint) return { uploaded: false, response: null };

  const form = new FormData();
  form.append('image', blob, meta.fileName);
  form.append('metadata', JSON.stringify(meta));
  form.append('timestamp', meta.timestamp);
  form.append('project', meta.project);
  form.append('area', meta.area);
  form.append('owner', meta.owner);
  if (state.evidenceApiKey) {
    form.append('api_key', state.evidenceApiKey);
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    body: form,
    cache: 'no-store'
  });

  if (!response.ok) {
    throw new Error(`Servidor respondió HTTP ${response.status}`);
  }

  let responseData = null;
  try {
    responseData = await response.json();
  } catch (_) {
    responseData = await response.text().catch(() => '');
  }

  return { uploaded: true, response: responseData };
}

async function captureEvidence(people = lastPeople, { force = false } = {}) {
  if (evidenceBusy) return false;

  const nonCompliant = people.filter(person => !person.ok);
  if (!nonCompliant.length) {
    $('evidenceStatus').textContent = 'No hay una persona en estado NO CUMPLE.';
    return false;
  }

  const now = new Date();
  const nowMs = now.getTime();

  if (!force) {
    if (!state.evidenceEnabled) return false;
    const cooldown = Math.max(5000, Number(state.evidenceCooldownMs) || 30000);
    if (nowMs - lastEvidenceAt < cooldown) return false;
  }

  evidenceBusy = true;
  lastEvidenceAt = nowMs;

  const project = safeFilePart($('reportProject')?.value) || 'proyecto';
  const timePart = now.toISOString().replaceAll(':', '-').replaceAll('.', '-');
  const fileName = `epp_no_cumple_${project}_${timePart}.jpg`;

  $('evidenceStatus').textContent = 'Generando evidencia...';

  try {
    const blob = await buildEvidenceBlob(nonCompliant, now, fileName);
    lastEvidenceBlob = blob;
    lastEvidenceFileName = fileName;
    $('downloadEvidenceBtn').disabled = false;

    const meta = evidenceMetadata(nonCompliant, now, fileName);
    let uploadInfo = { uploaded: false, response: null };

    if (String(state.evidenceEndpoint || '').trim()) {
      $('evidenceStatus').textContent = 'Enviando fotografía al servidor...';
      uploadInfo = await uploadEvidence(blob, meta);
    }

    evidenceEvents.push({
      ...meta,
      uploaded: uploadInfo.uploaded,
      serverResponse: uploadInfo.response || null,
      bytes: blob.size
    });

    localStorage.setItem(
      'eppEvidenceEvents',
      JSON.stringify(evidenceEvents.slice(-250))
    );

    $('evidenceStatus').textContent = uploadInfo.uploaded
      ? `Evidencia enviada · ${now.toLocaleTimeString()}`
      : `Evidencia capturada · ${now.toLocaleTimeString()} · lista para descargar`;

    return true;
  } catch (error) {
    $('evidenceStatus').textContent = `Error de evidencia: ${error.message}`;
    console.error(error);
    return false;
  } finally {
    evidenceBusy = false;
  }
}

function downloadLastEvidence() {
  if (!lastEvidenceBlob || !lastEvidenceFileName) {
    $('evidenceStatus').textContent = 'Aún no existe una evidencia para descargar.';
    return;
  }

  const url = URL.createObjectURL(lastEvidenceBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = lastEvidenceFileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function updateAcquisitionClock() {
  const label = $('sampleCountdownLabel');
  if (!label) return;

  const intervalMs = Math.max(1000, Number(state.reportSampleIntervalMs) || 5000);
  const intervalSec = Math.round(intervalMs / 1000);

  if (!running) {
    label.textContent = `Tiempo configurado: cada ${intervalSec} s · en espera de una fuente.`;
    return;
  }

  if (!lastReportSampleAt) {
    label.textContent = `Tiempo configurado: cada ${intervalSec} s · próxima toma: inmediata.`;
    return;
  }

  const remainingMs = Math.max(0, intervalMs - (Date.now() - lastReportSampleAt));
  const remainingSec = remainingMs / 1000;
  label.textContent =
    `Tiempo configurado: cada ${intervalSec} s · próxima toma en ${remainingSec.toFixed(1)} s.`;
}

setInterval(updateAcquisitionClock, 250);

function persistReportSession() {
  try {
    const compact = sessionLog.slice(-2000);
    localStorage.setItem('eppSessionLog', JSON.stringify(compact));
    localStorage.setItem('eppReportStartedAt', reportStartedAt.toISOString());
    localStorage.setItem('eppEvidenceEvents', JSON.stringify(evidenceEvents.slice(-250)));
  } catch (error) {
    console.warn('No se pudo persistir el reporte en el navegador:', error);
  }
}

function updateReportStats(people) {
  const nowMs = Date.now();
  const intervalMs = Math.max(1000, Number(state.reportSampleIntervalMs) || 5000);
  if (nowMs - lastReportSampleAt < intervalMs) return;

  lastReportSampleAt = nowMs;
  const now = new Date();
  const intervalSec = Math.round(intervalMs / 1000);
  const sampleId = ++sampleSequence;

  for (const person of people) {
    let stat = personStats.get(person.id);
    if (!stat) {
      stat = {
        id: person.id,
        firstSeen: now.toISOString(),
        lastSeen: now.toISOString(),
        samples: 0,
        compliantSamples: 0,
        nonCompliantSamples: 0
      };
      personStats.set(person.id, stat);
    }

    stat.lastSeen = now.toISOString();
    stat.samples += 1;
    if (person.ok) stat.compliantSamples += 1;
    else stat.nonCompliantSamples += 1;

    sessionLog.push({
      sampleId,
      timestamp: now.toISOString(),
      localTime: now.toLocaleString(),
      sampleIntervalSec: intervalSec,
      source: $('sourceLabel').textContent,
      personId: person.id,
      compliance: person.ok ? 'CUMPLE' : 'NO CUMPLE',
      detected: [...state.required]
        .filter(id => person.epp.has(id))
        .map(id => metadata.classes[id]),
      missing: person.missing.map(id => metadata.classes[id]),
      fps: Number(currentFps.toFixed(1)),
      ips: Number(currentIps.toFixed(1)),
      latencyMs: Number(lastLatencyMs.toFixed(1)),
      pipelineMs: Number(lastPipelineMs.toFixed(1)),
      detections: lastBoxes.length,
      provider: $('provider')?.textContent || '',
      model: modelLabel(currentModelKey),
      inputSize: state.inputSize,
      roi: state.roi ? { ...state.roi } : null
    });
  }

  if (!people.length) {
    sessionLog.push({
      sampleId,
      timestamp: now.toISOString(),
      localTime: now.toLocaleString(),
      sampleIntervalSec: intervalSec,
      source: $('sourceLabel').textContent,
      personId: '',
      compliance: 'SIN PERSONAS',
      detected: [],
      missing: [],
      fps: Number(currentFps.toFixed(1)),
      ips: Number(currentIps.toFixed(1)),
      latencyMs: Number(lastLatencyMs.toFixed(1)),
      pipelineMs: Number(lastPipelineMs.toFixed(1)),
      detections: lastBoxes.length,
      provider: $('provider')?.textContent || '',
      model: modelLabel(currentModelKey),
      inputSize: state.inputSize,
      roi: state.roi ? { ...state.roi } : null
    });
  }

  $('reportEventsBadge').textContent =
    `Registro · cada ${intervalSec} s · ${sessionLog.length}`;

  $('reportEventsBadge').className = 'badge pass';
  $('lastRecordLabel').textContent =
    `Última toma: ${now.toLocaleTimeString()} · muestra #${sampleId}.`;

  persistReportSession();

  // La fotografía corre fuera del ciclo de inferencia para no frenar la IA.
  if (state.evidenceEnabled && people.some(person => !person.ok)) {
    setTimeout(() => {
      captureEvidence(people, { force: false }).catch(console.error);
    }, 0);
  }
}

async function processCurrentFrame() {
  if (!session) {
    const [width, height] = sourceDimensions();
    if (width && height) {
      ensureCanvasSize(width, height);
      ctx.clearRect(0, 0, width, height);
      drawRoi(width, height);
    }
    return;
  }

  const detections = await infer();
  lastBoxes = updateTracks(detections);
  lastPeople = associate(lastBoxes);
  render(lastBoxes, lastPeople);
  updateReportStats(lastPeople);
}

function startProcessingLoops() {
  if (!renderLoopActive) {
    renderLoopActive = true;
    requestAnimationFrame(renderLoop);
  }
  if (!inferenceLoopActive) {
    inferenceLoopActive = true;
    inferenceLoop();
  }
}

function renderLoop() {
  if (!running) {
    renderLoopActive = false;
    return;
  }

  // El overlay se redibuja de forma independiente de ONNX. Aunque una inferencia
  // tarde cientos de milisegundos, la última caja estable nunca se borra.
  render(lastBoxes, lastPeople, false);

  fpsFrames++;
  const now = performance.now();
  if (now - lastFpsAt > 1000) {
    currentFps = fpsFrames * 1000 / (now - lastFpsAt);
    $('fpsLabel').textContent = `${currentFps.toFixed(1)} FPS`;
    const ipsNow = performance.now();
    inferenceTimestamps = inferenceTimestamps.filter(t => ipsNow - t <= 1000);
    currentIps = inferenceTimestamps.length;
    $('ips').textContent = `${currentIps.toFixed(1)} IPS`;
    fpsFrames = 0;
    lastFpsAt = now;
  }

  requestAnimationFrame(renderLoop);
}

async function inferenceLoop() {
  if (!running) {
    inferenceLoopActive = false;
    return;
  }

  if (!session || inferenceBusy) {
    setTimeout(inferenceLoop, 16);
    return;
  }

  const loopStart = performance.now();
  inferenceBusy = true;
  try {
    const detections = await infer();
    lastBoxes = updateTracks(detections);
    lastPeople = associate(lastBoxes);
    // Actualizar paneles una sola vez por inferencia; el dibujo visual corre aparte.
    render(lastBoxes, lastPeople, true);
    updateReportStats(lastPeople);
    markInferenceComplete();
  } catch (error) {
    console.error(error);
    setStatus(`Error de inferencia: ${error.message}`);
    running = false;
    inferenceLoopActive = false;
    inferenceBusy = false;
    return;
  }
  inferenceBusy = false;

  // Ritmo de IA: 0 = máxima velocidad. En otros modos se completa
  // el periodo objetivo contando también el tiempo real de inferencia.
  const elapsed = performance.now() - loopStart;
  const targetPeriod = Math.max(0, Number(state.inferenceMinIntervalMs) || 0);
  const delay = Math.max(0, targetPeriod - elapsed);
  setTimeout(inferenceLoop, delay);
}

function canvasPoint(event) {
  const rect = roiCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * roiCanvas.width / rect.width;
  const y = (event.clientY - rect.top) * roiCanvas.height / rect.height;
  return {
    x: Math.max(0, Math.min(roiCanvas.width, x)),
    y: Math.max(0, Math.min(roiCanvas.height, y))
  };
}

function updateRoiUi() {
  const active = Boolean(state.roi);
  $('roiBtn').classList.toggle('roi-active', active);
  $('roiBtn').textContent = active ? 'Cambiar ROI' : 'Seleccionar ROI';
  $('roiStatus').textContent = active ? 'ROI: ACTIVO' : 'ROI: COMPLETO';
  $('roiStatus').classList.toggle('active', active);
  roiCanvas.classList.toggle('roi-selecting', state.roiSelecting);
}

function beginRoiSelection() {
  const [w, h] = sourceDimensions();
  if (!w || !h) {
    setStatus('Abra primero una cámara, video o imagen y después seleccione el ROI.');
    return;
  }
  state.roiSelecting = true;
  roiDraft = null;
  $('roiHint').classList.remove('hidden');
  updateRoiUi();
  drawRoi(...sourceDimensions());
  setStatus('Modo ROI activo: arrastre sobre la imagen para definir la zona de detección.');
}

function finishRoiSelection() {
  state.roiSelecting = false;
  $('roiHint').classList.add('hidden');
  updateRoiUi();
  drawRoi(...sourceDimensions());
}

roiCanvas.addEventListener('pointerdown', event => {
  if (!state.roiSelecting) return;
  event.preventDefault();
  roiCanvas.setPointerCapture(event.pointerId);
  const p = canvasPoint(event);
  roiPointerStart = p;
  roiDraft = { x1: p.x / roiCanvas.width, y1: p.y / roiCanvas.height, x2: p.x / roiCanvas.width, y2: p.y / roiCanvas.height };
});

roiCanvas.addEventListener('pointermove', event => {
  if (!state.roiSelecting || !roiPointerStart) return;
  const p = canvasPoint(event);
  roiDraft = {
    x1: roiPointerStart.x / roiCanvas.width,
    y1: roiPointerStart.y / roiCanvas.height,
    x2: p.x / roiCanvas.width,
    y2: p.y / roiCanvas.height
  };
  drawRoi(...sourceDimensions());
});

roiCanvas.addEventListener('pointerup', event => {
  if (!state.roiSelecting || !roiPointerStart) return;
  const p = canvasPoint(event);
  const draft = {
    x1: roiPointerStart.x / roiCanvas.width,
    y1: roiPointerStart.y / roiCanvas.height,
    x2: p.x / roiCanvas.width,
    y2: p.y / roiCanvas.height
  };
  roiPointerStart = null;
  const width = Math.abs(draft.x2 - draft.x1);
  const height = Math.abs(draft.y2 - draft.y1);
  if (width >= 0.04 && height >= 0.04) {
    state.roi = {
      x1: Math.min(draft.x1, draft.x2), y1: Math.min(draft.y1, draft.y2),
      x2: Math.max(draft.x1, draft.x2), y2: Math.max(draft.y1, draft.y2)
    };
    resetTracking('ROI modificado');
    updateRoiUi();
    setStatus('ROI aplicado. La zona azul permanecerá marcada y solo esa región será analizada.');
  } else {
    setStatus('ROI demasiado pequeño. Intente nuevamente.');
  }
  roiDraft = null;
  finishRoiSelection();
  drawRoi(...sourceDimensions());
  if (sourceType === 'image') processCurrentFrame();
});

function clearRoi() {
  state.roi = null;
  roiDraft = null;
  finishRoiSelection();
  drawRoi(...sourceDimensions());
  resetTracking('ROI eliminado');
  updateRoiUi();
  drawRoi(...sourceDimensions());
  render(lastBoxes, lastPeople);
  setStatus('ROI eliminado. Se procesará la imagen completa.');
  if (sourceType === 'image') processCurrentFrame();
}

function reportMeta() {
  return {
    project: $('reportProject').value.trim(),
    area: $('reportArea').value.trim(),
    owner: $('reportOwner').value.trim(),
    startedAt: reportStartedAt.toISOString(),
    generatedAt: new Date().toISOString(),
    model: metadata?.name || $('modelSelect').selectedOptions[0]?.textContent || '',
    requiredEpp: metadata ? [...state.required].map(id => metadata.classes[id]) : [],
    confidence: state.confidence,
    iouNms: state.iou,
    inputSize: state.inputSize,
    performanceMode: state.performanceMode,
    reportSampleIntervalSec: Math.round(state.reportSampleIntervalMs / 1000),
    inferenceMinIntervalMs: state.inferenceMinIntervalMs,
    evidenceEnabled: state.evidenceEnabled,
    evidenceCooldownSec: Math.round(state.evidenceCooldownMs / 1000),
    roi: state.roi
  };
}

function reportSummary() {
  const currentPersons = lastPeople.length;
  const compliantNow = lastPeople.filter(p => p.ok).length;
  const nonCompliantNow = currentPersons - compliantNow;
  return {
    uniquePersons: personStats.size,
    currentPersons,
    compliantNow,
    nonCompliantNow,
    records: sessionLog.length,
    fps: Number(currentFps.toFixed(1)),
    ips: Number(currentIps.toFixed(1)),
    latencyMs: Number(lastLatencyMs.toFixed(1)),
    pipelineMs: Number(lastPipelineMs.toFixed(1)),
    detections: lastBoxes.length
  };
}

function reportPayload() {
  return {
    meta: reportMeta(),
    summary: reportSummary(),
    persons: [...personStats.values()].map(stat => ({
      ...stat,
      complianceRate: stat.samples ? Number((100 * stat.compliantSamples / stat.samples).toFixed(1)) : 0
    })),
    records: sessionLog,
    evidence: evidenceEvents
  };
}

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
  setStatus(`Archivo generado: ${filename}`);
}

function csvCell(value) {
  const s = Array.isArray(value) ? value.join(' | ') : String(value ?? '');
  return `"${s.replaceAll('"', '""')}"`;
}

function downloadCsv() {
  const header = [
    'sampleId','timestamp','localTime','sampleIntervalSec',
    'source','personId','compliance','detected','missing',
    'fps','ips','latencyMs','pipelineMs','detections',
    'provider','model','inputSize'
  ];
  const lines = [header.join(',')];
  for (const row of sessionLog) {
    lines.push(header.map(key => csvCell(row[key])).join(','));
  }
  downloadBlob(`reporte_epp_${new Date().toISOString().replaceAll(':','-')}.csv`, '\ufeff' + lines.join('\n'), 'text/csv;charset=utf-8');
}

function downloadJson() {
  downloadBlob(`reporte_epp_${new Date().toISOString().replaceAll(':','-')}.json`, JSON.stringify(reportPayload(), null, 2), 'application/json');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;' }[c]));
}

function printReport() {
  const payload = reportPayload();
  const report = $('printableReport');
  report.innerHTML = `
    <h1>Reporte de Monitoreo de EPP</h1>
    <p><b>Proyecto:</b> ${escapeHtml(payload.meta.project || '—')} &nbsp; <b>Área:</b> ${escapeHtml(payload.meta.area || '—')}</p>
    <p><b>Responsable:</b> ${escapeHtml(payload.meta.owner || '—')} &nbsp; <b>Generado:</b> ${escapeHtml(new Date(payload.meta.generatedAt).toLocaleString())}</p>
    <p><b>Modelo:</b> ${escapeHtml(payload.meta.model)} &nbsp; <b>EPP obligatorios:</b> ${escapeHtml(payload.meta.requiredEpp.join(', '))}</p>
    <div class="summary-box">
      <div><b>Personas únicas</b><br>${payload.summary.uniquePersons}</div>
      <div><b>Cumplen ahora</b><br>${payload.summary.compliantNow}</div>
      <div><b>No cumplen ahora</b><br>${payload.summary.nonCompliantNow}</div>
      <div><b>Registros</b><br>${payload.summary.records}</div>
    </div>
    <h2>Resumen por persona</h2>
    <table><thead><tr><th>ID</th><th>Primera detección</th><th>Última detección</th><th>Muestras</th><th>% cumplimiento</th></tr></thead><tbody>
      ${payload.persons.map(p => `<tr><td>${p.id}</td><td>${escapeHtml(new Date(p.firstSeen).toLocaleString())}</td><td>${escapeHtml(new Date(p.lastSeen).toLocaleString())}</td><td>${p.samples}</td><td>${p.complianceRate}%</td></tr>`).join('')}
    </tbody></table>
    <h2>Registros</h2>
    <table><thead><tr><th>Fecha/hora</th><th>Persona</th><th>Estado</th><th>Detectado</th><th>Faltante</th></tr></thead><tbody>
      ${payload.records.slice(-250).map(r => `<tr><td>${escapeHtml(new Date(r.timestamp).toLocaleString())}</td><td>${r.personId}</td><td>${r.compliance}</td><td>${escapeHtml(r.detected.join(', '))}</td><td>${escapeHtml(r.missing.join(', '))}</td></tr>`).join('')}
    </tbody></table>
    <p class="small">El reporte refleja detecciones automáticas del modelo de visión artificial y debe interpretarse junto con la verificación de campo correspondiente.</p>`;
  window.print();
}

function clearReport() {
  sessionLog.length = 0;
  personStats.clear();
  reportStartedAt = new Date();
  lastReportSampleAt = 0;
  sampleSequence = 0;
  evidenceEvents.length = 0;
  lastEvidenceAt = 0;
  lastEvidenceBlob = null;
  lastEvidenceFileName = '';
  $('reportEventsBadge').textContent =
    `Registro · cada ${Math.round(state.reportSampleIntervalMs / 1000)} s · 0`;
  $('lastRecordLabel').textContent = 'Aún sin muestras.';
  $('downloadEvidenceBtn').disabled = true;
  $('evidenceStatus').textContent = state.evidenceEnabled
    ? 'Captura automática activa.'
    : 'Captura desactivada.';
  localStorage.removeItem('eppSessionLog');
  localStorage.removeItem('eppReportStartedAt');
  localStorage.removeItem('eppEvidenceEvents');
  setStatus('Registros del reporte reiniciados.');
}

async function sendThingSpeak() {
  const key = $('thingSpeakKey').value.trim();
  if (!key) {
    $('iotStatus').textContent = 'Ingrese la Write API Key.';
    return;
  }
  localStorage.setItem('eppThingSpeakKey', key);
  const s = reportSummary();
  const compliancePct = s.currentPersons
    ? Number((100 * s.compliantNow / s.currentPersons).toFixed(1))
    : 0;

  const params = new URLSearchParams({
    api_key: key,
    field1: String(s.currentPersons),
    field2: String(s.compliantNow),
    field3: String(s.nonCompliantNow),
    field4: String(s.latencyMs),
    field5: String(s.ips),
    field6: String(s.detections),
    field7: String(s.fps),
    field8: String(compliancePct)
  });
  $('iotStatus').textContent = 'Enviando...';
  try {
    await fetch(`https://api.thingspeak.com/update?${params.toString()}`, { mode: 'no-cors', cache: 'no-store' });
    $('iotStatus').textContent = `Envío realizado ${new Date().toLocaleTimeString()}.`;
  } catch (error) {
    $('iotStatus').textContent = `No se pudo enviar: ${error.message}`;
  }
}

function configureThingSpeakTimer() {
  if (iotTimer) clearInterval(iotTimer);
  iotTimer = null;
  const seconds = Number($('thingSpeakInterval').value);
  if (seconds > 0) iotTimer = setInterval(sendThingSpeak, seconds * 1000);
}


$('adminAccessBtn').onclick = () => {
  if (!hasAdminCredential()) showAuthOverlay('create');
  else showAuthOverlay('unlock');
};

$('cancelUnlockBtn').onclick = hideAuthOverlay;

$('createAdminBtn').onclick = async () => {
  const pin = $('setupPin').value.trim();
  const confirm = $('setupPinConfirm').value.trim();
  if (!/^\d{4,8}$/.test(pin)) {
    $('authError').textContent = 'Use un PIN numérico de 4 a 8 dígitos.';
    return;
  }
  if (pin !== confirm) {
    $('authError').textContent = 'Los PIN no coinciden.';
    return;
  }
  await createAdminCredential(pin);
  hideAuthOverlay();
  setAppMode('admin');
  setStatus('Administrador creado. Configure la estación y guarde los cambios.');
};

$('unlockAdminBtn').onclick = async () => {
  const ok = await verifyAdminPin($('unlockPin').value.trim());
  if (!ok) {
    $('authError').textContent = 'PIN incorrecto.';
    return;
  }
  hideAuthOverlay();
  setAppMode('admin');
  setStatus('Modo administrador desbloqueado.');
};

$('lockAdminBtn').onclick = lockAdmin;
$('saveConfigBtn').onclick = () => {
  saveFactoryConfig();
  setStatus('Configuración guardada en este dispositivo.');
};
$('startInstalledBtn').onclick = startInstalledMonitoring;
$('monitorStartBtn').onclick = startInstalledMonitoring;
$('monitorStopBtn').onclick = () => stopSource(false);
$('switchCameraBtn').onclick = () => switchToNextCamera().catch(error => setStatus(error.message));
$('refreshCamerasBtn').onclick = () => refreshCameras(true).catch(error => setStatus(error.message));
$('testCameraBtn').onclick = () => startCamera().catch(error => setStatus(error.message));
$('cameraSelect').onchange = event => {
  selectedCameraId = event.target.value;
  localStorage.setItem('eppSelectedCameraId', selectedCameraId);
  saveFactoryConfig();
};

['click','keydown','pointerdown','touchstart'].forEach(name => {
  document.addEventListener(name, () => {
    if (adminUnlocked) resetAdminIdleTimer();
  }, { passive: true });
});

navigator.mediaDevices?.addEventListener?.('devicechange', () => {
  refreshCameras(false).catch(console.warn);
});

window.addEventListener('beforeinstallprompt', event => {
  event.preventDefault();
  deferredInstallPrompt = event;
  $('installAppBtn')?.classList.remove('hidden');
});

$('installAppBtn').onclick = async () => {
  if (deferredInstallPrompt) {
    deferredInstallPrompt.prompt();
    await deferredInstallPrompt.userChoice.catch(() => null);
    deferredInstallPrompt = null;
    $('installAppBtn').classList.add('hidden');
  } else {
    setStatus(DEVICE.isIOS
      ? 'En iPhone: Compartir → Agregar a pantalla de inicio.'
      : 'Use la opción Instalar aplicación del menú del navegador.');
  }
};

$('loadDefaultBtn').onclick = () => loadDefault().catch(error => {
  setStatus(`No se pudo cargar el modelo predeterminado: ${error.message}`);
});

$('modelInput').onchange = event => {
  const file = event.target.files[0];
  if (file) createSession(file.arrayBuffer(), Number($('resolutionSelect').value || 640)).catch(error => setStatus(error.message));
};

$('metadataInput').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  metadata = JSON.parse(await file.text());
  applyMetadata();
  resetTracking();
};

$('cameraBtn').onclick = () => startCamera().catch(error => setStatus(error.message));
$('videoInput').onchange = event => event.target.files[0] && openVideo(event.target.files[0]);
$('imageInput').onchange = event => event.target.files[0] && openImage(event.target.files[0]);
$('stopBtn').onclick = () => stopSource(false);
$('roiBtn').onclick = beginRoiSelection;
$('clearRoiBtn').onclick = clearRoi;

$('confRange').oninput = event => {
  state.confidence = Number(event.target.value);
  $('confOut').value = state.confidence.toFixed(2);
};

$('iouRange').oninput = event => {
  state.iou = Number(event.target.value);
  $('iouOut').value = state.iou.toFixed(2);
};

$('frameSkip').onchange = event => {
  state.inferenceMinIntervalMs = Number(event.target.value) || 0;
  localStorage.setItem(
    'eppInferenceMinIntervalMs',
    String(state.inferenceMinIntervalMs)
  );

  const label = event.target.selectedOptions[0]?.textContent || '';
  setStatus(`Ritmo de IA: ${label}.`);
};

$('modelSelect').onchange = event => {
  if (DEVICE.isIOS && event.target.value === 'yolo11m_dataset15') {
    event.target.value = 'yolo11s_dataset15';
    currentModelKey = 'yolo11s_dataset15';
    setStatus('En iPhone/iPad se utiliza YOLO11s 480×480 para evitar errores de memoria.');
    return;
  }
  const modelKey = event.target.value;
  if (!session) {
    currentModelKey = modelKey;
    $('activeModel').textContent = `${modelLabel(modelKey)} (pendiente)`;
    setStatus(`${modelLabel(modelKey)} seleccionado. Pulse Cargar modelo.`);
    return;
  }
  const size = Number($('resolutionSelect').value || currentProfile || 512);
  loadProfile(size, { fallback: true, reason: 'manual', modelKey }).catch(error =>
    setStatus(`No se pudo cargar ${modelLabel(modelKey)}: ${error.message}`)
  );
};

$('resolutionSelect').onchange = event => {
  const size = Number(event.target.value);
  if (!session) {
    state.inputSize = size;
    $('inputSize').textContent = `${size}×${size} (pendiente)`;
    return;
  }
  loadProfile(size, { fallback: true, reason: 'manual' }).catch(error => setStatus(`No se pudo cargar ${size}×${size}: ${error.message}`));
};

$('performanceMode').onchange = event => {
  state.performanceMode = event.target.value;
  latencyHistory = [];
  setStatus(state.performanceMode === 'auto'
    ? 'Modo rendimiento automático activo. Ajustará la resolución según la latencia si los perfiles ONNX están disponibles.'
    : 'Modo rendimiento manual activo.');
};

$('downloadCsvBtn').onclick = downloadCsv;
$('downloadJsonBtn').onclick = downloadJson;
$('printReportBtn').onclick = printReport;
$('clearReportBtn').onclick = clearReport;
$('reportSampleInterval').onchange = event => {
  const seconds = Math.max(1, Number(event.target.value) || 5);
  state.reportSampleIntervalMs = seconds * 1000;
  lastReportSampleAt = 0;

  localStorage.setItem('eppReportSampleIntervalSec', String(seconds));

  $('reportEventsBadge').textContent =
    `Registro · cada ${seconds} s · ${sessionLog.length}`;

  updateAcquisitionClock();
  setStatus(`Tiempo entre tomas de datos: ${seconds} s.`);
};


$('evidenceEnabled').onchange = event => {
  state.evidenceEnabled = Boolean(event.target.checked);
  localStorage.setItem('eppEvidenceEnabled', state.evidenceEnabled ? '1' : '0');
  $('evidenceStatus').textContent = state.evidenceEnabled
    ? 'Captura automática activa para NO CUMPLE.'
    : 'Captura desactivada.';
};

$('evidenceCooldown').onchange = event => {
  const seconds = Math.max(5, Number(event.target.value) || 30);
  state.evidenceCooldownMs = seconds * 1000;
  localStorage.setItem('eppEvidenceCooldownSec', String(seconds));
};

$('evidenceQuality').onchange = event => {
  state.evidenceQuality = Math.min(0.95, Math.max(0.55, Number(event.target.value) || 0.82));
  localStorage.setItem('eppEvidenceQuality', String(state.evidenceQuality));
};

$('evidenceEndpoint').onchange = event => {
  state.evidenceEndpoint = event.target.value.trim();
  localStorage.setItem('eppEvidenceEndpoint', state.evidenceEndpoint);
};

$('evidenceApiKey').onchange = event => {
  state.evidenceApiKey = event.target.value.trim();
  localStorage.setItem('eppEvidenceApiKey', state.evidenceApiKey);
};

$('captureEvidenceBtn').onclick = () => {
  captureEvidence(lastPeople, { force: true }).catch(console.error);
};

$('downloadEvidenceBtn').onclick = downloadLastEvidence;

$('sendThingSpeakBtn').onclick = sendThingSpeak;
$('thingSpeakInterval').onchange = () => {
  localStorage.setItem(
    'eppThingSpeakIntervalSec',
    $('thingSpeakInterval').value
  );
  configureThingSpeakTimer();
};
$('thingSpeakKey').value = localStorage.getItem('eppThingSpeakKey') || '';


function restoreReportSession() {
  try {
    const saved = JSON.parse(localStorage.getItem('eppSessionLog') || '[]');
    if (Array.isArray(saved) && saved.length) {
      sessionLog.push(...saved);
      const intervalSec = Math.round(state.reportSampleIntervalMs / 1000);
      $('reportEventsBadge').textContent =
        `Registro · cada ${intervalSec} s · ${sessionLog.length}`;
      $('lastRecordLabel').textContent = `Se restauraron ${sessionLog.length} registros del navegador.`;
    }
    const started = localStorage.getItem('eppReportStartedAt');
    if (started) reportStartedAt = new Date(started);

    sampleSequence = sessionLog.reduce(
      (maxValue, row) => Math.max(maxValue, Number(row.sampleId) || 0),
      0
    );

    const savedEvidence = JSON.parse(localStorage.getItem('eppEvidenceEvents') || '[]');
    if (Array.isArray(savedEvidence) && savedEvidence.length) {
      evidenceEvents.push(...savedEvidence.slice(-250));
    }
  } catch (error) {
    console.warn('No se pudo restaurar el reporte:', error);
  }
}

$('modelSelect').value = 'yolo11s_dataset15';
currentModelKey = 'yolo11s_dataset15';
$('activeModel').textContent = 'YOLO11s · Dataset 15 (pendiente)';
$('resolutionSelect').value = '480';
$('performanceMode').value = state.performanceMode;
state.inputSize = 480;
$('inputSize').textContent = '480×480 (pendiente)';

const savedReportIntervalSec = Math.max(
  1,
  Number(localStorage.getItem('eppReportSampleIntervalSec') || 5)
);
state.reportSampleIntervalMs = savedReportIntervalSec * 1000;
if ($('reportSampleInterval')) {
  $('reportSampleInterval').value = String(savedReportIntervalSec);
  $('reportEventsBadge').textContent =
    `Registro · cada ${savedReportIntervalSec} s · 0`;
}

const savedInferenceMinIntervalMs = Math.max(
  0,
  Number(localStorage.getItem('eppInferenceMinIntervalMs') || 0)
);
state.inferenceMinIntervalMs = savedInferenceMinIntervalMs;
if ($('frameSkip')) {
  $('frameSkip').value = String(savedInferenceMinIntervalMs);
}


const savedEvidenceEnabled = localStorage.getItem('eppEvidenceEnabled') === '1';
const savedEvidenceCooldownSec = Math.max(
  5,
  Number(localStorage.getItem('eppEvidenceCooldownSec') || 30)
);
const savedEvidenceQuality = Math.min(
  0.95,
  Math.max(0.55, Number(localStorage.getItem('eppEvidenceQuality') || 0.82))
);

state.evidenceEnabled = savedEvidenceEnabled;
state.evidenceCooldownMs = savedEvidenceCooldownSec * 1000;
state.evidenceQuality = savedEvidenceQuality;
state.evidenceEndpoint = localStorage.getItem('eppEvidenceEndpoint') || '';
state.evidenceApiKey = localStorage.getItem('eppEvidenceApiKey') || '';

$('evidenceEnabled').checked = state.evidenceEnabled;
$('evidenceCooldown').value = String(savedEvidenceCooldownSec);
$('evidenceQuality').value = String(savedEvidenceQuality);
$('evidenceEndpoint').value = state.evidenceEndpoint;
$('evidenceApiKey').value = state.evidenceApiKey;
$('evidenceStatus').textContent = state.evidenceEnabled
  ? 'Captura automática activa para NO CUMPLE.'
  : 'Captura desactivada.';

const savedThingSpeakInterval = localStorage.getItem(
  'eppThingSpeakIntervalSec'
);
if (savedThingSpeakInterval && $('thingSpeakInterval')) {
  $('thingSpeakInterval').value = savedThingSpeakInterval;
}
configureThingSpeakTimer();


function applyDeviceProfile() {
  const profile = devicePerformanceProfile();

  // Solo imponemos límites conservadores en móviles. En escritorio se conserva
  // la configuración guardada por el administrador.
  if (DEVICE.isMobile) {
    state.inputSize = profile.inputSize;
    state.performanceMode = 'manual';
    state.inferenceMinIntervalMs = Math.max(state.inferenceMinIntervalMs, profile.inferenceMs);

    if ($('resolutionSelect')) $('resolutionSelect').value = String(profile.inputSize);
    if ($('performanceMode')) $('performanceMode').value = 'manual';
    if ($('frameSkip')) {
      const values = [...$('frameSkip').options].map(o => Number(o.value));
      const closest = values.reduce((a, b) => Math.abs(b - profile.inferenceMs) < Math.abs(a - profile.inferenceMs) ? b : a, values[0]);
      $('frameSkip').value = String(closest);
      state.inferenceMinIntervalMs = closest;
    }

    for (const option of $('modelSelect')?.options || []) {
      if (option.value === 'yolo11m_dataset15') option.disabled = true;
    }

    if (DEVICE.isIOS) {
      state.evidenceEnabled = false;
      if ($('evidenceEnabled')) $('evidenceEnabled').checked = false;
    }
  }

  if ($('provider')) $('provider').textContent = profile.provider.toUpperCase();
  if ($('engineBadge')) {
    $('engineBadge').textContent = `Motor: ${profile.provider.toUpperCase()} · ${DEVICE.isIOS ? 'iOS' : DEVICE.isAndroid ? 'Android' : 'PC'}`;
  }

  updateSystemDiagnostics();
}

restoreReportSession();
applyDeviceProfile();
updateAcquisitionClock();
updateRoiUi();

loadMetadata()
  .then(() => {
    loadFactoryConfig();
    applyDeviceProfile();
    updateMonitorIdentity();
    updateMonitorView([]);
    refreshCameras(false).catch(console.warn);

    if (!hasAdminCredential()) {
      setAppMode('monitor');
      showAuthOverlay('create');
    } else {
      setAppMode('monitor');
    }
  })
  .catch(error => setStatus(error.message));

window.addEventListener('beforeunload', () => {
  saveFactoryConfig();
  if (session && typeof session.release === 'function') {
    try {
      session.release();
    } catch (_) {}
  }
});
