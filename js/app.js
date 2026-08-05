const $ = id => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

let session = null;
let metadata = null;
let stream = null;
let running = false;
let frameNo = 0;
let lastFpsAt = performance.now();
let fpsFrames = 0;
let imageBitmap = null;
let sourceType = 'none';
let lastBoxes = [];
let tracks = [];
let nextTrackId = 1;

const state = {
  confidence: 0.35,
  iou: 0.45,
  skip: 2,
  required: new Set(),
  visible: new Set(),
  persistence: 6,
  smoothing: 0.42
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

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

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
  $('inputSize').textContent = `${metadata.inputSize}×${metadata.inputSize}`;
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
    render(lastBoxes);
  };

  vis.onchange = e => {
    const i = Number(e.target.dataset.visible);
    e.target.checked ? state.visible.add(i) : state.visible.delete(i);
    render(lastBoxes);
  };
}

async function createSession(model) {
  setStatus('Cargando modelo...');
  const providers = [];
  if ('gpu' in navigator) providers.push('webgpu');
  providers.push('wasm');

  let lastError;
  for (const provider of providers) {
    try {
      session = await ort.InferenceSession.create(model, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all'
      });
      $('provider').textContent = provider.toUpperCase();
      $('engineBadge').textContent = `Motor: ${provider.toUpperCase()}`;
      $('engineBadge').className = 'badge pass';
      setStatus('Modelo listo. Seleccione cámara, video o imagen.');
      await warmup();
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function warmup() {
  const size = metadata.inputSize || 640;
  const zero = new Float32Array(3 * size * size);
  const inputName = session.inputNames[0];
  await session.run({
    [inputName]: new ort.Tensor('float32', zero, [1, 3, size, size])
  });
}

async function loadDefault() {
  if (!metadata) await loadMetadata();
  await createSession($('modelSelect').value);
}

function setStatus(text) {
  $('status').textContent = text;
}

async function startCamera() {
  stopSource();
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: 'environment'
    },
    audio: false
  });
  video.srcObject = stream;
  video.controls = false;
  await video.play();
  sourceType = 'video';
  running = true;
  $('sourceLabel').textContent = 'Cámara';
  loop();
}

function openVideo(file) {
  stopSource();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.controls = true;
  video.loop = true;
  video.onloadeddata = () => {
    video.play();
    sourceType = 'video';
    running = true;
    $('sourceLabel').textContent = file.name;
    loop();
  };
}

async function openImage(file) {
  stopSource();
  imageBitmap = await createImageBitmap(file);
  sourceType = 'image';
  running = true;
  $('sourceLabel').textContent = file.name;
  await processCurrentFrame();
}

function stopSource() {
  running = false;
  lastBoxes = [];
  tracks = [];
  nextTrackId = 1;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.pause();
  imageBitmap?.close?.();
  imageBitmap = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function sourceDimensions() {
  return sourceType === 'image'
    ? [imageBitmap?.width || 0, imageBitmap?.height || 0]
    : [video.videoWidth, video.videoHeight];
}

function drawSource(width, height) {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);
  const source = sourceType === 'image' ? imageBitmap : video;
  if (source) ctx.drawImage(source, 0, 0, width, height);
}

async function infer() {
  const [sourceWidth, sourceHeight] = sourceDimensions();
  if (!sourceWidth || !sourceHeight) return [];

  canvas.width = sourceWidth;
  canvas.height = sourceHeight;

  const size = metadata.inputSize || 640;
  const offscreen = new OffscreenCanvas(size, size);
  const offCtx = offscreen.getContext('2d', { willReadFrequently: true });
  offCtx.fillStyle = '#000';
  offCtx.fillRect(0, 0, size, size);

  const scale = Math.min(size / sourceWidth, size / sourceHeight);
  const resizedWidth = sourceWidth * scale;
  const resizedHeight = sourceHeight * scale;
  const padX = (size - resizedWidth) / 2;
  const padY = (size - resizedHeight) / 2;

  offCtx.drawImage(
    sourceType === 'image' ? imageBitmap : video,
    padX,
    padY,
    resizedWidth,
    resizedHeight
  );

  const rgba = offCtx.getImageData(0, 0, size, size).data;
  const input = new Float32Array(3 * size * size);
  const plane = size * size;

  for (let i = 0; i < plane; i++) {
    input[i] = rgba[i * 4] / 255;
    input[plane + i] = rgba[i * 4 + 1] / 255;
    input[2 * plane + i] = rgba[i * 4 + 2] / 255;
  }

  const start = performance.now();
  const inputName = session.inputNames[0];
  const output = await session.run({
    [inputName]: new ort.Tensor('float32', input, [1, 3, size, size])
  });
  $('latency').textContent = `${(performance.now() - start).toFixed(1)} ms`;

  const tensor = output[session.outputNames[0]];
  return decode(tensor, scale, padX, padY, sourceWidth, sourceHeight);
}

function decode(tensor, scale, padX, padY, sourceWidth, sourceHeight) {
  const data = tensor.data;
  const dims = tensor.dims;
  let rows = [];

  if (dims.at(-1) === 6) {
    const count = dims.at(-2);
    for (let i = 0; i < count; i++) {
      const offset = i * 6;
      rows.push([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        Math.round(data[offset + 5])
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
      (row[0] - padX) / scale,
      (row[1] - padY) / scale,
      (row[2] - padX) / scale,
      (row[3] - padY) / scale,
      row[4],
      row[5]
    ])
    .map(row => [
      Math.max(0, row[0]),
      Math.max(0, row[1]),
      Math.min(sourceWidth, row[2]),
      Math.min(sourceHeight, row[3]),
      row[4],
      row[5]
    ]);

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
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
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

function updateTracks(detections) {
  const unmatchedTracks = new Set(tracks.map((_, i) => i));
  const unmatchedDetections = new Set(detections.map((_, i) => i));
  const candidates = [];

  for (let ti = 0; ti < tracks.length; ti++) {
    for (let di = 0; di < detections.length; di++) {
      if (tracks[ti].classId !== detections[di][5]) continue;
      const overlap = iou(tracks[ti].box, detections[di]);
      const distance = normalizedCenterDistance(tracks[ti].box, detections[di]);
      if (overlap >= 0.12 || distance <= 0.40) {
        const score = overlap + Math.max(0, 0.40 - distance);
        candidates.push({ ti, di, score });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  for (const { ti, di } of candidates) {
    if (!unmatchedTracks.has(ti) || !unmatchedDetections.has(di)) continue;
    const track = tracks[ti];
    const det = detections[di];
    const a = state.smoothing;

    for (let k = 0; k < 4; k++) track.box[k] = track.box[k] * (1 - a) + det[k] * a;
    track.box[4] = track.box[4] * 0.65 + det[4] * 0.35;
    track.box[5] = det[5];
    track.missed = 0;
    track.hits += 1;
    unmatchedTracks.delete(ti);
    unmatchedDetections.delete(di);
  }

  for (const ti of unmatchedTracks) tracks[ti].missed += 1;

  for (const di of unmatchedDetections) {
    const det = detections[di];
    tracks.push({
      id: nextTrackId++,
      classId: det[5],
      box: det.slice(),
      missed: 0,
      hits: 1
    });
  }

  tracks = tracks.filter(track => track.missed <= state.persistence);

  return tracks.map(track => {
    const box = track.box.slice();
    box.trackId = track.id;
    box.missed = track.missed;
    return box;
  });
}

function associate(boxes) {
  const people = boxes
    .filter(box => box[5] === metadata.personClass)
    .map((box, index) => ({
      id: box.trackId || index + 1,
      box,
      epp: new Map(),
      ok: false,
      missing: []
    }));

  for (const item of boxes.filter(box => box[5] !== metadata.personClass)) {
    const centerX = (item[0] + item[2]) / 2;
    const centerY = (item[1] + item[3]) / 2;
    let bestPerson = null;
    let bestScore = 0;

    for (const person of people) {
      const [x1, y1, x2, y2] = person.box;
      if (centerX < x1 || centerX > x2 || centerY < y1 || centerY > y2) continue;

      const relativeY = (centerY - y1) / Math.max(1, y2 - y1);
      const zone = metadata.zones?.[item[5]] || [0, 1];
      if (relativeY < zone[0] || relativeY > zone[1]) continue;

      const horizontalDistance = Math.abs(centerX - (x1 + x2) / 2) / Math.max(1, x2 - x1);
      const score = item[4] * (1 - horizontalDistance);
      if (score > bestScore) {
        bestPerson = person;
        bestScore = score;
      }
    }

    if (bestPerson) {
      const previous = bestPerson.epp.get(item[5]);
      if (!previous || item[4] > previous[4]) bestPerson.epp.set(item[5], item);
    }
  }

  for (const person of people) {
    person.missing = [...state.required].filter(classId => !person.epp.has(classId));
    person.ok = person.missing.length === 0;
  }

  return people;
}

function drawDetectionBox(box, color, label, lineWidth = 3) {
  const [x1, y1, x2, y2] = box;
  const width = x2 - x1;
  const height = y2 - y1;
  const fontSize = Math.max(13, canvas.width / 65);

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 5;
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

function render(boxes = lastBoxes) {
  const [width, height] = sourceDimensions();
  if (!width || !height) return;

  canvas.width = width;
  canvas.height = height;
  drawSource(width, height);

  const people = associate(boxes);
  const personMap = new Map(people.map(person => [person.box, person]));

  // Primero se dibujan los EPP para que la caja de la persona quede visible encima.
  for (const box of boxes) {
    if (box[5] === metadata.personClass || !state.visible.has(box[5])) continue;
    const className = metadata.classes[box[5]];
    const label = `${className} ${(box[4] * 100).toFixed(0)}%`;
    drawDetectionBox(box, colors[box[5] % colors.length], label, Math.max(2, width / 520));
  }

  for (const box of boxes) {
    if (box[5] !== metadata.personClass || !state.visible.has(box[5])) continue;
    const person = personMap.get(box);
    const color = person?.ok ? '#22d47b' : '#ff5d67';
    const status = person?.ok ? 'CUMPLE' : 'NO CUMPLE';
    const label = `Persona ${person?.id || ''} · ${status} · ${(box[4] * 100).toFixed(0)}%`;
    drawDetectionBox(box, color, label, Math.max(3, width / 420));
  }

  renderCompliance(people);
  $('detectionCount').textContent = boxes.length;
}

function renderCompliance(people) {
  const container = $('persons');

  if (!people.length) {
    container.innerHTML = '<p class="empty">No se han detectado personas.</p>';
    $('globalCompliance').textContent = 'SIN PERSONAS';
    $('globalCompliance').className = 'compliance neutral';
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
      ${[...state.required].map(classId => `
        <div class="ppe-row">
          <span>${metadata.classes[classId]}</span>
          <b class="${person.epp.has(classId) ? 'yes' : 'no'}">${person.epp.has(classId) ? '✓ Detectado' : '✗ No detectado'}</b>
        </div>
      `).join('')}
    `;
    container.appendChild(card);
  }

  $('globalCompliance').textContent = allComply ? 'CUMPLE' : 'NO CUMPLE';
  $('globalCompliance').className = `compliance ${allComply ? 'pass' : 'fail'}`;
}

async function processCurrentFrame() {
  if (!session) {
    const [width, height] = sourceDimensions();
    if (width && height) {
      canvas.width = width;
      canvas.height = height;
      drawSource(width, height);
    }
    return;
  }

  const detections = await infer();
  lastBoxes = updateTracks(detections);
  render(lastBoxes);
}

async function loop() {
  if (!running) return;
  frameNo++;

  try {
    if (session && frameNo % state.skip === 0) {
      const detections = await infer();
      lastBoxes = updateTracks(detections);
    }
    render(lastBoxes);
  } catch (error) {
    console.error(error);
    setStatus(`Error de inferencia: ${error.message}`);
    running = false;
    return;
  }

  fpsFrames++;
  const now = performance.now();
  if (now - lastFpsAt > 1000) {
    $('fpsLabel').textContent = `${(fpsFrames * 1000 / (now - lastFpsAt)).toFixed(1)} FPS`;
    fpsFrames = 0;
    lastFpsAt = now;
  }

  if (sourceType !== 'image') requestAnimationFrame(loop);
}

$('loadDefaultBtn').onclick = () => loadDefault().catch(error => {
  setStatus(`No se pudo cargar el modelo predeterminado: ${error.message}`);
});

$('modelInput').onchange = event => {
  const file = event.target.files[0];
  if (file) createSession(file.arrayBuffer()).catch(error => setStatus(error.message));
};

$('metadataInput').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  metadata = JSON.parse(await file.text());
  applyMetadata();
};

$('cameraBtn').onclick = () => startCamera().catch(error => setStatus(error.message));
$('videoInput').onchange = event => event.target.files[0] && openVideo(event.target.files[0]);
$('imageInput').onchange = event => event.target.files[0] && openImage(event.target.files[0]);
$('stopBtn').onclick = stopSource;

$('confRange').oninput = event => {
  state.confidence = Number(event.target.value);
  $('confOut').value = state.confidence.toFixed(2);
};

$('iouRange').oninput = event => {
  state.iou = Number(event.target.value);
  $('iouOut').value = state.iou.toFixed(2);
};

$('frameSkip').onchange = event => {
  state.skip = Number(event.target.value);
};

loadMetadata().catch(error => setStatus(error.message));
