import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFile, readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { crc32, makeZip, resultsCsv, initImageBank } from '../js/image-bank.js';

test('inferencia de banco usa imagen completa, letterbox y decodificación a píxeles originales', async () => {
  const source = await readFile(new URL('../js/app.js', import.meta.url), 'utf8');
  const infer = source.slice(source.indexOf('async function infer('), source.indexOf('function decode('));
  const decode = source.slice(source.indexOf('function decode('), source.indexOf('function boxCenter('));
  const roi = source.slice(source.indexOf('function roiPixels('), source.indexOf('function drawRoi('));
  const image = { width: 1280, height: 640 };
  let drawn, disposed = false;
  const workspace = {
    input: new Float32Array(3 * 640 * 640), tensor: {},
    ctx: { fillRect() {}, drawImage(...args) { drawn = args; },
      getImageData: () => ({ data: new Uint8ClampedArray(640 * 640 * 4) }) }
  };
  const context = vm.createContext({
    performance, image,
    state: { inputSize: 640, confidence: .35, iou: .45, roi: { x1: .4, y1: .4, x2: .6, y2: .6 } },
    metadata: { classes: ['Person'] }, VALIDATED_INPUT_SIZE: 640,
    MAX_NMS_CANDIDATES: 1200, MAX_DETECTIONS: 300,
    sourceDimensions() { throw Error('No usar cámara'); },
    ensureCanvasSize() { throw Error('No modificar visor'); },
    sourceObject() { throw Error('No usar cámara'); },
    getInferenceWorkspace: () => workspace,
    pushPerfSample() {}, recordLatencyForAutoTune() {},
    performanceDiagnostics: { preprocess: [], latency: [], postprocess: [], pipeline: [] },
    $: () => ({}),
    session: { inputNames: ['images'], outputNames: ['output0'], run: async () => ({ output0: {
      dims: [1, 1, 6], data: [50, 180, 150, 230, .9, 0], dispose() { disposed = true; }
    } }) }
  });
  vm.runInContext(infer + decode + roi, context);
  const boxes = await vm.runInContext('infer(image)', context);
  assert.deepEqual(drawn, [image, 0, 0, 1280, 640, 0, 160, 640, 320]);
  assert.deepEqual(Array.from(boxes[0]), [100, 40, 300, 140, .9, 0]);
  assert.equal(disposed, true);
});

test('ZIP UTF-8, CRC y contenidos verificables con un lector independiente', async () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
  const zip = await makeZip([
    { name: 'imagenes/0001_casco.jpg', blob: new Blob(['imagen']) },
    { name: 'resultados_ñ.json', blob: new Blob(['{"detections":[]}']) }
  ]);
  await writeFile(new URL('./results-test.zip', import.meta.url), new Uint8Array(await zip.arrayBuffer()));
});

test('CSV conserva vacíos, errores y coordenadas; protege fórmulas y comillas', () => {
  const csv = resultsCsv([
    { file: '=imagen, "1".png', status: 'sin_detecciones', detections: [] },
    { file: 'rota.png', status: 'error', error: 'No decodificable' },
    { file: 'casco.png', status: 'detectada', detections: [
      { classId: 1, className: 'Hardhat', confidence: .9, box: [1, 2, 3, 4] }
    ] }
  ]);
  assert.ok(csv.includes("\"'=imagen, \"\"1\"\".png\""));
  assert.ok(csv.includes('sin_detecciones'));
  assert.ok(csv.includes('No decodificable'));
  assert.ok(csv.includes('"1","Hardhat","0.9","1","2","3","4"'));
});

test('lote continúa tras imagen corrupta, libera bitmaps y permite ZIP parcial/cancelación', async () => {
  const ids = ['bankFiles', 'bankFolder', 'bankStart', 'bankCancel', 'bankDownload', 'bankStatus', 'bankProgress', 'bankPreview'];
  const elements = Object.fromEntries(ids.map(id => [id, { disabled: false, hidden: false, files: [], value: '' }]));
  const downloads = [];
  globalThis.document = {
    getElementById: id => elements[id],
    querySelectorAll: () => ids.filter(id => !['bankStatus', 'bankProgress', 'bankPreview'].includes(id)).map(id => elements[id]),
    body: { appendChild() {} },
    createElement: name => name === 'a' ? { click() { downloads.push(this.download); }, remove() {} } : {
      width: 0, height: 0,
      getContext: () => ({ drawImage() {}, strokeRect() {}, fillRect() {}, fillText() {}, measureText: () => ({ width: 20 }) }),
      toBlob: callback => callback(new Blob(['jpeg']))
    }
  };
  let closed = 0, calls = 0, cancelAfterDetect = false;
  globalThis.createImageBitmap = async file => {
    if (file.name === 'rota.png') throw new Error('Imagen corrupta');
    return { width: 100, height: 50, close() { closed++; } };
  };
  initImageBank({
    prepare: async () => ({ classes: ['Person'], confidence: .35 }),
    detect: async () => {
      calls++;
      if (cancelAfterDetect) elements.bankCancel.onclick();
      return [[1, 2, 20, 30, .9, 0]];
    }
  });
  const select = names => elements.bankFiles.onchange({ target: {
    files: names.map(name => ({ name, size: 10 })), value: ''
  } });
  select(['a.png', 'rota.png', 'b.png', 'omitido.txt']);
  await elements.bankStart.onclick();
  assert.equal(calls, 2);
  assert.equal(closed, 2);
  assert.equal(elements.bankProgress.value, 3);
  assert.match(elements.bankStatus.textContent, /3\/3 imágenes; 1 errores/);
  assert.equal(elements.bankDownload.disabled, false);
  cancelAfterDetect = true;
  select(['a.png', 'b.png']);
  await elements.bankStart.onclick();
  assert.equal(elements.bankProgress.value, 1);
  assert.match(elements.bankStatus.textContent, /Lote detenido/);
  assert.equal(elements.bankDownload.disabled, false);
  // No se descarga aquí para evitar dejar temporizadores de URL vivos en Node.
  assert.equal(downloads.length, 0);
});
