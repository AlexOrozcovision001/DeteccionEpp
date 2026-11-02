// ZIP sin compresión: las imágenes JPEG ya están comprimidas. Sin dependencias externas.
export function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export async function makeZip(entries) {
  const chunks = [], directory = [];
  let offset = 0, directorySize = 0;
  const header = size => {
    const bytes = new Uint8Array(size);
    return [bytes, new DataView(bytes.buffer)];
  };
  for (const entry of entries) {
    const name = new TextEncoder().encode(entry.name);
    const bytes = new Uint8Array(await entry.blob.arrayBuffer());
    const crc = crc32(bytes);
    const [local, l] = header(30);
    l.setUint32(0, 0x04034b50, true);
    l.setUint16(4, 20, true);
    l.setUint16(6, 0x800, true); // UTF-8
    l.setUint16(12, 33, true); // 1980-01-01
    l.setUint32(14, crc, true);
    l.setUint32(18, bytes.length, true);
    l.setUint32(22, bytes.length, true);
    l.setUint16(26, name.length, true);
    chunks.push(local, name, entry.blob);
    const [central, c] = header(46);
    c.setUint32(0, 0x02014b50, true);
    c.setUint16(4, 20, true);
    c.setUint16(6, 20, true);
    c.setUint16(8, 0x800, true);
    c.setUint16(14, 33, true);
    c.setUint32(16, crc, true);
    c.setUint32(20, bytes.length, true);
    c.setUint32(24, bytes.length, true);
    c.setUint16(28, name.length, true);
    c.setUint32(42, offset, true);
    directory.push(central, name);
    directorySize += central.length + name.length;
    offset += local.length + name.length + bytes.length;
  }
  const [end, e] = header(22);
  e.setUint32(0, 0x06054b50, true);
  e.setUint16(8, entries.length, true);
  e.setUint16(10, entries.length, true);
  e.setUint32(12, directorySize, true);
  e.setUint32(16, offset, true);
  return new Blob([...chunks, ...directory, end], { type: 'application/zip' });
}

export function csvCell(value) {
  let text = String(value ?? '');
  // Evita que nombres de archivo se interpreten como fórmulas al abrir el CSV.
  if (/^[\s]*[=+@-]/.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"', '""') + '"';
}

export function resultsCsv(results) {
  const rows = [['archivo', 'estado', 'error', 'anotada', 'ancho', 'alto', 'tiempo_ms', 'clase_id', 'clase', 'confianza', 'x1', 'y1', 'x2', 'y2']];
  for (const row of results) {
    for (const box of row.detections?.length ? row.detections : [null]) {
      rows.push([row.file, row.status, row.error, row.annotated, row.width, row.height, row.elapsedMs,
        box?.classId, box?.className, box?.confidence, ...(box?.box || ['', '', '', ''])]);
    }
  }
  return '\ufeff' + rows.map(row => row.map(csvCell).join(',')).join('\r\n');
}

export function initImageBank({ prepare, detect }) {
  const $ = id => document.getElementById(id);
  let files = [], results = [], entries = [], config = null;
  let active = false, cancelled = false, previewUrl = null, selectedCount = 0;
  const status = text => { $('bankStatus').textContent = text; };
  const supported = file => /\.(jpe?g|png|webp|bmp)$/i.test(file.name);
  const select = event => {
    const selected = [...event.target.files];
    const images = selected.filter(supported);
    files = images.slice(0, 500);
    $('bankStart').disabled = !files.length;
    status(`${files.length} imágenes seleccionadas. ${selected.length - images.length} archivos omitidos.` +
      (images.length > 500 ? ' Se usarán las primeras 500 imágenes.' : ''));
    event.target.value = '';
  };
  $('bankFiles').onchange = select;
  $('bankFolder').onchange = select;
  $('bankCancel').onclick = () => {
    cancelled = true;
    $('bankCancel').disabled = true;
    status('Cancelando después de la imagen en curso…');
  };
  $('bankStart').onclick = async () => {
    if (active || !files.length) return;
    active = true;
    cancelled = false;
    results = [];
    entries = [];
    config = null;
    selectedCount = files.length;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $('bankPreview').hidden = true;
    const controls = [...document.querySelectorAll('button, input, select')];
    const disabled = controls.map(control => control.disabled);
    controls.forEach(control => { control.disabled = true; });
    $('bankCancel').disabled = false;
    $('bankProgress').value = 0;
    $('bankProgress').max = files.length;
    let outputBytes = 0, failure = '';
    try {
      status('Preparando modelo…');
      config = await prepare();
      config = { ...config, startedAt: new Date().toISOString() };
      for (const [index, file] of files.entries()) {
        if (cancelled) break;
        const row = { file: file.webkitRelativePath || file.name, status: 'error', detections: [] };
        const started = performance.now();
        let image = null;
        let surface = null;
        try {
          if (file.size > 40 * 1024 * 1024) throw new Error('Archivo mayor a 40 MB.');
          image = await createImageBitmap(file);
          if (image.width * image.height > 24000000) throw new Error('Imagen mayor a 24 megapíxeles.');
          row.width = image.width;
          row.height = image.height;
          status(`Detectando ${index + 1}/${files.length}: ${file.name}`);
          const boxes = await detect(image);
          row.detections = boxes.map(box => ({ classId: box[5], className: config.classes[box[5]],
            confidence: box[4], box: box.slice(0, 4) }));
          surface = document.createElement('canvas');
          surface.width = image.width;
          surface.height = image.height;
          const ctx = surface.getContext('2d');
          ctx.drawImage(image, 0, 0);
          const font = Math.max(14, Math.round(image.width / 75));
          ctx.font = `bold ${font}px system-ui`;
          ctx.lineWidth = Math.max(2, image.width / 500);
          for (const box of row.detections) {
            const [x1, y1, x2, y2] = box.box;
            const label = `${box.className} ${(box.confidence * 100).toFixed(1)}%`;
            const y = Math.max(font + 4, y1);
            ctx.strokeStyle = '#00e5ff';
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = '#07111f';
            ctx.fillRect(x1, y - font - 4, ctx.measureText(label).width + 8, font + 8);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 4, y);
          }
          const blob = await new Promise(resolve => surface.toBlob(resolve, 'image/jpeg', 0.9));
          if (!blob) throw new Error('No se pudo generar la imagen anotada.');
          if (outputBytes + blob.size > 100 * 1024 * 1024) {
            cancelled = true;
            throw new Error('Límite de 100 MB de resultados alcanzado. Continúe en otro lote.');
          }
          outputBytes += blob.size;
          const safeName = file.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 100) || 'imagen';
          row.annotated = `imagenes/${String(index + 1).padStart(4, '0')}_${safeName}.jpg`;
          entries.push({ name: row.annotated, blob });
          row.status = boxes.length ? 'detectada' : 'sin_detecciones';
          if (previewUrl) URL.revokeObjectURL(previewUrl);
          previewUrl = URL.createObjectURL(blob);
          $('bankPreview').src = previewUrl;
          $('bankPreview').hidden = false;
        } catch (error) {
          row.error = error.message || String(error);
        } finally {
          image?.close();
          if (surface) { surface.width = 0; surface.height = 0; }
          row.elapsedMs = Math.round(performance.now() - started);
          results.push(row);
          $('bankProgress').value = results.length;
        }
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    } catch (error) {
      failure = error.message || String(error);
    } finally {
      active = false;
      controls.forEach((control, index) => { control.disabled = disabled[index]; });
      $('bankCancel').disabled = true;
      $('bankDownload').disabled = !results.length;
      const errors = results.filter(row => row.status === 'error').length;
      status(failure ? `No se pudo iniciar: ${failure}` :
        `${cancelled ? 'Lote detenido' : 'Lote terminado'}: ${results.length}/${files.length} imágenes; ${errors} errores. Puede descargar los resultados.`);
    }
  };
  $('bankDownload').onclick = async () => {
    if (active || !results.length) return;
    active = true;
    $('bankDownload').disabled = true;
    $('bankStart').disabled = true;
    $('bankFiles').disabled = true;
    $('bankFolder').disabled = true;
    try {
      status('Generando ZIP…');
      const payload = { configuration: config, completedAt: new Date().toISOString(),
        cancelled, selected: selectedCount, processed: results.length, results };
      const blob = await makeZip([...entries,
        { name: 'detecciones.csv', blob: new Blob([resultsCsv(results)], { type: 'text/csv;charset=utf-8' }) },
        { name: 'resultados.json', blob: new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }) }
      ]);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `deteccion_banco_${new Date().toISOString().replaceAll(':', '-')}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      status('ZIP generado con imágenes anotadas e informes CSV y JSON.');
    } catch (error) {
      status(`No se pudo generar el ZIP: ${error.message}`);
    } finally {
      active = false;
      $('bankDownload').disabled = false;
      $('bankStart').disabled = !files.length;
      $('bankFiles').disabled = false;
      $('bankFolder').disabled = false;
    }
  };
}
