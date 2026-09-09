# EPP Guard · Dataset 17

Aplicación web para detección local de equipos de protección personal con
ONNX Runtime Web. Permite comparar los modelos finales YOLO11n, YOLO11s y
YOLO11m entrenados con el mismo Dataset 17.

## Configuración controlada

- Entrada fija: `1 × 3 × 640 × 640`.
- Precisión: FP32.
- Salida esperada: `1 × 12 × 8400`.
- Ocho clases: Person, Hardhat, Safety Vest, Safety Glasses, Gloves,
  Safety Boots, Regular Glasses y Regular Shoes.
- WebGPU en navegadores compatibles y WASM como respaldo.
- YOLO11s es el modelo predeterminado por su equilibrio entre F1, recall,
  exactitud y latencia.

## Exportar los tres modelos

En el servidor que contiene los pesos finales:

```bash
cd /ruta/al/repositorio/DeteccionEpp
source /home/aorozcogalan/MROB/PPE_YOLO11/.venv_yolo11_84116/bin/activate
pip install -r requirements.txt

python scripts/export_final_models.py \
  --training-root /home/aorozcogalan/MROB/PPE_YOLO11 \
  --app-root "$PWD"
```

El script no modifica los entrenamientos. Copia temporalmente cada `best.pt`,
exporta y valida los ONNX, y genera:

```text
models/yolo11n_dataset17/model_640.onnx
models/yolo11s_dataset17/model_640.onnx
models/yolo11m_dataset17/model_640.onnx
models/export_summary.json
```

## Probar localmente

No abra `index.html` con `file://`; la cámara, los modelos y el service worker
requieren un servidor HTTP/HTTPS.

```bash
python3 -m http.server 8000
```

Abra `http://localhost:8000`. En otro dispositivo de la red la cámara exige
HTTPS; GitHub Pages ya cumple este requisito.

## Generar figuras para la tesis

```bash
python scripts/generate_thesis_figures.py
```

Los PNG, la tabla CSV y el resumen JSON quedan en `docs/figures/`. La fuente
original permanece en `docs/evaluation/dataset17_models_comparison_v1/`.

## Publicar en GitHub Pages

Después de verificar que los tres ONNX estén presentes:

```bash
git add index.html css js manifest.webmanifest sw.js assets models scripts docs
git commit -m "Integra modelos finales Dataset 17 y evidencia de tesis"
git push origin main
```

GitHub no acepta archivos individuales mayores de 100 MB. Los ONNX FP32 de
estas variantes normalmente quedan por debajo de ese límite, pero debe
comprobarse antes del commit:

```bash
find models -name '*.onnx' -printf '%s %p\n' | sort -n
```

## Seguridad y alcance de los resultados

- Las claves de evidencia y ThingSpeak se conservan solo durante la pestaña
  activa; no se guardan de forma persistente en el navegador.
- Las métricas de exactitud provienen del mismo conjunto `test` del Dataset 17.
- Las latencias de la V100 permiten comparar modelos bajo un entorno común,
  pero no predicen la velocidad en celulares, tabletas o laptops. Use el panel
  Benchmark en cada dispositivo final.
