# Detección EPP en tiempo real

Aplicación web estática para ejecutar un modelo YOLO de EPP en el navegador con ONNX Runtime Web.

## Clases

0 Person, 1 Hardhat, 2 Safety Vest, 3 Safety Glasses, 4 Gloves, 5 Safety Boots, 6 Regular Glasses, 7 Regular Shoes.

## Preparación

GitHub Pages no ejecuta directamente archivos `.pt`. En Visual Studio Code:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install ultralytics onnx onnxsim
python scripts/export_onnx.py --weights best.pt --imgsz 640
```

Primero pruebe la exportación sin `--nms`. Copie el ONNX generado como:

```text
models/epp-yolo11/model.onnx
```

También puede abrir la web y usar **Cargar ONNX**, sin subir el modelo a GitHub.

## Publicación

1. Suba todos los archivos a la rama `main`.
2. Abra `Settings > Pages`.
3. Seleccione `Deploy from a branch`, `main` y `/root`.
4. Abra la URL de GitHub Pages usando HTTPS para permitir la cámara.

## Consideraciones

- WebGPU se usa cuando está disponible; WASM funciona como respaldo.
- Para más FPS, seleccione procesar cada 2, 3 o 4 frames.
- El cumplimiento se evalúa por persona mediante asociación espacial y zonas anatómicas.
- La comparación de precisión real entre modelos requiere un conjunto etiquetado; la web compara rendimiento visual y velocidad.
