# EPP Guard · Pruebas con YOLO11s Dataset 17

Aplicación web para detección local de equipos de protección personal con
ONNX Runtime Web. La versión 15 utiliza un único modelo recuperado:
`models/yolo11s_dataset17/model_640.onnx`.

## Uso

1. Abra la aplicación en GitHub Pages y configure o desbloquee el administrador.
2. Pulse **Cargar modelo**, o **Iniciar monitoreo** para cargarlo e iniciar la cámara.
3. En el administrador también puede abrir una imagen o un video para las pruebas.
4. Configure los EPP obligatorios y use Benchmark para comparar dispositivos.

La selección de modelo y resolución está fija. Las preferencias guardadas de
versiones anteriores se migran a YOLO11s y 640 px conservando los demás ajustes.
La aplicación ya no permite importar otro ONNX ni otros metadatos desde la interfaz.

## Configuración

- Entrada fija `images`: `1 × 3 × 640 × 640`, FP32.
- Salida `output0`: `1 × 12 × 8400`, con NMS externo.
- Clases: Person, Hardhat, Safety Vest, Safety Glasses, Gloves,
  Safety Boots, Regular Glasses y Regular Shoes.
- WebGPU en navegadores compatibles y WASM como respaldo.
- Procedencia y SHA-256: consulte `models/README.md`.

## Prueba local

```bash
python3 -m http.server 8000
```

Abra `http://localhost:8000`. En otro dispositivo, la cámara requiere HTTPS.
GitHub Pages ofrece HTTPS. No abra `index.html` mediante `file://`.

## Resultados y documentación

Los informes de la comparación histórica n/s/m de Dataset 17 permanecen en
`docs/evaluation/dataset17_models_comparison_v1`. No certifican por sí solos
la equivalencia exacta del ONNX recuperado con aquellos checkpoints.
Use las pruebas actuales para medir la carga y la inferencia en cada dispositivo.
Los scripts de exportación se conservan para cuando se recuperen los pesos `.pt`.

## Publicación

Publique la rama `main` desde la raíz. El ONNX está incluido como archivo binario
normal. La versión de caché y las referencias de la aplicación se actualizaron
para que los navegadores reciban la configuración de modelo único.

Las claves opcionales de evidencia y ThingSpeak se conservan durante la pestaña
activa. La carga del modelo y la inferencia se realizan en el dispositivo.

## Banco de imágenes (versión 19)

En el panel del administrador, use **Seleccionar imágenes** para cargar varios
archivos o **Seleccionar carpeta** para incluir las imágenes de sus subcarpetas.
Pulse **Aplicar detección**; el modelo fijo se carga automáticamente si hace falta.
El monitoreo se detiene mientras se procesa el lote. Para retomarlo, inicie la cámara.

- Formatos: JPG/JPEG, PNG, WebP y BMP. Los demás archivos se omiten.
- Máximo 500 imágenes por lote, 40 MB y 24 megapíxeles por imagen.
- Se procesa una imagen a la vez, con la confianza y el IoU configurados.
- Se usa la imagen completa, sin ROI, seguimiento ni persistencia temporal.
- **Cancelar lote** termina la imagen en curso y conserva los resultados parciales.
- **Descargar resultados ZIP** genera `imagenes/` con JPEG anotados,
  `detecciones.csv` y `resultados.json`. Los informes incluyen archivos sin
  detecciones y errores; el JSON conserva las clases, configuración y coordenadas
  `[x1,y1,x2,y2]` en píxeles de la imagen decodificada.
- Los nombres de salida llevan un índice para evitar colisiones entre carpetas.
- Al alcanzar 100 MB de imágenes de salida, el lote se detiene y permite descargar
  lo procesado. Divida bancos grandes en varios lotes.

La detección y exportación son locales y no envían el banco al servidor de
 evidencias ni a ThingSpeak. El banco informa detecciones por imagen; no genera
 dictámenes de cumplimiento basados en la persistencia temporal de la cámara.
Los resultados se conservan en memoria hasta iniciar otro lote o cerrar/recargar
la página. Descargue el ZIP antes de hacerlo.

