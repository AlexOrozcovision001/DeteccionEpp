# Auditoría técnica de la aplicación

## Correcciones aplicadas

| Hallazgo | Corrección |
|---|---|
| Catálogo apuntaba a modelos anteriores y al Dataset 15 | Sustituido por YOLO11n/s/m del Dataset 17 |
| Selector permitía 480, 512 y 640 con un ONNX final fijo a 640 | Entrada bloqueada a 640 para evitar incompatibilidad de forma |
| ONNX Runtime Web utilizaba la etiqueta móvil `@dev` | Versión fijada en 1.27.0 |
| Los reportes no distinguían la variante n/s/m | Se agregaron nombre, clave del modelo, dataset y versiones |
| NMS podía procesar una cantidad ilimitada de candidatos | Se limitaron candidatos y detecciones |
| La aplicación declaraba instalación, pero no tenía service worker | Se agregó caché del shell e iconos PWA |
| Claves externas persistían en `localStorage` | Se cambiaron a `sessionStorage` |
| La documentación describía modelos y resoluciones antiguas | Se reemplazó por instrucciones del Dataset 17 |

## Elementos conservados

- Preprocesamiento letterbox, RGB, CHW y normalización 0–1.
- Decodificación de la salida sin NMS integrado `1 × 12 × 8400`.
- NMS por clase.
- Asociación temporal de persona y EPP.
- ROI, imagen, video, cámara, evidencias y reportes.
- WebGPU con respaldo WASM.
- PIN local derivado mediante PBKDF2.
- Benchmark reproducible con modelo y resolución fijos.

## Pendiente antes de publicar

Los dos ZIP analizados no contienen los tres ONNX finales. Debe ejecutarse
`scripts/export_final_models.py` en el servidor que sí contiene los `best.pt`.
La aplicación no debe publicarse como terminada hasta verificar que existan:

```text
models/yolo11n_dataset17/model_640.onnx
models/yolo11s_dataset17/model_640.onnx
models/yolo11m_dataset17/model_640.onnx
```

Después se debe abrir cada variante, ejecutar al menos una inferencia con la
misma imagen y realizar un benchmark separado en cada dispositivo objetivo.
