# Detección EPP en tiempo real

Aplicación web estática para detección de EPP con modelos YOLO exportados a ONNX y ejecutados directamente en el navegador mediante ONNX Runtime Web.

## Funciones

- Cámara, video e imagen local.
- WebGPU con respaldo WASM.
- Detección de 8 clases de EPP.
- Seguimiento temporal de personas con ID persistente.
- Persistencia e histéresis de EPP para reducir el parpadeo de detecciones.
- Selección interactiva de ROI (Region of Interest).
- EPP obligatorios configurables y evaluación CUMPLE / NO CUMPLE por persona.
- Reporte descargable en CSV y JSON.
- Reporte imprimible / guardable como PDF desde el navegador.
- Envío opcional de resumen a ThingSpeak mediante Write API Key ingresada por el usuario.
- Carga local de modelos ONNX y metadatos JSON compatibles.

## Orden de uso recomendado

1. Pulsar **Cargar predeterminado**.
2. Seleccionar los EPP obligatorios.
3. Opcionalmente seleccionar un ROI.
4. Iniciar cámara, abrir video o abrir imagen.
5. Revisar IDs de personas y cumplimiento.
6. Descargar el reporte o enviarlo a ThingSpeak.

## Modelo

GitHub Pages no ejecuta `.pt`. El modelo debe estar exportado a ONNX y ubicado en:

```text
models/epp-yolo11/model.onnx
```

Los metadatos están en:

```text
models/epp-yolo11/metadata.json
```

## GitHub Pages

Publicar desde la rama `main`, carpeta `/ (root)`.


## Perfiles de rendimiento 640 / 512 / 480

La aplicación v6 puede cambiar entre tres modelos ONNX estáticos. Un modelo exportado a 640 no se vuelve realmente más rápido solo porque el navegador redimensione la imagen; para reducir cómputo se necesitan exportaciones separadas.

Desde la raíz del proyecto, con el entorno de Ultralytics activo:

```bash
python scripts/export_web_profiles.py --weights best.pt
```

Se crearán:

```text
models/epp-yolo11/model.onnx      # 640x640
models/epp-yolo11/model_512.onnx  # 512x512
models/epp-yolo11/model_480.onnx  # 480x480
```

La interfaz muestra FPS, latencia e IPS. En modo automático, si la latencia media es alta baja 640→512→480; si el dispositivo es muy rápido puede volver a subir la resolución. Si falta uno de los perfiles, el modo automático se detiene y la interfaz lo indica.


## Catálogo multi-modelo

La interfaz soporta tres entrenamientos y tres resoluciones por entrenamiento:

- YOLO11s modelo anterior: 640, 512 y 480.
- YOLO11s Dataset 15: 640, 512 y 480.
- YOLO11m Dataset 15: 640, 512 y 480.

Cada modelo se carga bajo demanda; abrir la página no descarga los nueve archivos ONNX.
