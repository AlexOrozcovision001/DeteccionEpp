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
