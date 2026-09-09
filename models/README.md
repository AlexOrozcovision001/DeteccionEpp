# Modelos finales del Dataset 17

Esta aplicación carga un solo modelo a la vez. Los tres archivos se generan a
640 × 640, lote 1, FP32, opset 17, grafo simplificado y NMS externo.

Ubicación esperada:

```text
models/yolo11n_dataset17/model_640.onnx
models/yolo11s_dataset17/model_640.onnx
models/yolo11m_dataset17/model_640.onnx
```

Los archivos ONNX no se incluyen en este paquete porque los ZIP recibidos no
contenían los pesos finales. Ejecute `scripts/export_final_models.py` en el
servidor que contiene los tres `best.pt`.
