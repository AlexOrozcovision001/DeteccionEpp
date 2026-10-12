# Modelo activo de pruebas · Dataset 17

La aplicación utiliza exclusivamente `yolo11s_dataset17/model_640.onnx`.
Es una copia exacta de `yolo11s_ppe_dataset17_fp32_640.onnx`, recuperado del
archivo compartido el 26 de agosto de 2026; no se ha reentrenado ni reexportado.

- Exportación registrada en ONNX: 2026-08-26T19:16:49.802154+00:00.
- Dataset registrado: `17_PPE_ECUADOR_BALANCED_V1`.
- Entrada `images`: `[1, 3, 640, 640]`, FP32.
- Salida `output0`: `[1, 12, 8400]`, FP32; ocho clases y NMS externo.
- Tamaño: 37938851 bytes.
- SHA-256: `59d50c776166a9620c5a315626114195e5a019b66e5c10a226734d64b9f9fb2a`.

Los informes históricos de comparación n/s/m permanecen en `docs/evaluation`.
No incluyen una huella de los pesos que permita certificar la equivalencia
exacta de este ONNX con el checkpoint evaluado el 31 de agosto.
Las pruebas actuales deben registrar sus propios resultados.

Las versiones anteriores de modelos que existan en el repositorio se conservan
como archivos históricos y no forman parte del catálogo activo.
