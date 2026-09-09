# Evidencia cuantitativa para la tesis

Este archivo organiza datos comprobados; no sustituye la redacción personal de
la metodología ni de las conclusiones.

## Condiciones de comparación

| Elemento controlado | Valor |
|---|---|
| Dataset | 17_PPE_ECUADOR_BALANCED_V1 |
| Partición | test |
| Resolución | 640 × 640 |
| Batch de evaluación | 16 |
| GPU | Tesla V100-PCIE-16GB |
| Confianza de evaluación | 0.001 |
| IoU de evaluación | 0.7 |
| Máximo de detecciones | 300 |
| Precisión numérica | FP32 |

## Comparación global

| Modelo | Precisión | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 | Inferencia V100 (ms/img) | Peso PT (MB) | Memoria (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.8123 | 0.7424 | 0.7758 | 0.8011 | 0.5074 | 2.301 | 5.211 | 0.445 |
| YOLO11s | 0.8129 | **0.7832** | **0.7977** | 0.8208 | 0.5286 | 3.044 | 18.281 | 0.793 |
| YOLO11m | **0.8279** | 0.7692 | 0.7975 | **0.8226** | **0.5331** | 7.194 | 38.633 | 1.479 |

## Lectura objetiva de los resultados

- Mayor mAP@0.50:0.95: YOLO11m, con 0.5331.
- Mayor mAP@0.50 y precisión: YOLO11m.
- Mayor recall y F1: YOLO11s.
- Menor latencia, peso y memoria en la V100: YOLO11n.
- Modelo predeterminado de la aplicación: YOLO11s, debido al equilibrio entre
  el recall más alto, el F1 más alto y una exactitud cercana a YOLO11m.
- YOLO11m queda disponible cuando se prioriza la exactitud sobre el costo.
- YOLO11n queda disponible para dispositivos con recursos más limitados.

## Diferencias relevantes

| Comparación | Resultado |
|---|---:|
| mAP@0.50:0.95 de YOLO11m frente a YOLO11s | +0.0044 |
| Recall de YOLO11s frente a YOLO11m | +0.0139 |
| Latencia V100 de YOLO11m / YOLO11s | 2.36× |
| Latencia V100 de YOLO11s / YOLO11n | 1.32× |

Estas diferencias describen el experimento controlado en V100. Para afirmar
rendimiento operativo en celulares, tabletas y laptops se deben registrar al
menos latencia media, percentil 95, IPS/FPS, estabilidad, memoria estimada y
consumo de batería durante pruebas separadas con el mismo video o conjunto de
imágenes.

## Figuras generadas

1. `01_metricas_globales.png`: precisión, recall, F1 y mAP globales.
2. `02_map5095_por_clase.png`: desempeño por cada una de las ocho clases.
3. `03_exactitud_vs_latencia.png`: relación entre exactitud y velocidad.
4. `04_recursos_modelos.png`: tamaño, memoria y latencia.

## Fuente y trazabilidad

Los valores proceden de `comparison_summary.csv`, `class_metrics.csv` y
`comparison_report.json`. Los resultados completos, curvas y matrices están en
`docs/evaluation/dataset17_models_comparison_v1/`.
