#!/usr/bin/env python3
"""Genera figuras reproducibles desde la evaluación final del Dataset 17."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"YOLO11n": "#18a6c9", "YOLO11s": "#19b979", "YOLO11m": "#ff9f43"}
ORDER = ["YOLO11n", "YOLO11s", "YOLO11m"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 220, "font.size": 10,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
    })


def save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def global_metrics(rows: list[dict], out: Path) -> None:
    metrics = [("Precisión", "precision"), ("Recall", "recall"), ("F1", "f1"),
               ("mAP@0.50", "map50"), ("mAP@0.50:0.95", "map50_95")]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.8))
    indexed = {row["model"]: row for row in rows}
    for index, model in enumerate(ORDER):
        values = [float(indexed[model][key]) * 100 for _, key in metrics]
        bars = ax.bar(x + (index - 1) * width, values, width,
                      label=model, color=COLORS[model])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_title("Comparación global de los modelos YOLO11 · Dataset 17 (test)")
    ax.set_ylabel("Resultado (%)")
    ax.set_xticks(x, [name for name, _ in metrics])
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper center")
    ax.text(0, -0.19,
            "Evaluación controlada: mismo conjunto test, 640 px, batch 16 y Tesla V100.",
            transform=ax.transAxes, fontsize=9)
    save(fig, out / "01_metricas_globales.png")


def per_class(rows: list[dict], out: Path) -> None:
    classes = [row["class_name"] for row in rows if row["model"] == ORDER[0]]
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 6.2))
    for index, model in enumerate(ORDER):
        selected = [row for row in rows if row["model"] == model]
        values = [float(row["map50_95"]) * 100 for row in selected]
        ax.bar(x + (index - 1) * width, values, width,
               label=model, color=COLORS[model])
    ax.set_title("mAP@0.50:0.95 por clase · Dataset 17 (test)")
    ax.set_ylabel("mAP@0.50:0.95 (%)")
    ax.set_xticks(x, classes, rotation=24, ha="right")
    ax.set_ylim(0, 80)
    ax.legend(ncol=3)
    save(fig, out / "02_map5095_por_clase.png")


def accuracy_latency(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for row in rows:
        model = row["model"]
        latency = float(row["inference_ms"])
        score = float(row["map50_95"]) * 100
        weight_size = float(row["weights_size_mb"])
        ax.scatter(latency, score, s=220 + weight_size * 12,
                   color=COLORS[model], alpha=0.88, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{model}\n{latency:.2f} ms · {score:.2f}%", (latency, score),
                    xytext=(9, 7), textcoords="offset points", weight="bold")
    ax.set_title("Compromiso entre exactitud y latencia de inferencia")
    ax.set_xlabel("Latencia en Tesla V100 (ms/imagen; menor es mejor)")
    ax.set_ylabel("mAP@0.50:0.95 (%; mayor es mejor)")
    ax.text(0, -0.18,
            "El área del punto representa el tamaño del peso .pt. La latencia debe medirse nuevamente en cada dispositivo final.",
            transform=ax.transAxes, fontsize=9)
    save(fig, out / "03_exactitud_vs_latencia.png")


def resources(rows: list[dict], out: Path) -> None:
    models = [row["model"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    definitions = [
        ("weights_size_mb", "Tamaño de pesos .pt", "MB"),
        ("maximum_memory_gib", "Memoria máxima", "GiB"),
        ("inference_ms", "Latencia de inferencia", "ms/imagen"),
    ]
    for ax, (key, title, unit) in zip(axes, definitions):
        values = [float(row[key]) for row in rows]
        bars = ax.bar(models, values, color=[COLORS[model] for model in models])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.22)
    fig.suptitle("Recursos y velocidad bajo el mismo entorno de evaluación",
                 fontsize=14, weight="bold")
    save(fig, out / "04_recursos_modelos.png")


def write_summary(rows: list[dict], class_rows: list[dict], out: Path) -> None:
    columns = ["model", "precision", "recall", "f1", "map50", "map50_95",
               "inference_ms", "theoretical_fps", "weights_size_mb", "maximum_memory_gib"]
    with (out / "tabla_comparacion_modelos.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in rows])

    indexed = {row["model"]: row for row in rows}
    summary = {
        "evaluation_scope": "Mismo Dataset 17 test, imagen 640, batch 16, Tesla V100",
        "highest_map50_95": max(rows, key=lambda row: float(row["map50_95"]))["model"],
        "highest_recall": max(rows, key=lambda row: float(row["recall"]))["model"],
        "highest_f1": max(rows, key=lambda row: float(row["f1"]))["model"],
        "lowest_inference_ms": min(rows, key=lambda row: float(row["inference_ms"]))["model"],
        "recommended_default": "YOLO11s",
        "recommendation_basis": {
            "recall": float(indexed["YOLO11s"]["recall"]),
            "f1": float(indexed["YOLO11s"]["f1"]),
            "map50_95": float(indexed["YOLO11s"]["map50_95"]),
            "inference_ms_v100": float(indexed["YOLO11s"]["inference_ms"]),
        },
        "caution": "La latencia V100 no sustituye el benchmark en celular, tableta o laptop.",
        "models": rows,
        "class_metrics": class_rows,
    }
    (out / "resumen_documentacion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['model']}</td>"
        f"<td>{float(row['precision']):.4f}</td>"
        f"<td>{float(row['recall']):.4f}</td>"
        f"<td>{float(row['f1']):.4f}</td>"
        f"<td>{float(row['map50']):.4f}</td>"
        f"<td>{float(row['map50_95']):.4f}</td>"
        f"<td>{float(row['inference_ms']):.3f}</td>"
        "</tr>"
        for row in rows
    )
    dashboard = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparación YOLO11 · Dataset 17</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:28px;color:#142033}}
h1,h2{{color:#071b31}} .notice{{background:#eaf8f2;border-left:5px solid #19b979;padding:14px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccd6df;padding:9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} img{{width:100%;border:1px solid #dce4ea;margin:12px 0 28px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Comparación final YOLO11 · Dataset 17</h1>
<p class="notice"><b>Selección operativa:</b> YOLO11s por equilibrio. YOLO11m maximiza exactitud y YOLO11n minimiza costo.</p>
<table><thead><tr><th>Modelo</th><th>Precisión</th><th>Recall</th><th>F1</th><th>mAP50</th><th>mAP50-95</th><th>ms/img V100</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<h2>Figuras</h2><div class="grid">
<img src="01_metricas_globales.png" alt="Métricas globales">
<img src="02_map5095_por_clase.png" alt="mAP por clase">
<img src="03_exactitud_vs_latencia.png" alt="Exactitud y latencia">
<img src="04_recursos_modelos.png" alt="Recursos de los modelos">
</div>
<p><b>Alcance:</b> la latencia procede de una Tesla V100 y no sustituye el benchmark en celulares, tabletas o laptops.</p>
</body></html>"""
    (out / "dashboard_comparacion.html").write_text(dashboard, encoding="utf-8")


def main() -> None:
    app_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=Path,
        default=app_root / "docs/evaluation/dataset17_models_comparison_v1",
    )
    parser.add_argument("--output", type=Path, default=app_root / "docs/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.evaluation / "comparison_summary.csv")
    rows.sort(key=lambda row: ORDER.index(row["model"]))
    class_rows = read_csv(args.evaluation / "class_metrics.csv")
    apply_style()
    global_metrics(rows, args.output)
    per_class(class_rows, args.output)
    accuracy_latency(rows, args.output)
    resources(rows, args.output)
    write_summary(rows, class_rows, args.output)
    print(f"ESTADO: FIGURAS GENERADAS\nSalida: {args.output}")


if __name__ == "__main__":
    main()
