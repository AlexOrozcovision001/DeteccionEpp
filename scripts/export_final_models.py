#!/usr/bin/env python3
"""Exporta los tres modelos finales del Dataset 17 hacia esta aplicación."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_onnx import export_one


MODELS = {
    "yolo11n_dataset17": "experiments/training/yolo11n_ppe_dataset17_v1/weights/best.pt",
    "yolo11s_dataset17": "experiments/training/yolo11s_ppe_dataset17_v1/weights/best.pt",
    "yolo11m_dataset17": "experiments/training/yolo11m_ppe_dataset17_v1/weights/best.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta YOLO11n/s/m del Dataset 17 a ONNX FP32 fijo 640."
    )
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("/home/aorozcogalan/MROB/PPE_YOLO11"),
        help="Raíz que contiene experiments/training.",
    )
    parser.add_argument(
        "--app-root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Raíz del repositorio DeteccionEpp.",
    )
    args = parser.parse_args()

    missing = [
        args.training_root / relative
        for relative in MODELS.values()
        if not (args.training_root / relative).is_file()
    ]
    if missing:
        details = "\n".join(f" - {path}" for path in missing)
        raise SystemExit(f"ERROR: faltan pesos finales:\n{details}")

    reports = []
    for model_key, relative in MODELS.items():
        weights = args.training_root / relative
        output = args.app_root / "models" / model_key / "model_640.onnx"
        print(f"\n===== EXPORTANDO {model_key} =====")
        reports.append(export_one(weights, output, 640))

    report_path = args.app_root / "models" / "export_summary.json"
    report_path.write_text(
        json.dumps({"status": "completed", "models": reports}, indent=2, ensure_ascii=False)
        + "\n", encoding="utf-8"
    )
    print(f"\nESTADO: 3 MODELOS EXPORTADOS Y VALIDADOS\nInforme: {report_path}")


if __name__ == "__main__":
    main()
