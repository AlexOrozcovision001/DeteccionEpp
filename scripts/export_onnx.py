#!/usr/bin/env python3
"""Exporta y valida un peso Ultralytics para ONNX Runtime Web."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_one(weights: Path, output: Path, imgsz: int = 640) -> dict:
    weights = weights.resolve()
    output = output.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"No existe el peso: {weights}")
    if imgsz != 640:
        raise ValueError("Esta aplicación fue validada exclusivamente a 640 × 640.")

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="epp_onnx_") as temp_name:
        temp = Path(temp_name)
        temp_weights = temp / "source.pt"
        shutil.copy2(weights, temp_weights)
        exported = Path(
            YOLO(str(temp_weights)).export(
                format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                simplify=True, opset=17, half=False, nms=False, device="cpu",
            )
        )
        shutil.copy2(exported, output)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_shape = [int(value) for value in model_input.shape]
    expected_shape = [1, 3, imgsz, imgsz]
    if input_shape != expected_shape:
        raise RuntimeError(f"Entrada ONNX {input_shape}; se esperaba {expected_shape}.")

    sample = np.zeros(expected_shape, dtype=np.float32)
    test_started = time.perf_counter()
    outputs = session.run(None, {model_input.name: sample})
    test_ms = (time.perf_counter() - test_started) * 1000
    if not outputs or not np.isfinite(outputs[0]).all():
        raise RuntimeError("La inferencia de validación produjo una salida inválida.")

    report = {
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_weights": str(weights),
        "onnx_model": str(output),
        "format": "ONNX",
        "precision": "FP32",
        "image_size": imgsz,
        "batch": 1,
        "opset": 17,
        "dynamic": False,
        "simplified": True,
        "integrated_nms": False,
        "input_name": model_input.name,
        "input_shape": input_shape,
        "outputs": [
            {"name": item.name, "shape": [int(v) for v in item.shape], "type": item.type}
            for item in session.get_outputs()
        ],
        "export_seconds": round(time.perf_counter() - started, 2),
        "cpu_test_inference_ms": round(test_ms, 2),
        "file_size_mb": round(output.stat().st_size / 1024**2, 2),
        "sha256": sha256(output),
        "validation": "passed",
    }
    output.with_suffix(".export.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(export_one(args.weights, args.output, args.imgsz), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
