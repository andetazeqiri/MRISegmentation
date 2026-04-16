#!/usr/bin/env python3
"""Sequential API stress test for the BraTS segmentation service.

This script sends multiple requests in sequence, measures per-request latency,
and stores the results in a CSV file for thesis reporting.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import statistics
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential API stress test.")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="Backend prediction endpoint.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("./outputs/BraTS20_Training_001_slice_071.npy"),
        help="Input MRI file (.npy, .nii, .nii.gz) used for all requests.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "resunet"],
        help="Model architecture to test.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of sequential requests to send.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        help="Slice axis for NIfTI inputs.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index for NIfTI inputs. Defaults to the middle slice if omitted.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV file for request-level results. If omitted, a timestamped file in outputs/ is created.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def _build_form_data(filename: str, model_type: str, axis: int, slice_index: int | None) -> dict[str, str]:
    data = {"model_type": model_type}
    lower = filename.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        data["axis"] = str(axis)
        if slice_index is not None:
            data["slice_index"] = str(slice_index)
    return data


def main() -> None:
    args = parse_args()

    if not args.input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    raw = args.input_file.read_bytes()
    filename = args.input_file.name

    if args.output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = Path("./outputs") / f"stress_test_{timestamp}.csv"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    success_count = 0
    error_count = 0

    print(f"Running stress test against: {args.api_url}")
    print(f"Input file: {args.input_file}")
    print(f"Model: {args.model_type}")
    print(f"Requests: {args.repeat}")

    for request_index in range(1, args.repeat + 1):
        files = {"file": (filename, raw, "application/octet-stream")}
        data = _build_form_data(filename, args.model_type, args.axis, args.slice_index)

        started = time.perf_counter()
        status_code = None
        error_message = ""
        mean_confidence = None
        predicted_shape = ""

        try:
            response = requests.post(args.api_url, files=files, data=data, timeout=args.timeout)
            status_code = response.status_code
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)

            payload = response.json()
            if response.ok:
                success_count += 1
                mean_confidence = payload.get("mean_confidence")
                predicted_shape = str(payload.get("prediction_shape", payload.get("shape", "")))
            else:
                error_count += 1
                error_message = str(payload.get("detail", payload))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            error_count += 1
            error_message = str(exc)

        rows.append(
            {
                "request_index": request_index,
                "status_code": status_code,
                "latency_ms": round(elapsed_ms, 3),
                "mean_confidence": None if mean_confidence is None else round(float(mean_confidence), 6),
                "prediction_shape": predicted_shape,
                "error": error_message,
            }
        )
        print(f"[{request_index:02d}] status={status_code} latency_ms={elapsed_ms:.2f} error={error_message or '-'}")

    average_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
    min_latency = min(latencies_ms) if latencies_ms else 0.0
    max_latency = max(latencies_ms) if latencies_ms else 0.0

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["request_index", "status_code", "latency_ms", "mean_confidence", "prediction_shape", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "requests",
                "success_count",
                "error_count",
                "average_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
                "model_type",
                "input_file",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "requests": args.repeat,
                "success_count": success_count,
                "error_count": error_count,
                "average_latency_ms": round(average_latency, 3),
                "min_latency_ms": round(min_latency, 3),
                "max_latency_ms": round(max_latency, 3),
                "model_type": args.model_type,
                "input_file": str(args.input_file),
            }
        )

    print("\nStress test complete.")
    print(f"Successful requests: {success_count}/{args.repeat}")
    print(f"Average latency (ms): {average_latency:.2f}")
    print(f"Min / Max latency (ms): {min_latency:.2f} / {max_latency:.2f}")
    print(f"Results CSV: {args.output_csv}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
