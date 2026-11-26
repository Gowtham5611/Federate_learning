"""Simple smoke test script to verify /health and CSV upload against local or Render backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Test FedSecureAI CSV upload & prediction endpoint.")
    parser.add_argument(
        "--url",
        default="https://federate-learning.onrender.com",
        help="Backend base URL (e.g. http://127.0.0.1:8000 or https://federate-learning.onrender.com)",
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "data" / "sample_demo_20_rows.csv"),
        help="Path to sample CSV file",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"1. Checking GET {base}/health ...")
    health_resp = requests.get(f"{base}/health", timeout=90)
    print(f"   Status: {health_resp.status_code}")
    print(f"   Body:   {health_resp.json()}")

    csv_path = Path(args.csv)
    print(f"\n2. Uploading {csv_path.name} via multipart/form-data to POST {base}/api/v1/predict/upload ...")
    with open(csv_path, "rb") as f:
        upload_resp = requests.post(
            f"{base}/api/v1/predict/upload",
            files={"file": (csv_path.name, f, "text/csv")},
            data={"model_id": "auto"},
            timeout=90,
        )
    print(f"   Status: {upload_resp.status_code}")
    data = upload_resp.json()
    print(
        json.dumps(
            {
                "prediction_id": data.get("prediction_id"),
                "model_name": data.get("model_name"),
                "evaluated_sequences": data.get("evaluated_sequences"),
                "flagged_laundering_count": data.get("flagged_laundering_count"),
                "evaluation_metrics": data.get("evaluation_metrics"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
