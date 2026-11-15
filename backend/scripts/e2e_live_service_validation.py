"""Phase 15: Live HTTP Service Socket & Route Transport Validation Script.

Starts the Uvicorn FastAPI server in a controlled subprocess, performs live HTTP requests
over localhost:8000, validates all endpoints and demo scenarios, and cleanly terminates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE_DIR = Path(__file__).resolve().parents[1]
UVICORN_EXE = BASE_DIR / ".venv/Scripts/uvicorn.exe"
PYTHON_EXE = BASE_DIR / ".venv/Scripts/python.exe"

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))


def wait_for_server(url: str, timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def http_get(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "FedSecureAI-Phase15-Validator"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        data = json.loads(e.read().decode("utf-8"))
        return e.code, data


def http_post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "FedSecureAI-Phase15-Validator",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        data = json.loads(e.read().decode("utf-8"))
        return e.code, data


def main():
    print("=" * 70)
    print("FedSecureAI — Phase 15 Live HTTP Socket Validation")
    print("=" * 70)

    # 1. Start Uvicorn subprocess
    env = dict(os.environ)
    backend_dir = str(BASE_DIR / "backend")
    env["PYTHONPATH"] = backend_dir + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    cmd = [
        str(PYTHON_EXE),
        "-m",
        "uvicorn",
        "interfaces.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "warning",
    ]
    print(f"Spawning backend: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        print("Waiting for http://127.0.0.1:8000/health...")
        if not wait_for_server("http://127.0.0.1:8000/health", timeout=10.0):
            print("ERROR: Server did not become healthy within 10s.")
            proc.poll()
            out, err = proc.communicate(timeout=2.0)
            print("SUBPROCESS STDOUT:\n", out)
            print("SUBPROCESS STDERR:\n", err)
            sys.exit(1)
        print("Backend server is online and accepting connections.")

        # 2. Test GET /health
        code, health = http_get("http://127.0.0.1:8000/health")
        print(f"\n[1] GET /health -> HTTP {code}")
        print(f"    Status: {health.get('status')}, Readiness: {health.get('readiness')}")
        print(f"    Loaded Models: {health.get('loaded_models')}")
        assert code == 200 and health.get("status") == "ok" and health.get("readiness") == "ready"

        # 3. Test GET /models
        code, models = http_get("http://127.0.0.1:8000/models")
        print(f"\n[2] GET /models -> HTTP {code}")
        print(f"    Models count: {len(models)}")
        assert code == 200 and len(models) == 2

        # 4. Test GET /metadata/HI-Small and /metadata/LI-Medium
        for ds in ("HI-Small", "LI-Medium"):
            code, meta = http_get(f"http://127.0.0.1:8000/metadata/{ds}")
            print(f"\n[3] GET /metadata/{ds} -> HTTP {code}")
            print(f"    Threshold: {meta.get('frozen_threshold')}, Features: {meta.get('expanded_feature_count')}")
            assert code == 200

        # 5. Test Demonstration Scenarios on POST /predict
        import re
        ts_file = BASE_DIR / "frontend/src/demo/demoSamples.ts"
        assert ts_file.exists(), "frontend/src/demo/demoSamples.ts must exist"
        content = ts_file.read_text(encoding="utf-8")
        match = re.search(r"export const DEMO_SAMPLES: DemoSample\[\] = (\[.*\]);", content, re.DOTALL)
        assert match, "Could not extract DEMO_SAMPLES from demoSamples.ts"
        samples = json.loads(match.group(1))

        print("\n[4] POST /predict (4 Demonstration Scenarios):")
        for s in samples:
            payload = {"dataset": s["dataset"], "sequence": s["sequence"]}
            code, res = http_post("http://127.0.0.1:8000/predict", payload)
            print(f"    - Sample: {s['id']}")
            print(f"      HTTP {code} | Prob: {res.get('predicted_probability'):.6f} | Thresh: {res.get('decision_threshold'):.6f} | Decision: {res.get('decision')} | Latency: {res.get('inference_time_ms'):.2f}ms")
            assert code == 200
            if "legit" in s["id"]:
                assert res.get("decision") == "CLEARED_LEGITIMATE"
            else:
                assert res.get("decision") == "FLAGGED_LAUNDERING"

        # 6. Test POST /explain on alert sample
        alert_sample = next(s for s in samples if "alert" in s["id"])
        explain_payload = {
            "dataset": alert_sample["dataset"],
            "sequence": alert_sample["sequence"],
            "steps": 25,
            "top_k": 5,
        }
        code, exp = http_post("http://127.0.0.1:8000/explain", explain_payload)
        print(f"\n[5] POST /explain -> HTTP {code}")
        print(f"    Method: {exp.get('method')} | Steps: {exp.get('steps')}")
        print(f"    Completeness Residual: {exp.get('completeness_residual'):.6f} | Latency: {exp.get('explanation_time_ms'):.2f}ms")
        print(f"    Top Positive Drivers Count: {len(exp.get('top_positive_drivers', []))}")
        print(f"    Top Negative Drivers Count: {len(exp.get('top_negative_drivers', []))}")
        assert code == 200 and exp.get("completeness_residual") < 0.05

        # 7. Test Error Handling on POST /predict
        bad_payload = {
            "dataset": "HI-Small",
            "sequence": {
                "features": [[0.1] * 56] + [[0.0] * 56] * 9,
                "length": 1,
                "sample_id": "bad_dim",
            },
        }
        code, err = http_post("http://127.0.0.1:8000/predict", bad_payload)
        print(f"\n[6] POST /predict (Malformed 56-dim input) -> HTTP {code}")
        print(f"    Rejection validated. Detail: {str(err.get('detail'))[:100]}...")
        assert code == 422

        print("\n" + "=" * 70)
        print("ALL LIVE HTTP SOCKET ENDPOINTS VERIFIED SUCCESSFULLY (PASS)")
        print("=" * 70)

    finally:
        print("\nShutting down backend process...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Backend process cleanly terminated.")


if __name__ == "__main__":
    main()
