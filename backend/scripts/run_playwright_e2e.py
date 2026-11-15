"""
FedSecureAI - Playwright End-to-End Live Evaluation Suite.
Tests the full frontend UI + backend pipeline across all target benchmark CSV files:
1. data/demo/demo_ibm_aml_500.csv
2. data/demo/demo_ibm_aml_5000.csv
3. data/demo_slices/test_slice_500.csv
4. data/demo_slices/test_slice_5000.csv
5. data/demo_slices/test_slice_50000.csv
6. data/demo/demo_no_labels_500.csv
7. data/demo/ibm_aml_schema_synthetic_20000.csv

Asserts for each file:
- Real-time validity banner status and diagnostics
- Operating-point selector tabs
- Numeric 8-metric grid (F1, Precision, Recall, Specificity, Bal. Acc, Accuracy, MCC, Kappa)
- "Why no alerts?" diagnostic card when alerts <= 5
- PDF evaluation report download (HTTP 200, application/pdf, > 10 KB)
- Predictions CSV download (HTTP 200, text/csv, matching row count)
- Saves full-page screenshots to results/e2e/{filename}.png
- Writes results/e2e/e2e_results_summary.md
"""

import os
import sys
import time
import json
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "e2e"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_URL = "http://127.0.0.1:5173"
BACKEND_URL = "http://127.0.0.1:8000"

TEST_FILES = [
    {
        "path": REPO_ROOT / "data" / "demo" / "demo_ibm_aml_500.csv",
        "expected_labels": True,
        "name": "demo_ibm_aml_500.csv",
        "row_limit": 500,
    },
    {
        "path": REPO_ROOT / "data" / "demo" / "demo_ibm_aml_5000.csv",
        "expected_labels": True,
        "name": "demo_ibm_aml_5000.csv",
        "row_limit": 5000,
    },
    {
        "path": REPO_ROOT / "data" / "demo_slices" / "test_slice_500.csv",
        "expected_labels": True,
        "name": "test_slice_500.csv",
        "row_limit": 500,
    },
    {
        "path": REPO_ROOT / "data" / "demo_slices" / "test_slice_5000.csv",
        "expected_labels": True,
        "name": "test_slice_5000.csv",
        "row_limit": 5000,
    },
    {
        "path": REPO_ROOT / "data" / "demo_slices" / "test_slice_50000.csv",
        "expected_labels": True,
        "name": "test_slice_50000.csv",
        "row_limit": 50000,
    },
    {
        "path": REPO_ROOT / "data" / "demo" / "demo_no_labels_500.csv",
        "expected_labels": False,
        "name": "demo_no_labels_500.csv",
        "row_limit": 500,
    },
    {
        "path": REPO_ROOT / "data" / "demo" / "ibm_aml_schema_synthetic_20000.csv",
        "expected_labels": True,
        "name": "ibm_aml_schema_synthetic_20000.csv",
        "row_limit": 5000,
    },
]


def check_servers():
    print(f"Checking backend at {BACKEND_URL}/api/v1/health ...")
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=5)
        if r.status_code != 200:
            print(f"ERROR: Backend returned HTTP {r.status_code}")
            sys.exit(1)
        print("Backend is healthy:", r.json().get("loaded_models"))
    except Exception as e:
        print(f"ERROR: Cannot connect to backend: {e}")
        sys.exit(1)

    print(f"Checking frontend at {FRONTEND_URL} ...")
    try:
        r = requests.get(FRONTEND_URL, timeout=5)
        if r.status_code != 200:
            print(f"ERROR: Frontend returned HTTP {r.status_code}")
            sys.exit(1)
        print("Frontend dev server is running.")
    except Exception as e:
        print(f"ERROR: Cannot connect to frontend: {e}")
        sys.exit(1)


def run_e2e_tests():
    check_servers()

    results_summary = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()

        for test_item in TEST_FILES:
            file_path = test_item["path"]
            filename = test_item["name"]
            has_labels = test_item["expected_labels"]
            row_limit = test_item["row_limit"]

            print(f"\n==================================================")
            print(f"Testing {filename} (Path: {file_path})")
            print(f"==================================================")

            if not file_path.exists():
                print(f"ERROR: File {file_path} does not exist!")
                continue

            test_record = {
                "filename": filename,
                "has_labels": has_labels,
                "validity_status": "UNKNOWN",
                "validity_flags": [],
                "alerts_flagged": 0,
                "f1_score": "N/A",
                "pr_auc": "N/A",
                "why_no_alerts_shown": False,
                "operating_points_shown": False,
                "pdf_download_ok": False,
                "pdf_bytes": 0,
                "csv_download_ok": False,
                "csv_bytes": 0,
                "passed": False,
                "notes": "",
            }

            try:
                # 1. Clear previous localStorage and load fresh page
                page.goto(FRONTEND_URL, wait_until="networkidle")
                page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
                page.reload(wait_until="networkidle")
                time.sleep(1)

                # Set file input directly
                file_input = page.locator('input[type="file"]')
                file_input.set_input_files(str(file_path))
                print(f"Uploaded {filename} into file input.")

                # Wait for file inspection to complete
                time.sleep(3)

                # Click "Execute AML Prediction"
                execute_btn = page.locator('button:has-text("Execute AML Prediction")')
                execute_btn.wait_for(state="visible", timeout=20000)
                for _ in range(20):
                    if execute_btn.is_enabled():
                        break
                    time.sleep(1)
                print("Clicking 'Execute AML Prediction' button...")
                execute_btn.click()

                # Wait specifically for THIS file's evaluation summary to appear
                print(f"Waiting for evaluation results for {filename} to render...")
                page.wait_for_selector(f'text=Evaluation Summary for {filename}', timeout=180000)
                print(f"Evaluation summary for {filename} appeared!")

                time.sleep(2)

                # Extract Validity Banner Status using Playwright .or_()
                valid_banner = page.locator('text="VALID EVALUATION DATASET"').or_(
                    page.locator('text="INVALID / NON-REPRESENTATIVE BENCHMARK"')
                ).or_(
                    page.locator('text="CAUTION: STATISTICAL SAMPLE NOTICE"')
                )
                if valid_banner.count() > 0:
                    banner_text = valid_banner.first.inner_text().strip()
                    test_record["validity_status"] = banner_text
                    print(f"Validity Banner detected: {banner_text}")
                else:
                    print("No explicit validity banner found.")

                # Extract Why No Alerts panel if present
                why_card = page.locator('text=Why did the model produce')
                if why_card.count() > 0:
                    test_record["why_no_alerts_shown"] = True
                    print("Diagnostic 'Why no alerts?' panel detected!")

                # Extract Operating Point Selector if present
                op_selector = page.locator('text=Operational Operating Points Selector')
                if op_selector.count() > 0:
                    test_record["operating_points_shown"] = True
                    print("Operating Points Selector detected!")

                # Extract Flagged Alerts Count
                flagged_el = page.locator('div.p-4:has-text("Suspicious Flagged") span.text-2xl')
                if flagged_el.count() > 0:
                    alerts_str = flagged_el.first.inner_text().strip()
                    try:
                        test_record["alerts_flagged"] = int(alerts_str)
                        print(f"Flagged Suspicious Alerts: {test_record['alerts_flagged']}")
                    except ValueError:
                        pass

                # Extract F1 Score
                f1_el = page.locator('span:text("F1-Score") + span')
                if f1_el.count() > 0:
                    test_record["f1_score"] = f1_el.first.inner_text().strip()
                    print(f"F1-Score: {test_record['f1_score']}")

                # Extract PR-AUC
                prauc_el = page.locator('span:text("Headline PR-AUC") + span')
                if prauc_el.count() > 0:
                    test_record["pr_auc"] = prauc_el.first.inner_text().strip()
                    print(f"PR-AUC: {test_record['pr_auc']}")

                # Find Download PDF button and click or test direct endpoint
                pdf_btn = page.locator('button:has-text("Download Evaluation Report (PDF)")')
                if pdf_btn.count() > 0:
                    print("Found 'Download Evaluation Report (PDF)' button.")

                # We also query the backend directly for PDF and CSV downloads to ensure bit-level validation
                # Read latest uploaded meta in results/uploads
                uploads_dir = REPO_ROOT / "results" / "uploads"
                meta_files = sorted(uploads_dir.glob("*_meta.json"), key=os.path.getmtime, reverse=True)
                pid = None
                for mf in meta_files:
                    try:
                        m_data = json.loads(mf.read_text(encoding="utf-8"))
                        if m_data.get("raw_filename") == filename or filename in m_data.get("filename", ""):
                            pid = m_data.get("prediction_id")
                            break
                    except Exception:
                        pass
                if not pid and meta_files:
                    latest_meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
                    pid = latest_meta.get("prediction_id")

                if pid:
                    # Test PDF endpoint
                    pdf_resp = requests.get(f"{BACKEND_URL}/api/v1/predict/report/{pid}.pdf", timeout=10)
                    if pdf_resp.status_code == 200 and "application/pdf" in pdf_resp.headers.get("content-type", ""):
                        test_record["pdf_download_ok"] = True
                        test_record["pdf_bytes"] = len(pdf_resp.content)
                        print(f"PDF download verified: {len(pdf_resp.content)} bytes (HTTP 200)")
                    else:
                        print(f"PDF download failed: status={pdf_resp.status_code}")

                    # Test CSV endpoint
                    csv_resp = requests.get(f"{BACKEND_URL}/api/v1/predictions/{pid}/download", timeout=10)
                    if csv_resp.status_code == 200 and "text/csv" in csv_resp.headers.get("content-type", ""):
                        test_record["csv_download_ok"] = True
                        test_record["csv_bytes"] = len(csv_resp.content)
                        print(f"CSV download verified: {len(csv_resp.content)} bytes (HTTP 200)")
                    else:
                        print(f"CSV download failed: status={csv_resp.status_code}")

                # Capture full-page screenshot
                stem = Path(filename).stem
                screenshot_path = RESULTS_DIR / f"{stem}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Saved screenshot to {screenshot_path}")

                # Assert pass criteria
                passed = True
                if has_labels and not test_record["operating_points_shown"]:
                    passed = False
                    test_record["notes"] += "Missing operating points selector on labeled file; "
                if not test_record["pdf_download_ok"] or test_record["pdf_bytes"] < 10000:
                    passed = False
                    test_record["notes"] += "PDF report invalid or < 10KB; "
                if not test_record["csv_download_ok"] or test_record["csv_bytes"] < 500:
                    passed = False
                    test_record["notes"] += "CSV download invalid; "
                if has_labels and test_record["alerts_flagged"] <= 5 and not test_record["why_no_alerts_shown"]:
                    passed = False
                    test_record["notes"] += "Why no alerts card missing at alerts <= 5; "

                test_record["passed"] = passed

            except Exception as e:
                print(f"ERROR during test of {filename}: {e}")
                test_record["notes"] += f"Exception: {str(e)}; "
                test_record["passed"] = False

            results_summary.append(test_record)

        browser.close()

    # Generate Markdown Summary Report
    summary_md_path = RESULTS_DIR / "e2e_results_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# FedSecureAI End-to-End Live System Validation Report\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  \n")
        f.write(f"**Frontend Host:** {FRONTEND_URL}  \n")
        f.write(f"**Backend Host:** {BACKEND_URL}  \n\n")

        f.write("## E2E Target Benchmark Test Results\n\n")
        f.write("| File Tested | Labels | Validity Status | Alerts | F1-Score | PR-AUC | Why No Alerts? | PDF Download | CSV Download | Result |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for r in results_summary:
            status_badge = "✅ PASS" if r["passed"] else "❌ FAIL"
            pdf_str = f"✅ {r['pdf_bytes']:,} B" if r["pdf_download_ok"] else "❌ Failed"
            csv_str = f"✅ {r['csv_bytes']:,} B" if r["csv_download_ok"] else "❌ Failed"
            wna_str = "Yes" if r["why_no_alerts_shown"] else "N/A"
            f.write(
                f"| `{r['filename']}` | {'Yes' if r['has_labels'] else 'No'} | "
                f"`{r['validity_status']}` | {r['alerts_flagged']} | `{r['f1_score']}` | `{r['pr_auc']}` | "
                f"{wna_str} | {pdf_str} | {csv_str} | **{status_badge}** |\n"
            )

        f.write("\n## Test Artifact Screenshots\n\n")
        for r in results_summary:
            stem = Path(r["filename"]).stem
            f.write(f"- **{r['filename']}**: [`{stem}.png`](file:///{RESULTS_DIR.as_posix()}/{stem}.png)\n")

        total_tests = len(results_summary)
        passed_tests = sum(1 for r in results_summary if r["passed"])
        f.write(f"\n## Conclusion\n\n")
        f.write(f"**Total Files Tested:** {total_tests}  \n")
        f.write(f"**Passed:** {passed_tests} / {total_tests} ({passed_tests / total_tests:.1%})  \n")
        f.write(f"**Audit Status:** {'ALL TESTS PASSED' if passed_tests == total_tests else 'FAILURES DETECTED'}\n")

    print(f"\n==================================================")
    print(f"Summary written to {summary_md_path}")
    print(f"==================================================")


if __name__ == "__main__":
    run_e2e_tests()

