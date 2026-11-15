import os
import sys
import json
import time
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "frontend" / "dist"
SCREENSHOTS_DIR = REPO_ROOT / "results" / "ui_screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the demo meta data
META_PATH = REPO_ROOT / "results" / "uploads" / "pred_1794301880_85a48325_meta.json"
assert META_PATH.exists(), f"Meta file not found: {META_PATH}"
with open(META_PATH, "r", encoding="utf-8") as f:
    demo_meta = json.load(f)

# Ensure necessary top-level fields for CsvEvaluateResponse
demo_result = {
    "prediction_id": demo_meta.get("prediction_id", "pred_1794301880_85a48325"),
    "filename": demo_meta.get("filename", "predictions_threshold_trigger_demo_only.csv"),
    "file_hash": demo_meta.get("file_hash", ""),
    "total_rows": demo_meta.get("row_count", 20000),
    "evaluated_sequences": demo_meta.get("row_count", 20000),
    "flagged_laundering_count": 2016,
    "cleared_count": 17984,
    "execution_time_ms": 1140,
    "model_name": demo_meta.get("model_name", "hi_small_lstm_fedavg_iid_seed42"),
    "model_id": demo_meta.get("model_id", "HI-Small"),
    "dataset": demo_meta.get("dataset", "HI-Small"),
    "decision_threshold": demo_meta.get("decision_threshold", 0.337784),
    "has_labels": True,
    "has_ground_truth": True,
    "evaluation_metrics": demo_meta.get("evaluation_metrics", {}),
    "operating_points": demo_meta.get("operating_points", {}),
    "validity_banner": {
        "status": "VALID",
        "title": "Universal Transaction Schema Validated",
        "message": "Successfully preprocessed 20,000 transactions across 14 banks and 4 currency formats.",
        "dataset_identified": "HI-Small",
        "row_count": 20000,
        "is_compatible": True,
    },
    "predictions": [],
}

class DistHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST_DIR), **kwargs)

    def log_message(self, format, *args):
        pass # Suppress logging

def start_static_server(port=5188):
    server = ThreadingHTTPServer(("127.0.0.1", port), DistHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

VIEWPORTS = [
    {"name": "1920_desktop_wide", "width": 1920, "height": 1080},
    {"name": "1440_desktop_standard", "width": 1440, "height": 900},
    {"name": "1280_small_desktop", "width": 1280, "height": 800},
    {"name": "1024_tablet_landscape", "width": 1024, "height": 768},
    {"name": "768_tablet_portrait", "width": 768, "height": 1024},
    {"name": "390_mobile", "width": 390, "height": 844},
]

def run_verification():
    port = 5188
    server = start_static_server(port)
    base_url = f"http://127.0.0.1:{port}"
    print(f"Static server started at {base_url}", flush=True)
    
    results_summary = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for vp in VIEWPORTS:
            w, h = vp["width"], vp["height"]
            vp_name = vp["name"]
            print(f"\n==================================================", flush=True)
            print(f"TESTING VIEWPORT: {vp_name} ({w}x{h})", flush=True)
            print(f"==================================================", flush=True)

            context = browser.new_context(viewport={"width": w, "height": h})

            # Mock backend API responses so page never hangs
            context.route(
                "**/api/**",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({
                        "status": "healthy",
                        "readiness": "ready",
                        "models": [],
                    }),
                ),
            )
            context.route(
                "http://127.0.0.1:8000/**",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({
                        "status": "healthy",
                        "readiness": "ready",
                        "models": [],
                    }),
                ),
            )

            page = context.new_page()

            # Pre-inject the demo evaluation result into localStorage
            page.goto(base_url, wait_until="domcontentloaded")
            page.evaluate(
                """(data) => {
                    localStorage.setItem('fedsecureai_current_result', JSON.stringify(data));
                }""",
                demo_result,
            )
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("header", timeout=5000)
            time.sleep(0.5)

            # -----------------------------------------------------------------
            # 1. AML Evaluation Page Checks
            # -----------------------------------------------------------------
            scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
            client_width = page.evaluate("() => document.documentElement.clientWidth")
            has_overflow = scroll_width > client_width
            print(f"[AML Evaluation] ScrollWidth: {scroll_width}, ClientWidth: {client_width}, Overflow: {has_overflow}", flush=True)
            assert not has_overflow, f"Horizontal page overflow detected on {vp_name}: scrollWidth={scroll_width} > clientWidth={client_width}"

            eval_screenshot_path = SCREENSHOTS_DIR / f"{vp_name}_01_aml_evaluation.png"
            page.screenshot(path=str(eval_screenshot_path), full_page=True)
            print(f"Captured: {eval_screenshot_path.name}", flush=True)

            # Check for Demo Disclaimer Banner
            banner_text = page.locator("text=Purpose-Built Synthetic Threshold-Trigger Demonstration").first
            assert banner_text.is_visible(), f"Demo Disclaimer Banner not visible on Evaluation Page at {vp_name}"
            print("✓ Demo Disclaimer Banner is visible on Evaluation Page", flush=True)

            # -----------------------------------------------------------------
            # 2. Results & Analytics Page Checks (Fixed Threshold Mode)
            # -----------------------------------------------------------------
            page.locator("button:has-text('Results & Analytics'), button:has-text('Analytics')").first.click()
            time.sleep(0.5)

            scroll_width_analytics = page.evaluate("() => document.documentElement.scrollWidth")
            client_width_analytics = page.evaluate("() => document.documentElement.clientWidth")
            has_overflow_analytics = scroll_width_analytics > client_width_analytics
            print(f"[Results & Analytics] ScrollWidth: {scroll_width_analytics}, ClientWidth: {client_width_analytics}, Overflow: {has_overflow_analytics}", flush=True)
            assert not has_overflow_analytics, f"Horizontal page overflow detected on Analytics {vp_name}"

            # Click Fixed Canonical Threshold
            page.locator("button:has-text('Fixed Canonical Threshold')").first.click()
            time.sleep(0.3)

            fixed_thr_screenshot = SCREENSHOTS_DIR / f"{vp_name}_02_results_fixed_threshold.png"
            page.screenshot(path=str(fixed_thr_screenshot), full_page=True)
            print(f"Captured: {fixed_thr_screenshot.name}", flush=True)

            # Validate Fixed Threshold numbers in the view:
            page.wait_for_selector("text=2,000")
            page.wait_for_selector("text=16")
            page.wait_for_selector("text=4,000")
            page.wait_for_selector("text=13,984")
            print("✓ Fixed Threshold Confusion Matrix verified: TP=2,000, FP=16, FN=4,000, TN=13,984", flush=True)

            # -----------------------------------------------------------------
            # 3. Results & Analytics Page Checks (Fixed Capacity Mode)
            # -----------------------------------------------------------------
            page.locator("button:has-text('Fixed Capacity: 50 alerts / 1k')").first.click()
            time.sleep(0.3)

            fixed_cap_screenshot = SCREENSHOTS_DIR / f"{vp_name}_03_results_fixed_capacity.png"
            page.screenshot(path=str(fixed_cap_screenshot), full_page=True)
            print(f"Captured: {fixed_cap_screenshot.name}", flush=True)

            # Validate Fixed Capacity numbers in the view:
            page.wait_for_selector("text=993")
            page.wait_for_selector("text=7")
            page.wait_for_selector("text=5,007")
            page.wait_for_selector("text=13,993")
            print("✓ Fixed Capacity Confusion Matrix verified: TP=993, FP=7, FN=5,007, TN=13,993", flush=True)

            results_summary.append({
                "viewport": vp_name,
                "width": w,
                "height": h,
                "overflow_evaluation": has_overflow,
                "overflow_analytics": has_overflow_analytics,
                "fixed_threshold_verified": True,
                "fixed_capacity_verified": True,
                "banner_verified": True,
            })

            context.close()

        browser.close()

    print("\n==================================================", flush=True)
    print("ALL VIEWPORT ACCEPTANCE CHECKS PASSED (100%)", flush=True)
    print("==================================================", flush=True)
    for r in results_summary:
        print(f"Viewport {r['viewport']} ({r['width']}x{r['height']}): Overflow=None, FixedThreshold=OK, FixedCapacity=OK, Banner=OK", flush=True)

if __name__ == "__main__":
    run_verification()
