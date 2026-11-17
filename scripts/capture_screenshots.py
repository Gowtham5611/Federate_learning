import argparse
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

VIEWPORTS = {
    "1920px": {"width": 1920, "height": 1080},
    "1440px": {"width": 1440, "height": 900},
    "1024px": {"width": 1024, "height": 768},
    "390px": {"width": 390, "height": 844},
}

TAB_SELECTORS = {
    "evaluation": "nav button:has-text('Evaluation'), nav button:has-text('Evaluate')",
    "analytics": "nav button:has-text('Analytics')",
    "models": "nav button:has-text('Model'), nav button:has-text('Models')",
    "ensemble": "nav button:has-text('Ensemble')",
    "history": "nav button:has-text('History')",
}

def capture_suite(out_dir: str, tabs=("models", "history", "evaluation"), themes=("dark", "light")):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        
        for theme in themes:
            for vp_name, vp in VIEWPORTS.items():
                context = browser.new_context(viewport=vp)
                page = context.new_page()
                
                # Navigate and set theme cleanly via localStorage
                page.goto("http://127.0.0.1:5173", wait_until="domcontentloaded")
                time.sleep(1.0)
                page.evaluate(f"""() => {{
                    localStorage.setItem('fedsecureai_theme', '{theme}');
                    if ('{theme}' === 'light') {{
                        document.documentElement.classList.add('light');
                        document.documentElement.classList.remove('dark');
                    }} else {{
                        document.documentElement.classList.add('dark');
                        document.documentElement.classList.remove('light');
                    }}
                }}""")
                time.sleep(0.5)
                
                for tab in tabs:
                    selector = TAB_SELECTORS.get(tab)
                    if selector:
                        # Find visible tab button
                        tab_btn = page.locator(selector).first
                        if tab_btn.is_visible():
                            tab_btn.click()
                            time.sleep(0.6)
                    
                    filename = f"{tab}_{vp_name}_{theme}.png"
                    filepath = out_path / filename
                    page.screenshot(path=str(filepath), full_page=False)
                    print(f"Captured: {filepath}", flush=True)
                
                context.close()
        
        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="screenshots/group1_verify")
    parser.add_argument("--tabs", nargs="+", default=["models", "history", "evaluation"])
    parser.add_argument("--themes", nargs="+", default=["dark", "light"])
    args = parser.parse_args()
    capture_suite(args.dir, tabs=args.tabs, themes=args.themes)
