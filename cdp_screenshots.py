import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT_DIR = r"C:\Users\Dell\Downloads\paper-explainer\paper-explainer\frontend\screenshots"
ARTIFACT_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\be9a6116-c476-4d89-bfcd-8f872ea12488"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


async def send_cdp(ws, method, params=None, msg_id=1):
    payload = {"id": msg_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(payload))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get("id") == msg_id:
            return resp.get("result", {})


async def capture_all():
    profile_dir = os.path.join(tempfile.gettempdir(), "cdp_chrome_run")
    shutil.rmtree(profile_dir, ignore_errors=True)
    os.makedirs(profile_dir, exist_ok=True)

    proc = subprocess.Popen([
        CHROME,
        "--headless=new",
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_dir}",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
    ])
    
    # Wait for debug port
    for _ in range(20):
        try:
            with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=1) as r:
                tabs = json.loads(r.read())
                page_tabs = [t for t in tabs if t.get("type") == "page"]
                if page_tabs:
                    break
        except Exception:
            pass
        time.sleep(0.5)

    try:
        ws_url = page_tabs[0]["webSocketDebuggerUrl"]
        print(f"Connecting to page tab: {ws_url}", flush=True)

        tasks = [
            # 1. Primary Light Theme Deliverables
            ("upload-1280.png", "http://127.0.0.1:8000/?theme=light", 1280, 900, False),
            ("upload-375.png", "http://127.0.0.1:8000/?theme=light", 375, 812, True),
            ("result-1280.png", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=light", 1280, 900, False),
            ("result-375.png", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=light", 375, 812, True),

            # 2. Dark Theme Deliverables (to confirm toggle and 4.5:1 contrast)
            ("upload-1280-dark.png", "http://127.0.0.1:8000/?theme=dark", 1280, 900, False),
            ("upload-375-dark.png", "http://127.0.0.1:8000/?theme=dark", 375, 812, True),
            ("result-1280-dark.png", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=dark", 1280, 900, False),
            ("result-375-dark.png", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=dark", 375, 812, True),
        ]

        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            await send_cdp(ws, "Page.enable", {}, 10)

            msg_counter = 20
            for filename, url, width, height, is_mobile in tasks:
                print(f"Capturing {filename} ({width}x{height}, mobile={is_mobile})...", flush=True)
                
                # Override metrics
                await send_cdp(ws, "Emulation.setDeviceMetricsOverride", {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": is_mobile,
                    "fitWindow": False
                }, msg_counter)
                msg_counter += 1

                # User agent
                ua = (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
                    if is_mobile else
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                await send_cdp(ws, "Emulation.setUserAgentOverride", {"userAgent": ua}, msg_counter)
                msg_counter += 1

                # Navigate
                await send_cdp(ws, "Page.navigate", {"url": url}, msg_counter)
                msg_counter += 1

                # Wait for render
                await asyncio.sleep(2.0)

                # Capture screenshot
                res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"}, msg_counter)
                msg_counter += 1
                
                img_bytes = base64.b64decode(res["data"])
                dest = os.path.join(OUT_DIR, filename)
                with open(dest, "wb") as f:
                    f.write(img_bytes)
                shutil.copy2(dest, os.path.join(ARTIFACT_DIR, filename))
                print(f"  Saved {filename} ({len(img_bytes)} bytes)", flush=True)

        print("All screenshots successfully captured!", flush=True)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(capture_all())
