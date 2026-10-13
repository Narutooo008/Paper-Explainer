import os
import shutil
import subprocess

out_dir = r"C:\Users\Dell\Downloads\paper-explainer\paper-explainer\frontend\screenshots"
artifact_dir = r"C:\Users\Dell\.gemini\antigravity\brain\be9a6116-c476-4d89-bfcd-8f872ea12488"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

tasks = [
    # 1. Light theme (Primary default requested by user)
    ("upload-1280-light.png", "1280,960", "http://127.0.0.1:8000/?theme=light"),
    ("upload-375-light.png", "375,812", "http://127.0.0.1:8000/?theme=light"),
    ("result-1280-light.png", "1280,960", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=light"),
    ("result-375-light.png", "375,812", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=light"),

    # Also capture canonical 1280 and 375 filenames
    ("upload-1280.png", "1280,960", "http://127.0.0.1:8000/?theme=light"),
    ("upload-375.png", "375,812", "http://127.0.0.1:8000/?theme=light"),
    ("result-1280.png", "1280,960", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=light"),
    ("result-375.png", "375,812", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=light"),

    # 2. Dark theme (to show the toggle / contrast verification)
    ("upload-1280-dark.png", "1280,960", "http://127.0.0.1:8000/?theme=dark"),
    ("result-1280-dark.png", "1280,960", "http://127.0.0.1:8000/?job=a612ed52b01bd0bf&theme=dark"),
]

for filename, size, url in tasks:
    dest = os.path.join(out_dir, filename)
    cmd = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        f"--screenshot={dest}",
        f"--window-size={size}",
        "--hide-scrollbars",
        url,
    ]
    print(f"Capturing {filename} ({size}) from {url}...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if os.path.exists(dest):
        print(f"  Success: {filename} ({os.path.getsize(dest)} bytes)")
        # Copy to artifact dir
        shutil.copy2(dest, os.path.join(artifact_dir, filename))
    else:
        print(f"  Failed: {filename}")
        print("  Stderr:", res.stderr)

print("All captures completed.")
