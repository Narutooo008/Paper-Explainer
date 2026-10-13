# PaperExplainer

Upload a research paper PDF (or paste an arXiv link/ID) and get an animated, narrated explainer video with diagrams, extracted PDF figures, subtitles, downloadable Markdown notes, style regeneration, and a paper Q&A chat.

**How it works:** PDF text & figures (PyMuPDF) → NVIDIA LLM writes a structured JSON script with visual diagrams → Pillow renders 24 fps animated scenes (`intro`, `flow`, `compare`, `bar_chart`, `cycle`, `figure`, `outro`) → `edge-tts` records narration → FFmpeg streams raw frames + audio into `video.mp4`.

## Features

- **PDF Upload & arXiv Support:** Drag-and-drop any PDF (up to 25 MB) or enter an arXiv URL/ID (e.g. `arxiv.org/abs/1706.03762`).
- **24 fps Animated Explainer Video:** Smooth ease-out animations with sliding titles, timed bullet reveals, yellow active-element highlights, and auto-laid-out diagrams (`flow`, `compare`, `bar_chart`, `cycle`), plus an animated **Intro** (paper title & overview) and **Outro** (3 key takeaways).
- **PDF Figure Extraction:** Automatically pulls the best 1–2 figures from the paper PDF with PyMuPDF and displays them on the matching scenes.
- **Voice, Speed & Style Options:** Choose from US, UK, and Indian English `edge-tts` voices, adjust playback speed (`0.85x`–`1.25x`), and regenerate any paper in **Simpler**, **More technical**, or **Very short** (60s) style.
- **Subtitles & Downloadable Notes:** Built-in `<track>` video subtitles, `.srt` download, and a `.md` study guide download (title, summary, scene explanations, and glossary).
- **Chat with the Paper:** Ask follow-up questions below the video; answers are grounded strictly in the extracted paper text.
- **History & Dark Mode:** Browse or delete previous papers with first-slide thumbnails, and toggle light/dark mode (saved in `localStorage`).

## 1. Install

You need Python 3.10+ and FFmpeg.

- **FFmpeg:** Windows `winget install ffmpeg` · Mac `brew install ffmpeg` · Linux `sudo apt install ffmpeg`
- Check it with `ffmpeg -version` (restart your terminal after installing).

```bash
cd paper-explainer
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure `.env`

1. Sign up at https://build.nvidia.com and create a key under Settings → API Keys (it starts with `nvapi-`).
2. Copy `.env.example` to `.env` and paste your key:

```env
NVIDIA_API_KEY=nvapi-your-key
NVIDIA_MODEL=openai/gpt-oss-20b
NVIDIA_FAST_MODEL=meta/llama-3.1-8b-instruct
USE_ANIMATION=true
QUALITY=normal
```
- `NVIDIA_FAST_MODEL`: Fast model used to summarize long paper chunks in parallel (`ThreadPoolExecutor`, 4 workers, rate-limited under 40 RPM), while `NVIDIA_MODEL` writes the final JSON script.
- `QUALITY`: `normal` (`1280x720` at `24 fps`) or `fast` (`960x540` at `15 fps` for rapid rendering on long papers).
- `USE_ANIMATION`: Set to `false` to fall back to static slides.

## 3. Run

```bash
uvicorn backend.main:app --reload
```

Open http://localhost:8000, drop in a PDF or paste an arXiv ID, and wait 2 to 4 minutes. Videos and artifacts are saved in `jobs/<id>/`.

## 4. Tests

Run the unit test suite (JSON cleanup, visual validation, diagram non-overlap & bounds, and frame rendering):

```bash
python backend/test_unit.py
```

Render a standalone sample animation video (`test_animation.mp4`) covering all visual types without calling any API:

```bash
python backend/test_animate.py
```

Run the production SEO, metadata, canonical, and header tests:

```bash
python backend/test_production.py
```

## 5. Production Deployment & SEO Configuration

In production, PaperExplainer generates absolute canonical URLs, Open Graph image URLs, XML sitemaps, and robots.txt directives dynamically using your base URL. Never hard-code domains in the codebase.

### Environment Configuration

Configure the following variables in `.env`:

```env
# Base URL for public links, sitemap, canonical tags, and Open Graph cards (no trailing slash)
SITE_URL=https://paperexplainer.org

# Brand name displayed in metadata and titles
SITE_NAME=PaperExplainer

# Public contact address in Schema.org and copy
CONTACT_EMAIL=contact@paperexplainer.org
```

### Reverse-Proxy Setup

Run PaperExplainer behind a reverse proxy (such as Nginx or Caddy) to terminate HTTPS, compress static assets, forward host headers, and 301-redirect `www` to the bare domain.

#### Nginx Configuration

```nginx
# 1. Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name paperexplainer.org www.paperexplainer.org;
    return 301 https://paperexplainer.org$request_uri;
}

# 2. Redirect www.paperexplainer.org to bare domain
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name www.paperexplainer.org;

    ssl_certificate /etc/letsencrypt/live/paperexplainer.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/paperexplainer.org/privkey.pem;

    return 301 https://paperexplainer.org$request_uri;
}

# 3. Main server block
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name paperexplainer.org;

    ssl_certificate /etc/letsencrypt/live/paperexplainer.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/paperexplainer.org/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Gzip & Brotli compression
    gzip on;
    gzip_vary on;
    gzip_min_length 256;
    gzip_proxied any;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml
        image/svg+xml;

    # Upload limit for PDFs
    client_max_body_size 30M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket / long-running connection support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;
    }
}
```

#### Caddy Configuration

```caddy
# Redirect www to bare domain
www.paperexplainer.org {
    redir https://paperexplainer.org{uri} permanent
}

# Main application block with automatic HTTPS & compression
paperexplainer.org {
    encode zstd gzip

    request_body {
        max_size 30MB
    }

    reverse_proxy 127.0.0.1:8000 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
}
```

## Files

| File | Job |
|---|---|
| `backend/main.py` | FastAPI server: upload, arXiv fetch, history, regenerate, paper chat, serving `/media` |
| `backend/pipeline.py` | Orchestrates PDF text & figure extraction, script generation, intro/outro, subtitles, and notes |
| `backend/llm.py` | NVIDIA API calls, style presets, script & diagram prompt, JSON/visual cleanup, paper Q&A |
| `backend/animate.py` | Renders 24 fps animated 1280×720 scenes and 2× supersampled diagrams with Pillow |
| `backend/slides.py` | Static 1280×720 slide renderer (used when `USE_ANIMATION=false`) |
| `backend/video.py` | `edge-tts` narration synthesis and FFmpeg rawvideo streaming + clip concatenation |
| `backend/test_unit.py` | Unit tests for JSON cleanup, diagram layout bounds, and scene rendering |
| `backend/test_animate.py` | Offline animation test script producing `test_animation.mp4` |
| `frontend/index.html` | Single-page vanilla JS interface with dark mode, history, subtitles, and paper chat |
