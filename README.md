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
