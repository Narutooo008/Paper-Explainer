"""PaperExplainer server. Run from the project folder with:
    uvicorn backend.main:app --reload
"""
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets "import pipeline" work

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

import config
import llm
import pipeline

JOBS_DIR = ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
MAX_PDF_BYTES = 25 * 1024 * 1024

ALLOWED_VOICES = {
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
}
ALLOWED_RATES = {"-15%", "+0%", "+15%", "+25%"}
ALLOWED_STYLES = {"default", "simpler", "technical", "short"}
ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?\s*"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=llm.check_models_on_startup, daemon=True).start()
    yield


app = FastAPI(title="PaperExplainer", lifespan=_lifespan)
JOBS = {}                      # job_id -> status dict (in memory)
ONE_AT_A_TIME = threading.Semaphore(1)


def render_template(filename: str, replacements: dict[str, str] | None = None) -> str:
    template_path = ROOT / "frontend" / filename
    if not template_path.exists():
        raise HTTPException(404, "Page not found.")
    content = template_path.read_text(encoding="utf-8")
    mapping = {
        "{{SITE_URL}}": config.site_url(),
        "{{SITE_NAME}}": config.site_name(),
        "{{CONTACT_EMAIL}}": config.contact_email(),
    }
    if replacements:
        mapping.update(replacements)
    for key, val in mapping.items():
        content = content.replace(key, str(val))
    return content


@app.middleware("http")
async def seo_and_security_middleware(request: Request, call_next):
    path = request.url.path

    # 1. 301 redirect /index.html -> /
    if path == "/index.html":
        return RedirectResponse("/", status_code=301)

    # 2. Trailing slash normalization: redirect e.g. /about/ -> /about
    if path != "/" and path.endswith("/"):
        clean_path = path.rstrip("/")
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(f"{clean_path}{query}", status_code=301)

    response = await call_next(request)

    # 3. Private headers on private pages
    if (
        path.startswith("/api/")
        or path.startswith("/media/")
        or path.startswith("/p/")
        or path == "/history"
    ):
        response.headers["X-Robots-Tag"] = "noindex, nofollow"

    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=404, content={"detail": exc.detail or "Not Found"})
        try:
            html = render_template("404.html")
        except Exception:
            html = "<!DOCTYPE html><html><head><title>404 Not Found</title></head><body><h1>Page Not Found</h1></body></html>"
        return HTMLResponse(
            content=html,
            status_code=404,
            headers={"X-Robots-Tag": "noindex, nofollow"},
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})



class ArxivRequest(BaseModel):
    arxiv: str
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"
    style: str = "default"


class RegenerateRequest(BaseModel):
    style: str = "simpler"
    voice: str = ""
    rate: str = ""


class ChatRequest(BaseModel):
    question: str
    history: list = []


def _validate_options(voice: str, rate: str, style: str = "default") -> tuple[str, str, str]:
    default_voice = os.getenv("EDGE_VOICE", "en-US-AriaNeural")
    voice = (voice or default_voice).strip()
    rate = (rate or "+0%").strip()
    style = (style or "default").strip().lower()
    if voice not in ALLOWED_VOICES and voice != default_voice:
        raise HTTPException(400, "Unsupported voice selection.")
    if rate not in ALLOWED_RATES:
        raise HTTPException(400, "Unsupported speed selection.")
    if style not in ALLOWED_STYLES:
        raise HTTPException(400, "Unsupported style selection.")
    return voice, rate, style


def worker(job_id: str, voice: str, rate: str, style: str = "default") -> None:
    job_dir = JOBS_DIR / job_id

    def report(step: str, pct: int) -> None:
        JOBS[job_id] = {"status": "running", "step": step, "progress": pct}

    report("Waiting in line", 1)
    with ONE_AT_A_TIME:
        try:
            result = pipeline.run(job_dir, job_id, report, voice=voice, rate=rate, style=style)
            JOBS[job_id] = {"status": "done", "result": result}
        except Exception as e:  # show a readable message in the browser
            traceback.print_exc()
            JOBS[job_id] = {"status": "error", "message": str(e) or type(e).__name__}


def _enqueue_pdf(
    data: bytes,
    background: BackgroundTasks,
    voice: str,
    rate: str,
    style: str = "default",
) -> dict:
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "That PDF is bigger than 25 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a valid PDF file.")

    default_voice = os.getenv("EDGE_VOICE", "en-US-AriaNeural")
    if voice == default_voice and rate == "+0%" and style == "default":
        job_id = hashlib.sha256(data).hexdigest()[:16]
    else:
        salt = f"|{voice}|{rate}|{style}".encode("utf-8")
        job_id = hashlib.sha256(data + salt).hexdigest()[:16]

    job_dir = JOBS_DIR / job_id

    res_path = job_dir / "result.json"
    if res_path.exists():
        try:
            cached = json.loads(res_path.read_text(encoding="utf-8"))
            same_anim = bool(cached.get("animated", False)) == pipeline.video.use_animation()
            same_qual = cached.get("quality", "normal") == config.quality_mode()
            if same_anim and same_qual:
                return {"job_id": job_id}
        except Exception:
            pass
    if JOBS.get(job_id, {}).get("status") == "running":
        return {"job_id": job_id}

    job_dir.mkdir(exist_ok=True)
    (job_dir / "paper.pdf").write_bytes(data)
    JOBS[job_id] = {"status": "running", "step": "Starting", "progress": 0}
    background.add_task(worker, job_id, voice, rate, style)
    return {"job_id": job_id}


@app.post("/api/upload")
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    voice: str = Form("en-US-AriaNeural"),
    rate: str = Form("+0%"),
    style: str = Form("default"),
):
    voice, rate, style = _validate_options(voice, rate, style)
    data = await file.read()
    return _enqueue_pdf(data, background, voice, rate, style)


@app.post("/api/arxiv")
def upload_arxiv(req: ArxivRequest, background: BackgroundTasks):
    voice, rate, style = _validate_options(req.voice, req.rate, req.style)
    raw = req.arxiv.strip()
    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]
    m = ARXIV_RE.search(raw)
    if not m:
        raise HTTPException(
            400,
            "Could not find a valid arXiv ID. Try a link like https://arxiv.org/abs/1706.03762 or an ID like 1706.03762.",
        )
    arxiv_id = m.group(1)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    request = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "PaperExplainer/1.0 (+https://arxiv.org)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as resp:
            data = resp.read(MAX_PDF_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(404, f"Paper '{arxiv_id}' was not found on arXiv.")
        raise HTTPException(502, f"arXiv returned HTTP {e.code} while downloading the PDF.")
    except Exception as e:
        raise HTTPException(502, f"Could not download PDF from arXiv: {e}")

    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "That arXiv PDF is bigger than 25 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(
            400,
            f"arXiv did not return a valid PDF for '{arxiv_id}'. Check the ID and try again.",
        )
    return _enqueue_pdf(data, background, voice, rate, style)


@app.post("/api/jobs/{job_id}/regenerate")
def regenerate_job(job_id: str, req: RegenerateRequest, background: BackgroundTasks):
    if not job_id.isalnum():
        raise HTTPException(400, "Bad job id.")
    src_dir = JOBS_DIR / job_id
    pdf_path = src_dir / "paper.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Original PDF not found for this job.")

    prev_voice = os.getenv("EDGE_VOICE", "en-US-AriaNeural")
    prev_rate = "+0%"
    res_file = src_dir / "result.json"
    if res_file.exists():
        try:
            prev = json.loads(res_file.read_text(encoding="utf-8"))
            prev_voice = prev.get("voice") or prev_voice
            prev_rate = prev.get("rate") or prev_rate
        except Exception:
            pass

    voice, rate, style = _validate_options(
        req.voice or prev_voice, req.rate or prev_rate, req.style
    )
    data = pdf_path.read_bytes()
    out = _enqueue_pdf(data, background, voice, rate, style)
    dst_dir = JOBS_DIR / out["job_id"]
    if dst_dir != src_dir and (src_dir / "cache").exists():
        dst_cache = dst_dir / "cache"
        dst_cache.mkdir(parents=True, exist_ok=True)
        for cname in ("raw_paper.txt", "condensed.txt"):
            cfile = src_dir / "cache" / cname
            if cfile.exists() and not (dst_cache / cname).exists():
                shutil.copy2(cfile, dst_cache / cname)
    return out


@app.get("/api/jobs")
def list_jobs():
    items = []
    for d in JOBS_DIR.iterdir():
        if not d.is_dir() or not d.name.isalnum():
            continue
        res_file = d / "result.json"
        if not res_file.exists():
            continue
        try:
            data = json.loads(res_file.read_text(encoding="utf-8"))
            mtime = res_file.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone()
            thumb_path = d / "slides" / "slide_00.png"
            items.append(
                {
                    "job_id": d.name,
                    "title": data.get("title", "Paper explained"),
                    "summary": data.get("summary", ""),
                    "date": dt.strftime("%b %d, %Y"),
                    "timestamp": mtime,
                    "thumbnail": f"/media/{d.name}/slides/slide_00.png" if thumb_path.exists() else None,
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"jobs": items}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "Bad job id.")
    job_dir = JOBS_DIR / job_id
    if job_id in JOBS:
        state = JOBS[job_id]
        if state.get("status") == "done" and "result" in state:
            pipeline.ensure_extras(job_dir, job_id, state["result"])
        return state
    result_file = job_dir / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        pipeline.ensure_extras(job_dir, job_id, result)
        return {"status": "done", "result": result}
    raise HTTPException(404, "Unknown job.")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "Bad job id.")
    if JOBS.get(job_id, {}).get("status") == "running":
        raise HTTPException(409, "Cannot delete a job while it is still running.")
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Unknown job.")
    shutil.rmtree(job_dir, ignore_errors=True)
    JOBS.pop(job_id, None)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/chat")
def chat_with_paper(job_id: str, req: ChatRequest):
    if not job_id.isalnum():
        raise HTTPException(400, "Bad job id.")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    if len(question) > 1000:
        raise HTTPException(400, "Question is too long (max 1000 characters).")
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Unknown job.")
    try:
        paper_text = pipeline.get_paper_text(job_dir)
        answer = llm.answer_question(paper_text, question, req.history)
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, str(e) or "Could not answer that question right now.")


# ---------------- Public & Private Web Routes ----------------

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(render_template("index.html"))


@app.get("/how-it-works", response_class=HTMLResponse)
def how_it_works_page():
    return HTMLResponse(render_template("how-it-works.html"))


@app.get("/about", response_class=HTMLResponse)
def about_page():
    return HTMLResponse(render_template("about.html"))


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return HTMLResponse(render_template("privacy.html"))


@app.get("/history", response_class=HTMLResponse)
def history_page():
    return HTMLResponse(
        render_template("index.html"),
        headers={"X-Robots-Tag": "noindex, nofollow"},
    )


@app.get("/p/{job_id}", response_class=HTMLResponse)
def view_job_result(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(404, "Paper explainer not found.")
    job_dir = JOBS_DIR / job_id
    result_file = job_dir / "result.json"
    if not job_dir.exists() or not result_file.exists():
        raise HTTPException(404, "Paper explainer not found.")
    try:
        res_data = json.loads(result_file.read_text(encoding="utf-8"))
        title = res_data.get("title", "Paper Explained")
        summary = res_data.get("summary", "")
    except Exception:
        title = "Paper Explained"
        summary = ""
    html = render_template(
        "p.html",
        {
            "{{TITLE}}": title,
            "{{SUMMARY}}": summary,
            "{{JOB_ID}}": job_id,
        },
    )
    return HTMLResponse(
        content=html,
        headers={"X-Robots-Tag": "noindex, nofollow"},
    )


# ---------------- SEO & Discovery Endpoints ----------------

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /how-it-works\n"
        "Allow: /about\n"
        "Allow: /privacy\n"
        "Allow: /assets/\n"
        "Allow: /site.webmanifest\n"
        "Allow: /llms.txt\n"
        "Disallow: /api/\n"
        "Disallow: /media/\n"
        "Disallow: /p/\n"
        "Disallow: /history\n\n"
        f"Sitemap: {config.site_url()}/sitemap.xml\n"
    )
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml")
def sitemap_xml():
    base_url = config.site_url()
    pages = [
        ("/", "index.html", "1.0", "daily"),
        ("/how-it-works", "how-it-works.html", "0.8", "weekly"),
        ("/about", "about.html", "0.8", "monthly"),
        ("/privacy", "privacy.html", "0.5", "monthly"),
    ]
    urls_xml = []
    for path, filename, priority, changefreq in pages:
        filepath = ROOT / "frontend" / filename
        if filepath.exists():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
            lastmod = mtime.strftime("%Y-%m-%d")
        else:
            lastmod = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        loc = f"{base_url}{path}" if path != "/" else f"{base_url}/"
        urls_xml.append(
            f"  <url>\n"
            f"    <loc>{loc}</loc>\n"
            f"    <lastmod>{lastmod}</lastmod>\n"
            f"    <changefreq>{changefreq}</changefreq>\n"
            f"    <priority>{priority}</priority>\n"
            f"  </url>"
        )
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls_xml)
        + "\n</urlset>"
    )
    return Response(content=xml_content, media_type="application/xml")


@app.get("/llms.txt", response_class=PlainTextResponse)
def llms_txt():
    content = render_template("llms.txt")
    return PlainTextResponse(content, media_type="text/markdown; charset=utf-8")


@app.get("/site.webmanifest")
def site_webmanifest():
    manifest_path = ROOT / "frontend" / "site.webmanifest"
    if manifest_path.exists():
        return Response(
            content=manifest_path.read_text(encoding="utf-8"),
            media_type="application/manifest+json",
        )
    raise HTTPException(404, "Manifest not found.")


@app.get("/favicon.ico")
def favicon_ico():
    fav_path = ROOT / "frontend" / "assets" / "favicon-32x32.png"
    if fav_path.exists():
        return Response(content=fav_path.read_bytes(), media_type="image/png")
    raise HTTPException(404, "Favicon not found.")


# Static assets
app.mount("/assets", StaticFiles(directory=ROOT / "frontend" / "assets"), name="assets")
app.mount("/demo", StaticFiles(directory=ROOT / "frontend" / "demo"), name="demo")
app.mount("/media", StaticFiles(directory=JOBS_DIR), name="media")


