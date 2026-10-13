"""The full paper -> video pipeline."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time

import fitz  # PyMuPDF

import animate
import config
import llm
import slides
import video

MAX_INPUT_CHARS = 40_000    # cap total filtered input sent to summarization
MAX_DIRECT_CHARS = 20_000   # papers shorter than this go to the model in one piece
CHUNK_CHARS = 20_000        # larger chunks -> fewer LLM calls
MAX_CHUNKS = 4
SUMMARY_WORKERS = 4

# Headings to keep vs. skip when scanning line-by-line
_HEADING_PREFIX = r"^\s*(?:(?:[0-9]+(?:\.[0-9]+)*|[IVXLC]+|[A-Z])[\.\)\s]+)?\s*"
_KEEP_HEADING_RE = re.compile(
    _HEADING_PREFIX
    + r"(abstract|introduction|overview|motivation|preliminaries|problem\s+(?:statement|formulation|definition)"
    r"|method(?:ology|s)?|approach|proposed\s+\w+|architecture|model|framework|system\s+design|implementation"
    r"|experiment(?:s|al\s+setup|al\s+results)?|evaluation|results?(?:\s+and\s+discussion)?|empirical\s+\w+|analysis|ablation(?:\s+study)?"
    r"|discussion|limitations?(?:\s+and\s+future\s+work)?|conclusion(?:s)?|summary(?:\s+and\s+conclusion)?|future\s+work)\s*$",
    re.IGNORECASE,
)
_SKIP_HEADING_RE = re.compile(
    _HEADING_PREFIX
    + r"(related\s+work(?:s)?|prior\s+work|literature\s+review|background\s+and\s+related\s+work"
    r"|acknowledgm?ents?|funding|author\s+contributions?|competing\s+interests?|ethics\s+statement|broader\s+impact"
    r"|references|bibliography|works\s+cited"
    r"|appendix|appendices|supplementary(?:\s+material(?:s)?)?|supplemental(?:\s+material(?:s)?)?)\b.*$",
    re.IGNORECASE,
)


def extract_text(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    text = "\n".join(pages).strip()
    if len(text) < 500:
        raise ValueError(
            "I could not read text from this PDF. It may be a scanned image; "
            "try a PDF where you can select the text."
        )
    return text


def strip_references(text: str) -> str:
    """Cut the bibliography so it does not waste the model's attention."""
    cut = max(text.rfind("\nReferences"), text.rfind("\nREFERENCES"))
    return text[:cut] if cut > len(text) * 0.5 else text


def filter_sections(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Keep abstract, intro, method, results, discussion, conclusion; skip related work, appendix, references."""
    raw = strip_references(text)
    lines = raw.splitlines()
    kept_lines = []
    keeping = True  # keep title, abstract, and intro at the start of the paper

    for line in lines:
        stripped = line.strip()
        if 2 <= len(stripped) <= 85 and not stripped.endswith("."):
            if _SKIP_HEADING_RE.match(stripped):
                keeping = False
                continue
            if _KEEP_HEADING_RE.match(stripped):
                keeping = True
        if keeping:
            kept_lines.append(line)

    filtered = "\n".join(kept_lines).strip()
    # Fallback if unusual formatting caused too much text to be dropped
    if len(filtered) < min(1500, len(raw) // 4):
        filtered = raw

    if len(filtered) > max_chars:
        # Keep the first 75% (abstract, intro, method, main results) + last 25% (discussion, conclusion)
        head_len = int(max_chars * 0.75)
        tail_len = max_chars - head_len - 20
        filtered = filtered[:head_len] + "\n\n[...]\n\n" + filtered[-tail_len:]
    return filtered


def split_chunks(text: str, size: int) -> list:
    return [text[i : i + size] for i in range(0, len(text), size)]


def condense(text: str, report, cache_dir: Path | None = None) -> str:
    """Filter sections, cap at ~40k chars, and summarize 20k chunks in parallel (4 workers)."""
    filtered = filter_sections(text, max_chars=MAX_INPUT_CHARS)
    if len(filtered) <= MAX_DIRECT_CHARS:
        return filtered

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        condensed_file = cache_dir / "condensed.txt"
        if condensed_file.exists() and condensed_file.stat().st_size > 0:
            report("Loaded cached paper summary", 42)
            return condensed_file.read_text(encoding="utf-8")

    chunks = split_chunks(filtered, CHUNK_CHARS)[:MAX_CHUNKS]
    n = len(chunks)
    notes: list[str] = [""] * n
    done_count = 0

    def _summarize_idx(idx: int, chunk_text: str) -> tuple[int, str, bool]:
        if cache_dir is not None:
            chunk_file = cache_dir / f"chunk_{idx:02d}.txt"
            if chunk_file.exists() and chunk_file.stat().st_size > 0:
                return idx, chunk_file.read_text(encoding="utf-8"), True
        summary = llm.summarize_chunk(chunk_text)
        if cache_dir is not None:
            chunk_file = cache_dir / f"chunk_{idx:02d}.txt"
            chunk_file.write_text(summary, encoding="utf-8")
        return idx, summary, False

    report(f"Summarizing paper in {n} parts", 14)
    with ThreadPoolExecutor(max_workers=min(SUMMARY_WORKERS, n)) as pool:
        futures = [pool.submit(_summarize_idx, i, c) for i, c in enumerate(chunks)]
        for fut in as_completed(futures):
            idx, summary, was_cached = fut.result()
            notes[idx] = summary
            done_count += 1
            pct = 14 + int(28 * done_count / max(1, n))
            tag = " (cached)" if was_cached else ""
            report(f"Summarized part {done_count} of {n}{tag}", pct)

    combined = "\n\n".join(notes)
    if cache_dir is not None:
        (cache_dir / "condensed.txt").write_text(combined, encoding="utf-8")
    return combined



def _fmt_ts(seconds: float, sep: str = ",") -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _split_cues(narration: str) -> list:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", narration.strip()) if p.strip()]
    return parts or [narration.strip()]


def write_subtitles(scenes: list, job_dir: Path) -> None:
    srt_blocks = []
    vtt_blocks = ["WEBVTT\n"]
    idx = 1
    for scene in scenes:
        narration = str(scene.get("narration", "")).strip()
        if not narration:
            continue
        start = float(scene.get("start", 0.0))
        dur = float(scene.get("duration", 5.0))
        speech_dur = max(0.8, dur - video.PAD)
        cues = _split_cues(narration)
        total_chars = max(1, sum(len(c) for c in cues))
        t = start
        for i, cue in enumerate(cues):
            span = (
                (start + speech_dur - t)
                if i == len(cues) - 1
                else speech_dur * (len(cue) / total_chars)
            )
            t_end = max(t + 0.4, t + span)
            srt_blocks.append(f"{idx}\n{_fmt_ts(t, ',')} --> {_fmt_ts(t_end, ',')}\n{cue}\n")
            vtt_blocks.append(f"{idx}\n{_fmt_ts(t, '.')} --> {_fmt_ts(t_end, '.')}\n{cue}\n")
            idx += 1
            t = t_end

    (job_dir / "subtitles.srt").write_text("\n".join(srt_blocks), encoding="utf-8")
    (job_dir / "subtitles.vtt").write_text("\n".join(vtt_blocks), encoding="utf-8")


def write_notes(result: dict, job_dir: Path) -> None:
    lines = [
        f"# {result.get('title', 'Paper Explained')}",
        "",
        f"> {result.get('summary', '')}",
        "",
        "## Scene Breakdown",
        "",
    ]
    for i, s in enumerate(result.get("scenes", []), 1):
        lines.append(f"### {i}. {s.get('title', '')}")
        if s.get("source"):
            lines.append(f"*Source: {s.get('source')}*")
            lines.append("")
        for b in s.get("bullets", []):
            lines.append(f"- {b}")
        lines.append("")
        lines.append(s.get("narration", ""))
        lines.append("")

    glossary = result.get("glossary", [])
    if glossary:
        lines.append("## Glossary")
        lines.append("")
        for g in glossary:
            lines.append(f"- **{g.get('term', '')}**: {g.get('meaning', '')}")
        lines.append("")

    (job_dir / "notes.md").write_text("\n".join(lines), encoding="utf-8")


def ensure_extras(job_dir: Path, job_id: str, result: dict) -> dict:
    """Ensure subtitles and notes exist (including for jobs created before Phase 1)."""
    if not (job_dir / "subtitles.srt").exists() or not (job_dir / "subtitles.vtt").exists():
        write_subtitles(result.get("scenes", []), job_dir)
    if not (job_dir / "notes.md").exists():
        write_notes(result, job_dir)
    result["srt"] = f"/media/{job_id}/subtitles.srt"
    result["vtt"] = f"/media/{job_id}/subtitles.vtt"
    result["notes"] = f"/media/{job_id}/notes.md"
    return result


def get_paper_text(job_dir: Path) -> str:
    """Return saved paper.txt, or extract and cache it from paper.pdf if missing."""
    txt_path = job_dir / "paper.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    pdf_path = job_dir / "paper.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("Original paper text is not available for this job.")
    cleaned = strip_references(extract_text(pdf_path))
    txt_path.write_text(cleaned, encoding="utf-8")
    return cleaned


def _is_useful_image(img_bytes: bytes, min_w: int = 220, min_h: int = 160) -> tuple[bool, int]:
    """Check dimensions, aspect ratio, and color spread so tiny icons/blank boxes are skipped."""
    import io
    from PIL import Image, ImageStat

    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            w, h = im.size
            if w < min_w or h < min_h:
                return False, 0
            ratio = w / max(1, h)
            if ratio < 0.32 or ratio > 3.6:
                return False, 0
            gray = im.convert("L")
            stat = ImageStat.Stat(gray)
            if stat.stddev[0] < 12.0:  # near-blank or solid color
                return False, 0
            return True, w * h
    except Exception:
        return False, 0


def extract_figures(pdf_path: Path, out_dir: Path, max_figures: int = 4) -> list:
    """Pull candidate figures from the PDF using PyMuPDF (embedded images + caption crops)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    seen_xrefs = set()
    cap_re = re.compile(r"((?:Figure|Fig\.?)\s*\d+[:.\s][^\n]{4,110})", re.IGNORECASE)

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            page_text = page.get_text()
            caps = [m.group(1).strip() for m in cap_re.finditer(page_text)]
            default_cap = caps[0] if caps else f"Figure from page {page_idx + 1}"

            for info in page.get_images(full=True):
                xref = info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                    img_bytes = extracted.get("image", b"")
                except Exception:
                    continue
                ok, area = _is_useful_image(img_bytes)
                if not ok:
                    continue
                candidates.append(
                    {
                        "bytes": img_bytes,
                        "caption": default_cap[:75],
                        "context": page_text[:2000],
                        "score": area + (50_000 if caps else 0),
                    }
                )

        # Fallback: if fewer than 2 raster images were found, crop regions above "Figure N" captions
        if len(candidates) < 2:
            for page_idx, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for b in blocks:
                    btext = str(b[4]).strip() if len(b) > 4 else ""
                    m = cap_re.match(btext)
                    if not m:
                        continue
                    cap_y0 = float(b[1])
                    if cap_y0 < 160:
                        continue
                    clip = fitz.Rect(
                        page.rect.x0 + 36,
                        max(32.0, cap_y0 - 240.0),
                        page.rect.x1 - 36,
                        cap_y0 - 4.0,
                    )
                    try:
                        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2), alpha=False)
                        png_bytes = pix.tobytes("png")
                        ok, area = _is_useful_image(png_bytes, min_w=280, min_h=160)
                        if ok:
                            candidates.append(
                                {
                                    "bytes": png_bytes,
                                    "caption": m.group(1).strip()[:75],
                                    "context": page.get_text()[:2000],
                                    "score": area + 40_000,
                                }
                            )
                    except Exception:
                        continue

    candidates.sort(key=lambda c: c["score"], reverse=True)
    saved = []
    import io
    from PIL import Image

    for idx, cand in enumerate(candidates[:max_figures]):
        out_path = out_dir / f"fig_{idx:02d}.png"
        try:
            with Image.open(io.BytesIO(cand["bytes"])) as im:
                im.convert("RGB").save(out_path, "PNG")
            saved.append(
                {
                    "path": str(out_path),
                    "caption": cand["caption"],
                    "context": cand["context"],
                }
            )
        except Exception:
            continue
    return saved


def attach_figures(scenes: list, figures: list, max_attach: int = 2) -> None:
    """Attach the best 1 or 2 extracted PDF figures to the best-matching scenes."""
    if not figures or not scenes:
        return
    used_scenes = set()
    attached = 0
    for fig in figures[:max_attach]:
        fig_words = {
            w
            for w in re.findall(r"[a-zA-Z0-9]{4,}", (fig["caption"] + " " + fig["context"]).lower())
        }
        best_idx = None
        best_score = -1
        for i, s in enumerate(scenes):
            if i in used_scenes:
                continue
            vtype = (s.get("visual") or {}).get("type", "none")
            if vtype in ("intro", "outro"):
                continue
            s_text = f"{s.get('title', '')} {' '.join(s.get('bullets', []))} {s.get('narration', '')}".lower()
            overlap = sum(1 for w in fig_words if w in s_text)
            # Prefer replacing 'none' visuals or middle/results scenes
            bonus = 3 if vtype == "none" else 0
            score = overlap + bonus
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            used_scenes.add(best_idx)
            scenes[best_idx]["visual"] = {
                "type": "figure",
                "path": fig["path"],
                "caption": fig["caption"],
            }
            attached += 1
            if attached >= max_attach:
                break


def add_intro_outro(script: dict) -> list:
    """Wrap the body scenes with a short animated Intro (paper title) and Outro (3 key takeaways)."""
    title = script.get("title", "Paper Explained")
    summary = script.get("summary", "")
    takeaways = (script.get("takeaways") or [])[:3]
    if len(takeaways) < 3 and script.get("scenes"):
        for b in script["scenes"][-1].get("bullets", []):
            if b not in takeaways and len(takeaways) < 3:
                takeaways.append(b)

    intro_narration = f"{title}. {summary}".strip()
    intro_scene = {
        "title": title,
        "source": "Abstract & Overview",
        "bullets": [summary] if summary else [title],
        "narration": intro_narration,
        "visual": {"type": "intro", "summary": summary},
    }

    outro_spoken = "To wrap up, here are three key takeaways. " + " ".join(
        f"Number {i}: {t.rstrip('.')}." for i, t in enumerate(takeaways, 1)
    )
    outro_scene = {
        "title": "3 Key Takeaways",
        "source": "Core Conclusions",
        "bullets": takeaways,
        "narration": outro_spoken,
        "visual": {"type": "outro", "takeaways": takeaways},
    }
    return [intro_scene] + list(script.get("scenes", [])) + [outro_scene]


def _format_timings(job_id: str, timings: dict, raw_len: int, material_len: int) -> str:
    q = config.quality_mode()
    lines = [
        "=" * 58,
        f"[PaperExplainer] Stage Timings (job={job_id}, quality={q})",
        "-" * 58,
        f"  1. Extract PDF text     : {timings.get('extract_text_s', 0.0):6.2f}s  ({raw_len:,} chars)",
        f"  2. Extract PDF figures  : {timings.get('extract_figures_s', 0.0):6.2f}s",
        f"  3. Filter & summarize   : {timings.get('condense_s', 0.0):6.2f}s  (-> {material_len:,} chars)",
        f"  4. Generate LLM script  : {timings.get('make_script_s', 0.0):6.2f}s",
        f"  5. Prepare slides       : {timings.get('slides_prep_s', 0.0):6.2f}s",
        f"  6. TTS narration (x4)   : {timings.get('tts_narration_s', 0.0):6.2f}s",
        f"  7. Render clips (x3)    : {timings.get('render_clips_s', 0.0):6.2f}s",
        f"  8. Concat + extras      : {timings.get('concat_video_s', 0.0):6.2f}s",
        "-" * 58,
        f"  TOTAL TIME              : {timings.get('total_s', 0.0):6.2f}s",
        "=" * 58,
    ]
    return "\n".join(lines)


def run(
    job_dir: Path,
    job_id: str,
    report,
    voice: str | None = None,
    rate: str = "+0%",
    style: str = "default",
) -> dict:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    pdf_path = job_dir / "paper.pdf"
    cache_dir = job_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract PDF text (cached)
    t0 = time.perf_counter()
    raw_cache = cache_dir / "raw_paper.txt"
    txt_path = job_dir / "paper.txt"
    if raw_cache.exists() and raw_cache.stat().st_size > 0:
        report("Reading the PDF (cached)", 5)
        text = raw_cache.read_text(encoding="utf-8")
    else:
        report("Reading the PDF", 5)
        text = extract_text(pdf_path)
        raw_cache.write_text(text, encoding="utf-8")
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        txt_path.write_text(strip_references(text), encoding="utf-8")
    timings["extract_text_s"] = round(time.perf_counter() - t0, 2)

    # Stage 2: Extract figures (cached)
    t0 = time.perf_counter()
    figs_cache = cache_dir / "figures.json"
    figures = None
    if figs_cache.exists():
        try:
            cached_figs = json.loads(figs_cache.read_text(encoding="utf-8"))
            if all(Path(f["path"]).exists() for f in cached_figs):
                report("Extracting figures from the PDF (cached)", 9)
                figures = cached_figs
        except Exception:
            figures = None
    if figures is None:
        report("Extracting figures from the PDF", 9)
        figures = extract_figures(pdf_path, job_dir / "figures", max_figures=4)
        figs_cache.write_text(json.dumps(figures, indent=2), encoding="utf-8")
    timings["extract_figures_s"] = round(time.perf_counter() - t0, 2)

    # Stage 3: Filter sections & parallel chunk summarization (cached per chunk)
    t0 = time.perf_counter()
    report("Understanding the paper", 14)
    material = condense(text, report, cache_dir=cache_dir)
    timings["condense_s"] = round(time.perf_counter() - t0, 2)

    # Stage 4: Generate LLM script (cached per style)
    t0 = time.perf_counter()
    script_cache = cache_dir / f"script_{style}.json"
    script = None
    if script_cache.exists() and script_cache.stat().st_size > 0:
        try:
            cached_script = json.loads(script_cache.read_text(encoding="utf-8"))
            if cached_script.get("scenes"):
                report("Loaded cached video script", 45)
                script = cached_script
        except Exception:
            script = None
    if script is None:
        report("Writing the video script", 45)
        script = llm.make_script(material, style=style)
        script_cache.write_text(json.dumps(script, indent=2), encoding="utf-8")
    timings["make_script_s"] = round(time.perf_counter() - t0, 2)

    # Attach figures & bookends on a copy of scenes so cached script stays clean
    body_scenes = [dict(s) for s in script["scenes"]]
    attach_figures(body_scenes, figures, max_attach=2)
    full_scenes = add_intro_outro({**script, "scenes": body_scenes})
    script_with_bookends = {**script, "scenes": full_scenes}

    # Stage 5: Prepare slides
    t0 = time.perf_counter()
    slides_dir = job_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    if video.use_animation():
        report("Preparing animated scenes", 52)
        slide_paths = [slides_dir / f"slide_{i:02d}.png" for i in range(len(full_scenes))]
    else:
        report("Drawing the slides", 52)
        slide_paths = slides.render_all(script_with_bookends, slides_dir)
    timings["slides_prep_s"] = round(time.perf_counter() - t0, 2)

    # Stages 6-8: Parallel TTS, Parallel Clip Render, FFmpeg Concat
    chosen_voice = voice or os.getenv("EDGE_VOICE", "en-US-AriaNeural")
    timed_scenes = video.build_video(
        full_scenes,
        slide_paths,
        job_dir,
        chosen_voice,
        report,
        rate=rate,
        paper_title=script["title"],
        timings=timings,
    )

    t_extra = time.perf_counter()
    timings["total_s"] = round(time.perf_counter() - t_start, 2)

    result = {
        "title": script["title"],
        "summary": script["summary"],
        "takeaways": script.get("takeaways", []),
        "scenes": timed_scenes,
        "glossary": script["glossary"],
        "video": f"/media/{job_id}/video.mp4",
        "voice": chosen_voice,
        "rate": rate,
        "style": style,
        "animated": video.use_animation(),
        "quality": config.quality_mode(),
        "timings": timings,
    }
    ensure_extras(job_dir, job_id, result)
    timings["concat_video_s"] = round(
        timings.get("concat_video_s", 0.0) + (time.perf_counter() - t_extra), 2
    )
    timings["total_s"] = round(time.perf_counter() - t_start, 2)

    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_table = _format_timings(job_id, timings, len(text), len(material))
    print(summary_table, flush=True)
    (cache_dir / "timings.txt").write_text(summary_table + "\n", encoding="utf-8")
    report(f"Done in {timings['total_s']:.1f}s", 100)
    return result



