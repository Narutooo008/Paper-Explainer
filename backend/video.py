"""Turns slides or animated scenes + narration text into one MP4 using edge-tts and FFmpeg."""
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import edge_tts

import animate
import config

PAD = 0.7  # seconds of silence after each scene's narration
TTS_CONCURRENCY = 4
CLIP_WORKERS = 3


def use_animation() -> bool:
    val = os.getenv("USE_ANIMATION", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "FFmpeg is not installed. Install it (see README) and restart the server."
        )


def run(cmd: list) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-400:]}")
    return result.stdout


async def _synthesize_one(
    idx: int,
    text: str,
    out_mp3: Path,
    voice: str,
    rate: str,
    sem: asyncio.Semaphore,
    on_done=None,
) -> Path:
    if out_mp3.exists() and out_mp3.stat().st_size > 0:
        if on_done:
            on_done(idx, cached=True)
        return out_mp3

    tmp_mp3 = out_mp3.with_suffix(".tmp.mp3")
    async with sem:
        last_err = None
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, voice, rate=rate).save(str(tmp_mp3))
                if tmp_mp3.exists() and tmp_mp3.stat().st_size > 0:
                    tmp_mp3.replace(out_mp3)
                    if on_done:
                        on_done(idx, cached=False)
                    return out_mp3
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.6 * (attempt + 1))
        raise RuntimeError(
            f"Narration failed (edge-tts needs an internet connection): {last_err}"
        )


def synthesize(text: str, out_mp3: Path, voice: str, rate: str = "+0%") -> None:
    async def go():
        sem = asyncio.Semaphore(1)
        await _synthesize_one(0, text, out_mp3, voice, rate, sem)

    asyncio.run(go())


def synthesize_all(
    scenes: list,
    work_dir: Path,
    voice: str,
    rate: str = "+0%",
    report=None,
) -> list[Path]:
    """Generate edge-tts MP3s for all scenes concurrently (max 4 at once), with caching."""
    n = len(scenes)
    mp3_paths = [work_dir / f"voice_{i:02d}.mp3" for i in range(n)]
    done_count = 0

    def _on_done(_idx: int, cached: bool = False) -> None:
        nonlocal done_count
        done_count += 1
        if report:
            pct = 54 + int(12 * done_count / max(1, n))
            tag = " (cached)" if cached else ""
            report(f"Recording narration {done_count} of {n}{tag}", pct)

    async def _run_batch():
        sem = asyncio.Semaphore(TTS_CONCURRENCY)
        tasks = [
            _synthesize_one(i, str(s.get("narration", "")), mp3_paths[i], voice, rate, sem, _on_done)
            for i, s in enumerate(scenes)
        ]
        return await asyncio.gather(*tasks)

    if report:
        report(f"Recording narration (0 of {n})", 54)
    asyncio.run(_run_batch())
    return mp3_paths


def audio_duration(path: Path) -> float:
    out = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.strip())


def make_clip(
    image: Path,
    audio: Path,
    out: Path,
    duration: float,
    quality: str | None = None,
) -> None:
    out_w, out_h, fps = animate.quality_spec(quality)
    tmp_out = out.with_suffix(".tmp.mp4")
    run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(fps), "-i", str(image),
            "-i", str(audio),
            "-af", f"apad=pad_dur={PAD}",
            "-t", f"{duration:.3f}",
            "-vf", f"scale={out_w}:{out_h}",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-f", "mp4", str(tmp_out),
        ]
    )
    tmp_out.replace(out)


def make_animated_clip(
    scene: dict,
    index: int,
    total: int,
    paper_title: str,
    audio: Path,
    out: Path,
    duration: float,
    preview_path: Path | None = None,
    quality: str | None = None,
) -> None:
    out_w, out_h, fps = animate.quality_spec(quality)
    tmp_out = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-i", str(audio),
        "-af", f"apad=pad_dur={PAD}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-f", "mp4", str(tmp_out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for raw_frame in animate.render_scene_frames(
            scene,
            index,
            total,
            paper_title,
            duration,
            PreviewPath=preview_path,
            quality=quality,
        ):
            proc.stdin.write(raw_frame)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    proc.wait()
    if proc.returncode != 0:
        err_text = stderr_bytes.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"FFmpeg animation encode failed: {err_text}")
    tmp_out.replace(out)


def _build_scene_clip_worker(task: dict) -> tuple[int, bool]:
    """Top-level worker for ProcessPoolExecutor / ThreadPoolExecutor."""
    i = task["index"]
    n = task["total"]
    scene = task["scene"]
    paper_title = task["paper_title"]
    mp3 = Path(task["mp3"])
    clip = Path(task["clip"])
    slide = Path(task["slide"])
    duration = float(task["duration"])
    animated = bool(task["animated"])
    quality = task["quality"]

    if clip.exists() and clip.stat().st_size > 0 and (not animated or slide.exists()):
        return i, True

    if animated:
        make_animated_clip(
            scene,
            i,
            n,
            paper_title,
            mp3,
            clip,
            duration,
            preview_path=slide,
            quality=quality,
        )
    else:
        make_clip(slide, mp3, clip, duration, quality=quality)
    return i, False


def concat(clips: list, out: Path, work_dir: Path) -> None:
    list_file = work_dir / "clips.txt"
    list_file.write_text(
        "".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8"
    )
    tmp_out = out.with_suffix(".tmp.mp4")
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", "-movflags", "+faststart", "-f", "mp4", str(tmp_out),
        ]
    )
    tmp_out.replace(out)


def _prepare_work_cache(
    work: Path,
    scenes: list,
    voice: str,
    rate: str,
    animated: bool,
    quality: str,
) -> None:
    """Invalidate cached audio/clips only if narration text, voice, rate, or quality changed."""
    meta_file = work / "cache_meta.json"
    narration_sig = [str(s.get("narration", "")) for s in scenes]
    scene_sig = [json.dumps(s, sort_keys=True) for s in scenes]
    current_meta = {
        "voice": voice,
        "rate": rate,
        "animated": animated,
        "quality": quality,
        "narration_sig": narration_sig,
        "scene_sig": scene_sig,
    }
    if meta_file.exists():
        try:
            prev = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        voice_same = (
            prev.get("voice") == voice
            and prev.get("rate") == rate
            and prev.get("narration_sig") == narration_sig
        )
        render_same = (
            voice_same
            and prev.get("animated") == animated
            and prev.get("quality") == quality
            and prev.get("scene_sig") == scene_sig
        )
        if not voice_same:
            for f in work.glob("voice_*.mp3"):
                f.unlink(missing_ok=True)
        if not render_same:
            for f in work.glob("clip_*.mp4"):
                f.unlink(missing_ok=True)

    meta_file.write_text(json.dumps(current_meta, indent=2), encoding="utf-8")


def build_video(
    scenes: list,
    slide_paths: list,
    job_dir: Path,
    voice: str,
    report,
    rate: str = "+0%",
    paper_title: str = "Paper explained",
    timings: dict | None = None,
) -> list:
    """Makes video.mp4 in job_dir. Returns scenes with 'start' and 'duration' (seconds)."""
    check_ffmpeg()
    work = job_dir / "work"
    work.mkdir(exist_ok=True)
    animated = use_animation()
    quality = config.quality_mode()
    n = len(scenes)

    _prepare_work_cache(work, scenes, voice, rate, animated, quality)

    # Stage 1: Parallel edge-tts synthesis (max 4 concurrent)
    t0 = time.perf_counter()
    mp3_paths = synthesize_all(scenes, work, voice, rate=rate, report=report)
    t_tts = time.perf_counter() - t0
    if timings is not None:
        timings["tts_narration_s"] = round(t_tts, 2)

    # Compute exact scene timings in original order
    clips, timed, t = [], [], 0.0
    tasks = []
    for i, (scene, slide, mp3) in enumerate(zip(scenes, slide_paths, mp3_paths)):
        duration = audio_duration(mp3) + PAD
        clip = work / f"clip_{i:02d}.mp4"
        clips.append(clip)
        timed.append({**scene, "start": round(t, 2), "duration": round(duration, 2)})
        t += duration
        tasks.append(
            {
                "index": i,
                "total": n,
                "scene": scene,
                "paper_title": paper_title,
                "mp3": str(mp3),
                "clip": str(clip),
                "slide": str(slide),
                "duration": duration,
                "animated": animated,
                "quality": quality,
            }
        )

    # Stage 2: Parallel scene rendering & FFmpeg encoding (max 3 workers)
    t1 = time.perf_counter()
    workers = max(1, min(CLIP_WORKERS, n))
    done_clips = 0
    action_label = "Animating scene" if animated else "Building scene"
    report(f"{action_label} (0 of {n})", 66)

    def _run_pool(executor_cls):
        nonlocal done_clips
        with executor_cls(max_workers=workers) as pool:
            futures = [pool.submit(_build_scene_clip_worker, task) for task in tasks]
            for fut in as_completed(futures):
                _idx, was_cached = fut.result()
                done_clips += 1
                pct = 66 + int(26 * done_clips / max(1, n))
                tag = " (cached)" if was_cached else ""
                report(f"{action_label} {done_clips} of {n}{tag}", min(92, pct))

    if animated and workers > 1:
        try:
            _run_pool(ProcessPoolExecutor)
        except Exception as pool_err:
            # Fallback to threads if process spawning fails in restricted environments
            print(f"[video] ProcessPoolExecutor fallback to ThreadPoolExecutor: {pool_err}")
            done_clips = 0
            _run_pool(ThreadPoolExecutor)
    else:
        _run_pool(ThreadPoolExecutor)

    t_render = time.perf_counter() - t1
    if timings is not None:
        timings["render_clips_s"] = round(t_render, 2)

    # Stage 3: Concat clips into final video.mp4
    t2 = time.perf_counter()
    report("Putting the video together", 94)
    concat(clips, job_dir / "video.mp4", work)
    t_concat = time.perf_counter() - t2
    if timings is not None:
        timings["concat_video_s"] = round(t_concat, 2)

    return timed

