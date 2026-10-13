"""Standalone offline test script for Phase 2 animations.
Run from the project folder:
    python backend/test_animate.py
Produces test_animation.mp4 with one sample scene of each visual type (flow, compare,
bar_chart, cycle, none) without calling any external API.
"""
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import animate
import config
import llm
import video

SAMPLE_SCENES = [
    {
        "title": "How the Pipeline Works",
        "bullets": [
            "Extract clean text from PDF",
            "Generate structured JSON script",
            "Render 24 fps animated scenes",
            "Mux narration into final MP4",
        ],
        "narration": "Sample flow scene.",
        "visual": {
            "type": "flow",
            "nodes": ["Input PDF", "Text Chunks", "LLM Script", "24fps Frames", "Final MP4"],
            "edges": [
                {"from": 0, "to": 1, "label": "PyMuPDF"},
                {"from": 1, "to": 2, "label": "summarize"},
                {"from": 2, "to": 3, "label": "Pillow"},
                {"from": 3, "to": 4, "label": "FFmpeg"},
            ],
        },
    },
    {
        "title": "Static Slides vs Animated Scenes",
        "bullets": [
            "Old slides show everything at once",
            "Animated scenes reveal points in order",
            "Diagrams clarify methods and numbers",
        ],
        "narration": "Sample compare scene.",
        "visual": {
            "type": "compare",
            "left_title": "Static Slides",
            "left_items": ["All text at once", "No visual diagrams", "Harder to follow"],
            "right_title": "Animated Explainer",
            "right_items": ["Timed bullet reveal", "Auto-layout diagrams", "Active yellow focus"],
        },
    },
    {
        "title": "Experimental Yield Results",
        "bullets": [
            "Short chains reach 42% yield",
            "Mixed solvent reaches 55%",
            "Thioester boosts yield to 68%",
            "Vesicles remain stable for 85%",
        ],
        "narration": "Sample bar chart scene.",
        "visual": {
            "type": "bar_chart",
            "unit": "%",
            "bars": [
                {"label": "Baseline", "value": 18},
                {"label": "Short Chain", "value": 42},
                {"label": "Co-Solvent", "value": 55},
                {"label": "Thioester", "value": 68},
            ],
        },
    },
    {
        "title": "Iterative Self-Assembly Loop",
        "bullets": [
            "Nucleotides acylate in water",
            "Lipids self-assemble into micelles",
            "Bilayers close into vesicles",
            "Vesicles trap dye cargo inside",
        ],
        "narration": "Sample cycle scene.",
        "visual": {
            "type": "cycle",
            "steps": ["Acylate RNA", "Form Lipids", "Build Bilayer", "Trap Cargo"],
        },
    },
    {
        "title": "Key Takeaways from the Paper",
        "bullets": [
            "Simple nucleotides can form membranes in water",
            "No enzymes or extreme heat required",
            "Offers a plausible path to early protocells",
        ],
        "narration": "Sample bullets-only scene.",
        "visual": {"type": "none"},
    },
]


def main() -> None:
    video.check_ffmpeg()
    work = ROOT / "jobs" / "_test_anim_work"
    work.mkdir(parents=True, exist_ok=True)
    out_mp4 = ROOT / "test_animation.mp4"

    # Generate a 4.3s silent audio track with FFmpeg so no internet/API is needed
    silent_mp3 = work / "silent.mp3"
    video.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "4.3",
            "-c:a", "libmp3lame", "-q:a", "4",
            str(silent_mp3),
        ]
    )
    scene_duration = 5.0  # 4.3s audio + 0.7s PAD

    # Extract a real figure from an existing job PDF if available
    sample_pdfs = list((ROOT / "jobs").glob("*/paper.pdf"))
    fig_path = work / "sample_fig.png"
    fig_caption = "Figure 1: Extracted from research PDF"
    if sample_pdfs:
        import pipeline
        figs = pipeline.extract_figures(sample_pdfs[0], work / "figs", max_figures=1)
        if figs:
            fig_path = Path(figs[0]["path"])
            fig_caption = figs[0]["caption"]
    if not fig_path.exists():
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (480, 300), (240, 245, 252))
        dr = ImageDraw.Draw(im)
        dr.rectangle([40, 40, 440, 260], outline=(43, 80, 255), width=4)
        im.save(fig_path, "PNG")

    scenes_to_render = (
        [
            {
                "title": "Making Life's First Membranes Without Enzymes",
                "bullets": ["How simple nucleotides form cell-like vesicles in water."],
                "narration": "Intro scene.",
                "visual": {
                    "type": "intro",
                    "summary": "Scientists showed that simple nucleotides can turn into membrane-forming lipids in water.",
                },
            }
        ]
        + [
            {**s, "visual": llm.clean_visual(s["visual"])}
            for s in SAMPLE_SCENES
        ]
        + [
            {
                "title": "Extracted Figure from Paper",
                "bullets": [
                    "Automatically pulled with PyMuPDF",
                    "Matched to relevant scene by keywords",
                    "Displayed alongside key points",
                ],
                "narration": "Sample figure scene.",
                "visual": {
                    "type": "figure",
                    "path": str(fig_path),
                    "caption": fig_caption,
                },
            },
            {
                "title": "3 Key Takeaways",
                "bullets": [
                    "Nucleotides can lipidate in water without enzymes",
                    "Thioester activation reaches 68% diacylation yield",
                    "Self-assembled vesicles trap cargo for hours",
                ],
                "narration": "Outro scene.",
                "visual": {
                    "type": "outro",
                    "takeaways": [
                        "Nucleotides can lipidate in water without enzymes",
                        "Thioester activation reaches 68% diacylation yield",
                        "Self-assembled vesicles trap cargo for hours",
                    ],
                },
            },
        ]
    )

    from concurrent.futures import ProcessPoolExecutor, as_completed

    quality = config.quality_mode()
    out_w, out_h, fps_target = animate.quality_spec(quality)
    total = len(scenes_to_render)
    print(f"Rendering {total} scenes in parallel (workers={video.CLIP_WORKERS}, quality={quality} [{out_w}x{out_h} @ {fps_target}fps])...")

    t0 = time.perf_counter()
    clips = [work / f"clip_{i:02d}.mp4" for i in range(total)]
    tasks = [
        {
            "index": i,
            "total": total,
            "scene": scene,
            "paper_title": "PaperExplainer Animation Test",
            "mp3": str(silent_mp3),
            "clip": str(clips[i]),
            "slide": str(work / f"preview_{i:02d}.png"),
            "duration": scene_duration,
            "animated": True,
            "quality": quality,
        }
        for i, scene in enumerate(scenes_to_render)
    ]

    with ProcessPoolExecutor(max_workers=video.CLIP_WORKERS) as pool:
        futs = {pool.submit(video._build_scene_clip_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            idx, _ = fut.result()
            vtype = scenes_to_render[idx]["visual"]["type"]
            print(f"  [{idx + 1}/{total}] {vtype:<10} finished")

    video.concat(clips, out_mp4, work)
    shutil.rmtree(work, ignore_errors=True)
    total_elapsed = time.perf_counter() - t0
    print(f"\nDone in {total_elapsed:.2f}s -> {out_mp4}")


if __name__ == "__main__":
    main()
