"""Unit tests for JSON cleanup (llm.py), diagram layout & frame rendering (animate.py),
and figure/bookend helpers (pipeline.py).

Run from the project folder:
    python backend/test_unit.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PIL import Image

import animate
import config
import llm
import pipeline
import video


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


class TestJsonCleanup(unittest.TestCase):
    def test_parse_json_with_fences_and_chatter(self):
        raw = 'Here is your JSON:\n```json\n{"title": "Test", "scenes": []}\n```\nHope that helps!'
        parsed = llm.parse_json(raw)
        self.assertEqual(parsed["title"], "Test")

    def test_parse_json_missing_braces_raises(self):
        with self.assertRaises(ValueError):
            llm.parse_json("no json object here")

    def test_clean_visual_flow_valid_and_label_edges(self):
        vis = {
            "type": "flow",
            "nodes": [
                "First very long node label with many words",
                {"id": "n2", "label": "Second Step"},
                "Third Step",
            ],
            "edges": [
                {"from": 0, "to": "n2", "label": "parses input text"},
                {"from": "Second Step", "to": 2},
            ],
        }
        cleaned = llm.clean_visual(vis)
        self.assertEqual(cleaned["type"], "flow")
        self.assertEqual(len(cleaned["nodes"]), 3)
        self.assertLessEqual(len(cleaned["nodes"][0].split()), 4)
        self.assertEqual(cleaned["edges"][0]["from"], 0)
        self.assertEqual(cleaned["edges"][0]["to"], 1)
        self.assertEqual(cleaned["edges"][1]["from"], 1)
        self.assertEqual(cleaned["edges"][1]["to"], 2)

    def test_clean_visual_flow_fallback_on_too_few_nodes(self):
        self.assertEqual(llm.clean_visual({"type": "flow", "nodes": ["OnlyOne"]}), {"type": "none"})

    def test_clean_visual_compare(self):
        vis = {
            "type": "compare",
            "left_title": "Old Method",
            "left_items": ["Slow", "Manual", "Brittle"],
            "right_title": "New Method",
            "right_items": ["Fast", "Automated", "Robust"],
        }
        cleaned = llm.clean_visual(vis)
        self.assertEqual(cleaned["type"], "compare")
        self.assertEqual(len(cleaned["left_items"]), 3)
        self.assertEqual(len(cleaned["right_items"]), 3)

    def test_clean_visual_bar_chart_parses_numeric_strings(self):
        vis = {
            "type": "bar_chart",
            "unit": "%",
            "bars": [
                {"label": "Baseline", "value": "42%"},
                {"label": "Variant A", "value": 55.5},
                {"label": "Ours", "value": "68.2%"},
            ],
        }
        cleaned = llm.clean_visual(vis)
        self.assertEqual(cleaned["type"], "bar_chart")
        self.assertEqual(len(cleaned["bars"]), 3)
        self.assertAlmostEqual(cleaned["bars"][0]["value"], 42.0)
        self.assertAlmostEqual(cleaned["bars"][2]["value"], 68.2)

    def test_clean_visual_bar_chart_fallback_under_three(self):
        vis = {"type": "bar_chart", "bars": [{"label": "A", "value": 10}, {"label": "B", "value": 20}]}
        self.assertEqual(llm.clean_visual(vis), {"type": "none"})

    def test_clean_visual_cycle(self):
        vis = {"type": "cycle", "steps": ["Sample", "Train", "Evaluate", "Deploy"]}
        cleaned = llm.clean_visual(vis)
        self.assertEqual(cleaned["type"], "cycle")
        self.assertEqual(len(cleaned["steps"]), 4)

    def test_clean_script_populates_takeaways_and_strips_markdown(self):
        raw = {
            "title": "Test Paper",
            "summary": "A test summary.",
            "scenes": [
                {"title": "Scene 1", "bullets": ["Point A", "Point B"], "narration": "**Bold** and `code`."},
                {"title": "Scene 2", "bullets": ["Point C"], "narration": "Second narration."},
                {"title": "Scene 3", "bullets": ["Point D", "Point E"], "narration": "Third narration."},
            ],
            "glossary": [{"term": "RNA", "meaning": "Ribonucleic acid"}],
        }
        cleaned = llm.clean_script(raw)
        self.assertEqual(len(cleaned["scenes"]), 3)
        self.assertNotIn("*", cleaned["scenes"][0]["narration"])
        self.assertNotIn("`", cleaned["scenes"][0]["narration"])
        self.assertEqual(len(cleaned["takeaways"]), 3)


class TestLayoutAndAnimation(unittest.TestCase):
    def _assert_valid_boxes(self, boxes: list, sw: int = 640, sh: int = 445):
        self.assertTrue(len(boxes) > 0)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            self.assertGreaterEqual(x1, 0, f"Box {i} left out of bounds: {b}")
            self.assertGreaterEqual(y1, 0, f"Box {i} top out of bounds: {b}")
            self.assertLessEqual(x2, sw, f"Box {i} right out of bounds: {b}")
            self.assertLessEqual(y2, sh, f"Box {i} bottom out of bounds: {b}")
            for j in range(i + 1, len(boxes)):
                self.assertFalse(
                    _boxes_overlap(b, boxes[j]),
                    f"Box {i} {b} overlaps with Box {j} {boxes[j]}",
                )

    def test_flow_layout_no_overlap_and_in_bounds(self):
        for n in range(2, 7):
            vis = {"type": "flow", "nodes": [f"Node {i}" for i in range(n)], "edges": []}
            boxes = animate.compute_diagram_boxes(vis)
            self.assertEqual(len(boxes), n)
            self._assert_valid_boxes(boxes)

    def test_cycle_layout_no_overlap_and_in_bounds(self):
        for n in range(3, 6):
            vis = {"type": "cycle", "steps": [f"Step {i}" for i in range(n)]}
            boxes = animate.compute_diagram_boxes(vis)
            self.assertEqual(len(boxes), n)
            self._assert_valid_boxes(boxes)

    def test_compare_and_bar_chart_layout(self):
        comp = {
            "type": "compare",
            "left_items": ["A1", "A2", "A3", "A4"],
            "right_items": ["B1", "B2", "B3", "B4"],
        }
        self._assert_valid_boxes(animate.compute_diagram_boxes(comp))

        bars = {
            "type": "bar_chart",
            "bars": [{"label": f"M{i}", "value": (i + 1) * 15} for i in range(6)],
        }
        self._assert_valid_boxes(animate.compute_diagram_boxes(bars))

    def test_render_all_visual_types_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig_path = Path(tmp) / "sample_fig.png"
            Image.new("RGB", (400, 260), (40, 90, 180)).save(fig_path, "PNG")

            visuals = [
                {"type": "intro", "summary": "Short one-sentence overview."},
                {"type": "flow", "nodes": ["A", "B", "C"], "edges": [{"from": 0, "to": 1}, {"from": 1, "to": 2}]},
                {"type": "compare", "left_title": "Old", "left_items": ["A", "B"], "right_title": "New", "right_items": ["C", "D"]},
                {"type": "bar_chart", "unit": "%", "bars": [{"label": "X", "value": 10}, {"label": "Y", "value": 20}, {"label": "Z", "value": 30}]},
                {"type": "cycle", "steps": ["One", "Two", "Three"]},
                {"type": "figure", "path": str(fig_path), "caption": "Figure 1: Sample architecture"},
                {"type": "none"},
                {"type": "outro", "takeaways": ["First takeaway", "Second takeaway", "Third takeaway"]},
            ]
            for i, vis in enumerate(visuals):
                scene = {
                    "title": f"Test Scene {i}",
                    "bullets": ["Bullet one", "Bullet two", "Bullet three"],
                    "narration": "Test narration sentence.",
                    "visual": vis,
                }
                layout = animate._prepare_scene_layout(scene, i, len(visuals), "Test Paper")
                frame = animate.render_scene_frame(layout, t=2.0, duration=5.0, index=i, total=len(visuals), diagram_cache={})
                self.assertEqual(frame.size, (1280, 720))
                self.assertEqual(frame.mode, "RGB")

    def test_intro_outro_and_figure_matching(self):
        script = {
            "title": "Membrane Assembly",
            "summary": "Nucleotides form vesicles in water.",
            "takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],
            "scenes": [
                {"title": "Problem", "bullets": ["Hard in water"], "narration": "Making membranes is hard.", "visual": {"type": "none"}},
                {"title": "Microscopy Vesicles", "bullets": ["Fluorescent dye trapped"], "narration": "Microscopy shows vesicles trapping dye.", "visual": {"type": "none"}},
            ],
        }
        figs = [{"path": "/tmp/fig_00.png", "caption": "Figure 1: Microscopy of fluorescent vesicles", "context": "vesicles dye microscopy"}]
        pipeline.attach_figures(script["scenes"], figs, max_attach=1)
        self.assertEqual(script["scenes"][1]["visual"]["type"], "figure")

        full = pipeline.add_intro_outro(script)
        self.assertEqual(len(full), 4)
        self.assertEqual(full[0]["visual"]["type"], "intro")
        self.assertEqual(full[-1]["visual"]["type"], "outro")

    def test_section_filtering_and_char_cap(self):
        sample_paper = "\n".join(
            [
                "Paper Title",
                "Abstract",
                "This paper introduces FastNet for rapid inference. " * 25,
                "1. Introduction",
                "Deep models are often slow in production environments. " * 25,
                "2. Related Work",
                "Prior work by Smith et al. explored legacy architectures that we skip. " * 40,
                "3. Methodology",
                "Our method uses sparse attention blocks with linear scaling. " * 30,
                "4. Experiments and Results",
                "We achieve 94.2% accuracy with 3x lower latency on ImageNet. " * 30,
                "5. Conclusion",
                "FastNet provides a practical path to fast deployment. " * 20,
                "Acknowledgments",
                "We thank our anonymous reviewers.",
                "References",
                "[1] Smith et al. 2020.",
                "Appendix A",
                "Extra proofs and hyperparameter tables that should be skipped. " * 40,
            ]
        )
        filtered = pipeline.filter_sections(sample_paper, max_chars=40_000)
        self.assertIn("FastNet for rapid inference", filtered)
        self.assertIn("sparse attention blocks", filtered)
        self.assertIn("94.2% accuracy", filtered)
        self.assertIn("practical path to fast deployment", filtered)
        self.assertNotIn("Prior work by Smith", filtered)
        self.assertNotIn("Extra proofs and hyperparameter tables", filtered)

        huge = "Abstract\n" + ("Useful method and results content. " * 2000)
        capped = pipeline.filter_sections(huge, max_chars=40_000)
        self.assertLessEqual(len(capped), 40_000)

    def test_parallel_condense_preserves_order_and_caches(self):
        import time
        from unittest.mock import patch

        # Create text > 20,000 chars so it splits into 2 chunks
        chunk0 = "1. Introduction\n" + ("Alpha_Chunk_Zero_Content " * 850)
        chunk1 = "\n3. Methodology\n" + ("Beta_Chunk_One_Content " * 850)
        long_text = chunk0 + chunk1

        def fake_summarize(c: str) -> str:
            if "Beta_Chunk_One_Content" in c:
                return "SUMMARY_ONE"
            time.sleep(0.08)  # make chunk 0 finish AFTER chunk 1
            return "SUMMARY_ZERO"

        steps = []
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            with patch.object(llm, "summarize_chunk", side_effect=fake_summarize) as mock_sum:
                out1 = pipeline.condense(long_text, lambda s, p: steps.append((s, p)), cache_dir=cache_dir)
                self.assertEqual(out1, "SUMMARY_ZERO\n\nSUMMARY_ONE")
                self.assertEqual(mock_sum.call_count, 2)
                self.assertTrue((cache_dir / "chunk_00.txt").exists())
                self.assertTrue((cache_dir / "chunk_01.txt").exists())

                # Second call should load directly from cache without calling LLM
                out2 = pipeline.condense(long_text, lambda s, p: None, cache_dir=cache_dir)
                self.assertEqual(out2, "SUMMARY_ZERO\n\nSUMMARY_ONE")
                self.assertEqual(mock_sum.call_count, 2)

    def test_quality_spec_fast_and_normal(self):
        self.assertEqual(animate.quality_spec("fast"), (960, 540, 15))
        self.assertEqual(animate.quality_spec("normal"), (1280, 720, 24))
        scene = {
            "title": "Fast Mode Test",
            "bullets": ["Point 1"],
            "narration": "Short test.",
            "visual": {"type": "none"},
        }
        frames = list(animate.render_scene_frames(scene, 0, 1, "Title", duration=0.4, quality="fast"))
        self.assertEqual(len(frames), 6)  # 0.4s * 15 fps = 6 frames
        self.assertEqual(len(frames[0]), 960 * 540 * 3)

    def test_model_fallback_on_410_and_strict_message_content(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from openai import APIStatusError

        llm._dead_models.clear()
        calls = []

        def fake_create(model, messages, temperature, max_tokens):
            calls.append(model)
            if model == "retired/model-410":
                dummy_resp = SimpleNamespace(status_code=410, headers={}, request=SimpleNamespace(), text='{"status": 410}')
                err = APIStatusError("Model retired", response=dummy_resp, body={"status": 410})
                err.status_code = 410
                raise err
            msg = SimpleNamespace(content="FINAL_JSON_CONTENT", reasoning_content="DO_NOT_READ_REASONING")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )
        with patch.object(llm, "client", return_value=fake_client):
            reply = llm.chat("sys", "usr", model="retired/model-410")
            self.assertEqual(reply, "FINAL_JSON_CONTENT")
            self.assertIn("retired/model-410", llm._dead_models)
            self.assertGreaterEqual(len(calls), 2)
        llm._dead_models.clear()

    def test_backend_modules_and_settings_functions(self):
        import ast
        import glob
        import importlib
        import config

        # 1. Verify config and module settings functions
        self.assertTrue(callable(config.quality_mode))
        self.assertIn(config.quality_mode(), ("normal", "fast"))
        self.assertTrue(callable(animate.quality_mode))
        self.assertTrue(callable(animate.quality_spec))
        self.assertTrue(callable(video.use_animation))
        self.assertIsInstance(video.use_animation(), bool)
        self.assertTrue(callable(llm.model_name))
        self.assertTrue(callable(llm.fast_model_name))
        self.assertTrue(callable(llm.model_fallbacks))
        self.assertTrue(callable(llm.check_models_on_startup))

        # 2. Check every backend file's AST for attribute accesses on backend modules
        backend_mods = {"config", "animate", "slides", "video", "llm", "pipeline"}
        imported = {m: importlib.import_module(m) for m in backend_mods}
        missing_attrs = []
        for py_path in glob.glob(str(HERE / "*.py")):
            tree = ast.parse(Path(py_path).read_text(encoding="utf-8"), py_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    m_name, a_name = node.value.id, node.attr
                    if m_name in imported and not hasattr(imported[m_name], a_name):
                        missing_attrs.append(f"{Path(py_path).name}:{node.lineno} {m_name}.{a_name}")

        self.assertEqual(missing_attrs, [], f"Found missing module attributes: {missing_attrs}")


class TestDuplicateDetectionAndRepair(unittest.TestCase):
    def test_clean_script_preserves_source_field(self):
        raw = {
            "title": "Quantum Error Correction",
            "summary": "Surface codes achieve fault tolerance.",
            "scenes": [
                {"title": "Syndrome Extraction", "source": "Section 2.1", "bullets": ["Measure X and Z stabilizers"], "narration": "Stabilizer circuits detect bit flips."},
                {"title": "Logical Qubit Scaling", "source": "Section 3.4 & Table 1", "bullets": ["Distance-7 code achieves 1e-6 error"], "narration": "Larger lattices suppress errors exponentially."},
                {"title": "Cryogenic Control Limits", "source": "Section 5", "bullets": ["Thermal dissipation limits qubit count"], "narration": "Wiring heat presents the main engineering bottleneck."},
            ],
            "takeaways": ["Stabilizer circuits work", "Scale requires cooling", "Fault tolerance demonstrated"],
        }
        cleaned = llm.clean_script(raw)
        self.assertEqual(len(cleaned["scenes"]), 3)
        self.assertEqual(cleaned["scenes"][0]["source"], "Section 2.1")
        self.assertEqual(cleaned["scenes"][1]["source"], "Section 3.4 & Table 1")
        self.assertEqual(cleaned["scenes"][2]["source"], "Section 5")

    def test_detect_duplicates_clean_script(self):
        scenes = [
            {"title": "Self-Attention Bottleneck", "source": "Section 1", "bullets": ["Quadratic memory scaling", "Limits token window to 2048"], "narration": "Standard transformers struggle with lengthy contexts."},
            {"title": "FlashAttention GPU Kernel", "source": "Section 3", "bullets": ["Tiled matrix multiplication", "SRAM buffer reuse"], "narration": "By computing softmax in fast on-chip SRAM, memory reads vanish."},
            {"title": "3.5x Faster Training Latency", "source": "Section 4.2", "bullets": ["Benchmarked on 8x A100 GPUs", "Tested on 64k sequence length"], "narration": "Wall-clock training runs over three times faster without accuracy loss."},
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="FlashAttention Optimization")
        self.assertEqual(issues, {})

    def test_detect_exact_and_near_duplicate_titles(self):
        scenes = [
            {"title": "The Crisis of Social Reproduction", "source": "Section 1", "bullets": ["Care work devalued"], "narration": "Household care is strained by economic pressures."},
            {"title": "The Crisis of Social Reproduction", "source": "Section 2", "bullets": ["Institutions failing"], "narration": "Welfare systems have retreated from social support."},
            {"title": "3 Key Takeaways", "source": "Section 3", "bullets": ["Reforms needed"], "narration": "We must rethink the relationship between labor and life."},
            {"title": "Key Takeaways", "source": "Section 4", "bullets": ["Invest in communities"], "narration": "Collective networks offer alternative pathways."},
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="Care Labor")
        # Scene 1 has exact duplicate title with Scene 0
        self.assertTrue(any("duplicate" in r.lower() for r in issues.get(1, [])))
        # Scene 2 & 3 have generic / near-duplicate takeaway titles
        self.assertTrue(any("generic" in r.lower() or "takeaway" in r.lower() for r in issues.get(2, [])))
        self.assertTrue(any("generic" in r.lower() or "takeaway" in r.lower() or "duplicate" in r.lower() for r in issues.get(3, [])))

    def test_detect_generic_titles(self):
        scenes = [
            {"title": "The Results", "source": "Section 4", "bullets": ["Model scored 88%"], "narration": "Experiments show high fidelity across benchmarks."},
            {"title": "Limitations", "source": "Section 5", "bullets": ["High memory overhead"], "narration": "Training requires massive clusters."},
            {"title": "Takeaway", "source": "Section 6", "bullets": ["Promising future"], "narration": "The framework establishes a strong baseline."},
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="Novel AI")
        self.assertIn(0, issues)
        self.assertIn(1, issues)
        self.assertIn(2, issues)
        self.assertTrue(any("generic" in r.lower() for r in issues[0]))
        self.assertTrue(any("generic" in r.lower() for r in issues[1]))
        self.assertTrue(any("generic" in r.lower() for r in issues[2]))

    def test_detect_title_repeating_paper_title(self):
        scenes = [
            {"title": "Paper Explained", "source": "Section 1", "bullets": ["Intro points"], "narration": "Welcome to this research walkthrough."},
            {"title": "Deep Residual Learning for Image Recognition", "source": "Section 1", "bullets": ["Skip connections"], "narration": "Vanishing gradients degrade deep layers."},
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="Deep Residual Learning for Image Recognition")
        self.assertIn(1, issues)
        self.assertTrue(any("repeats" in r.lower() for r in issues[1]))

    def test_detect_narration_starting_with_title(self):
        scenes = [
            {
                "title": "Cryogenic Heat Dissipation",
                "source": "Section 4",
                "bullets": ["Wiring dissipation 1.2W"],
                "narration": "Cryogenic Heat Dissipation represents a major obstacle for scaling.",
            },
            {
                "title": "Superconducting Qubits",
                "source": "Section 2",
                "bullets": ["Coherence time 150us"],
                "narration": "In superconducting qubits, microwave pulses drive single-qubit rotations.",
            },
            {
                "title": "Scalable Readout Lines",
                "source": "Section 3",
                "bullets": ["Frequency multiplexing 8x"],
                "narration": "Engineers combine multiple microwave tones onto a single coaxial cable.",
            },
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="Quantum Processor Architecture")
        # Scene 0 begins directly with title
        self.assertIn(0, issues)
        self.assertTrue(any("repeating slide title" in r.lower() for r in issues[0]))
        # Scene 1 begins with "In {title}"
        self.assertIn(1, issues)
        self.assertTrue(any("repeating slide title" in r.lower() for r in issues[1]))
        # Scene 2 does not repeat title
        self.assertNotIn(2, issues)

    def test_detect_generic_and_duplicate_bullets(self):
        scenes = [
            {"title": "Empirical Verification", "source": "Section 4", "bullets": ["Limited empirical data", "Sample size 50 patients"], "narration": "Clinical trials are in early stages."},
            {"title": "Diagnostic Accuracy", "source": "Section 5", "bullets": ["Sample size 50 patients", "Sensitivity reached 96%"], "narration": "Screening detected early-stage markers reliably."},
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="Biomarker Discovery")
        # Scene 0 has generic bullet "Limited empirical data"
        self.assertIn(0, issues)
        self.assertTrue(any("generic filler" in r.lower() for r in issues[0]))
        # Scene 1 duplicates bullet "Sample size 50 patients"
        self.assertIn(1, issues)
        self.assertTrue(any("duplicates" in r.lower() for r in issues[1]))

    def test_detect_cross_scene_repeated_sentences(self):
        scenes = [
            {"title": "Sparse Attention Patterns", "source": "Section 2", "bullets": ["Fixed stride block size"], "narration": "We evaluate performance across sixteen benchmark datasets. Memory footprints drop by fifty percent."},
            {"title": "Ablation Analysis", "source": "Section 4", "bullets": ["Head count sensitivity"], "narration": "We evaluate performance across sixteen benchmark datasets. Different layers exhibit varying sparsity ratios."},
        ]
        issues = llm.detect_duplicate_scenes(scenes, paper_title="Sparse Attention")
        self.assertIn(1, issues)
        self.assertTrue(any("repeats scene 1" in r.lower() for r in issues[1]))

    def test_validate_and_repair_invokes_repair_call_and_fixes_issues(self):
        from unittest.mock import patch
        bad_script = {
            "title": "Quantum Computing",
            "summary": "Summary of quantum computing.",
            "takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],
            "scenes": [
                {
                    "title": "Superconducting Qubits",
                    "source": "Section 2",
                    "bullets": ["Transmon design with 100us coherence"],
                    "narration": "Microwave pulses steer each individual state on the Bloch sphere.",
                    "visual": {"type": "none"},
                },
                {
                    "title": "The Results",  # Generic title!
                    "source": "Section 4",
                    "bullets": ["Limited empirical data"],  # Generic bullet!
                    "narration": "The Results show fidelity improvements.",  # Starts with title!
                    "visual": {"type": "none"},
                },
            ],
            "glossary": [],
        }

        repaired_reply = json.dumps({
            "repaired_scenes": [
                {
                    "index": 1,
                    "title": "99.8% Two-Qubit Gate Fidelity",
                    "source": "Section 4.1 & Figure 3",
                    "bullets": ["Cross-resonance drive pulse", "0.2% error measured via randomized benchmarking"],
                    "narration": "Randomized benchmarking confirmed that error rates dropped well below the fault-tolerance threshold.",
                    "visual": {"type": "none"},
                }
            ]
        })

        chat_calls = []

        def fake_chat(system, user, temperature=0.3, max_tokens=3500, model=None):
            chat_calls.append({"system": system, "user": user, "model": model})
            return repaired_reply

        with patch.object(llm, "chat", side_effect=fake_chat):
            fixed_script = llm.validate_and_repair_script(bad_script, material="Sample paper material with 99.8% fidelity.")

            # Exactly one repair call was made
            self.assertEqual(len(chat_calls), 1)
            # Used the stronger model
            self.assertEqual(chat_calls[0]["model"], llm.model_name())

            # Scene 1 was replaced with concrete, non-duplicate content
            self.assertEqual(fixed_script["scenes"][1]["title"], "99.8% Two-Qubit Gate Fidelity")
            self.assertEqual(fixed_script["scenes"][1]["source"], "Section 4.1 & Figure 3")
            self.assertIn("Cross-resonance", fixed_script["scenes"][1]["bullets"][0])

            # Now verifying issues on repaired script -> zero issues!
            post_issues = llm.detect_duplicate_scenes(fixed_script["scenes"], fixed_script["title"])
            self.assertEqual(post_issues, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)



