"""Talks to NVIDIA's OpenAI-compatible API and turns a paper into a video script."""
import json
import os
import re
import threading
import time

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

BASE_URL = "https://integrate.api.nvidia.com/v1"

_client = None
_client_lock = threading.Lock()
_rate_lock = threading.Lock()
_next_allowed_time = 0.0
_MIN_INTERVAL = 60.0 / 36.0  # ~36 requests/min to stay safely under 40 RPM free-tier limit

DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_FAST_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_FALLBACKS = [
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
]
_dead_models: set[str] = set()


def model_name() -> str:
    return os.getenv("NVIDIA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def fast_model_name() -> str:
    return os.getenv("NVIDIA_FAST_MODEL", DEFAULT_FAST_MODEL).strip() or model_name()


def model_fallbacks() -> list[str]:
    raw = os.getenv("NVIDIA_MODEL_FALLBACKS", "")
    parsed = [m.strip() for m in raw.split(",") if m.strip()]
    for fb in DEFAULT_FALLBACKS:
        if fb not in parsed:
            parsed.append(fb)
    return parsed


def _candidate_models(requested: str | None = None) -> list[str]:
    chain = []
    for m in [requested or model_name(), model_name(), fast_model_name(), *model_fallbacks()]:
        if m and m not in chain and m not in _dead_models:
            chain.append(m)
    return chain or [DEFAULT_MODEL]


def client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            key = os.getenv("NVIDIA_API_KEY", "").strip()
            if not key or "paste-your-key" in key:
                raise RuntimeError(
                    "NVIDIA_API_KEY is missing. Copy .env.example to .env and paste your key."
                )
            _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=180)
        return _client


def _acquire_rate_slot() -> None:
    global _next_allowed_time
    with _rate_lock:
        now = time.monotonic()
        wait = max(0.0, _next_allowed_time - now)
        _next_allowed_time = max(now, _next_allowed_time) + _MIN_INTERVAL
    if wait > 0:
        time.sleep(wait)


def chat(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 3500,
    model: str | None = None,
) -> str:
    """One chat call with shared rate limiting, 429 retry, and automatic 404/410 model fallback.
    Always reads the final answer strictly from message.content (never from reasoning_content).
    """
    candidates = _candidate_models(model)
    last_error = None
    for chosen_model in candidates:
        for attempt in range(1, 4):
            _acquire_rate_slot()
            try:
                response = client().chat.completions.create(
                    model=chosen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = response.choices[0].message
                # Strictly read message.content and never reasoning_content
                content = msg.content if isinstance(getattr(msg, "content", None), str) else ""
                if content.strip():
                    return content
                last_error = RuntimeError(f"Model '{chosen_model}' returned empty message.content")
                break  # try next fallback model if content was empty
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                time.sleep(5 * attempt)
            except APIStatusError as e:
                last_error = e
                if e.status_code in (400, 404, 410):
                    if e.status_code in (404, 410):
                        _dead_models.add(chosen_model)
                    print(
                        f"[llm] Model '{chosen_model}' returned HTTP {e.status_code}; "
                        "switching to fallback model.",
                        flush=True,
                    )
                    break  # move to next model in candidates
                raise
    raise RuntimeError(f"The NVIDIA API could not complete the request: {last_error}")


def check_models_on_startup() -> dict:
    """Verify configured models on startup and switch past any 404/410 retired models."""
    status = {"primary": None, "fast": None, "fallbacks": model_fallbacks()}
    for role, target in [("primary", model_name()), ("fast", fast_model_name())]:
        if role == "fast" and target == status["primary"]:
            status["fast"] = status["primary"]
            continue
        for cand in _candidate_models(target):
            try:
                _acquire_rate_slot()
                resp = client().chat.completions.create(
                    model=cand,
                    messages=[{"role": "user", "content": "Reply OK"}],
                    temperature=0.1,
                    max_tokens=128,
                )
                msg = resp.choices[0].message
                content = msg.content if isinstance(getattr(msg, "content", None), str) else ""
                if content.strip():
                    status[role] = cand
                    break
            except APIStatusError as e:
                if e.status_code in (400, 404, 410):
                    if e.status_code in (404, 410):
                        _dead_models.add(cand)
                    print(f"[llm] Startup check: '{cand}' returned HTTP {e.status_code}, trying fallback...", flush=True)
                    continue
                break
            except Exception as e:
                print(f"[llm] Startup check warning for '{cand}': {e}", flush=True)
                break
    if status["primary"] and status["primary"] != os.getenv("NVIDIA_MODEL"):
        os.environ["NVIDIA_MODEL"] = status["primary"]
    if status["fast"] and status["fast"] != os.getenv("NVIDIA_FAST_MODEL"):
        os.environ["NVIDIA_FAST_MODEL"] = status["fast"]
    print(
        f"[llm] Active models -> NVIDIA_MODEL={model_name()} | "
        f"NVIDIA_FAST_MODEL={fast_model_name()} | fallbacks={model_fallbacks()}",
        flush=True,
    )
    return status


def parse_json(text: str) -> dict:
    """Pull a JSON object out of a model reply, even if it added ```json fences or chatter."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in the reply")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------- long papers

CHUNK_SYSTEM = (
    "You are a careful research assistant. You read part of a research paper and write "
    "compact, factual notes. Never invent details that are not in the text."
)


def summarize_chunk(chunk: str) -> str:
    return chat(
        CHUNK_SYSTEM,
        "Write notes (max 350 words) on this part of a research paper. Keep the problem, "
        "the method, key numbers, results and any limitations. Plain text, no preamble.\n\n"
        + chunk,
        max_tokens=1500,
        model=fast_model_name(),
    )


# ---------------------------------------------------------------- video script

SCRIPT_SYSTEM = (
    "You are an excellent teacher who explains research papers to a smart high-school "
    "student in short explainer videos. You are accurate, concrete and friendly. You use "
    "everyday analogies, define every technical term the first time you use it, and never "
    "invent results that are not in the paper. You reply with JSON only."
)

STYLE_INSTRUCTIONS = {
    "default": (
        "- Target audience: a smart high-school student.\n"
        "- Use 7 to 9 scenes progressing through the paper's specific narrative: concrete problem context, "
        "why it matters, core technical innovation, method/architecture pipeline (1-2 scenes), "
        "measured experimental results, boundaries/limitations, and practical big-picture implications.\n"
        "- Choose the visual type that best explains each scene. Use at least 4 diagrams across the video."
    ),
    "simpler": (
        "- STYLE: SIMPLER. Target audience: a curious middle-schooler or complete beginner.\n"
        "- Use 6 to 7 scenes. Use vivid everyday analogies, zero unexplained jargon, and short friendly sentences (35-55 words per narration).\n"
        "- Progress from real-world analogy to the mechanism to practical takeaway.\n"
        "- Choose simple visual diagrams (flow, compare, cycle, bar_chart) that make the idea intuitive."
    ),
    "technical": (
        "- STYLE: MORE TECHNICAL. Target audience: a graduate researcher or engineer.\n"
        "- Use 7 to 9 scenes covering problem formulation, architecture/method details, experimental setup, "
        "quantitative benchmarks, ablations, technical limitations, and open engineering challenges.\n"
        "- Use accurate domain terminology and include concrete numbers from the paper."
    ),
    "short": (
        "- STYLE: VERY SHORT. Make a concise 60-second lightning summary with EXACTLY 4 scenes: "
        "(1) Core Bottleneck, (2) Method Mechanism, (3) Concrete Measured Outcome, (4) Broader Significance.\n"
        "- Keep each scene's narration to 2 brisk sentences (25-40 words)."
    ),
}

SCRIPT_PROMPT = """Turn the research paper material below into an explainer video script.

Return ONE JSON object with exactly this shape:
{{
  "title": "short plain-English title for the video (under 8 words)",
  "summary": "one sentence: what this paper did and why it matters",
  "takeaways": ["3 short key takeaways from the paper, max 12 words each"],
  "scenes": [
    {{
      "title": "specific slide title, max 7 words",
      "source": "section or page in paper (e.g. 'Section 3.2' or 'Pages 4-5')",
      "bullets": ["3 or 4 concrete facts, max 12 words each"],
      "narration": "what the narrator says, 2-4 spoken sentences (40-70 words), conversational",
      "visual": {{
        "type": "flow | compare | bar_chart | cycle | none"
      }}
    }}
  ],
  "glossary": [{{"term": "hard word from the paper", "meaning": "simple definition"}}]
}}

Visual types (choose the one that best explains each scene, and vary them across the video):
1. "flow": a pipeline, method, or architecture (3 to 5 nodes):
   {{"type": "flow", "nodes": ["Step A", "Step B", "Step C"], "edges": [{{"from": 0, "to": 1, "label": "optional"}}, {{"from": 1, "to": 2}}]}}
2. "compare": contrasting old way vs new way (3 to 4 items per side):
   {{"type": "compare", "left_title": "Previous Work", "left_items": ["Item 1", "Item 2", "Item 3"], "right_title": "This Paper", "right_items": ["Item 1", "Item 2", "Item 3"]}}
3. "bar_chart": 3 to 6 quantitative results explicitly stated in the paper:
   {{"type": "bar_chart", "unit": "%", "bars": [{{"label": "Baseline", "value": 42.0}}, {{"label": "Method A", "value": 55.0}}, {{"label": "Ours", "value": 68.0}}]}}
4. "cycle": 3 to 5 repeating or iterative steps in a loop:
   {{"type": "cycle", "steps": ["Collect", "Train", "Evaluate", "Refine"]}}
5. "none": bullets only:
   {{"type": "none"}}

STRICT SCRIPT RULES:
{style_rules}
- UNIQUE & SPECIFIC SCENE TITLES:
  * Every scene title must be 100% unique across the script. Never create two scenes with the same or near-duplicate title.
  * Every scene title must name the actual concept, method, place, metric, or finding (e.g., "Transformer Attention Layers", "4.2x Speedup on GPUs", "Edge-Case Failures in Low Light").
  * NEVER use generic labels like "The Results", "Results", "Introduction", "Overview", "Method", "How It Works", "Limitations", "Takeaway", "Takeaways", "3 Key Takeaways", or "Conclusion".
  * Never use the paper's title or overall video title as a scene title.
- NO CROSS-SCENE REPETITION:
  * No sentence, claim, or distinctive bullet may appear in more than one scene.
  * Every scene must cover distinct, non-overlapping information.
- FINAL SCENE INSIGHT:
  * The final scene must add a practical or big-picture insight (e.g. real-world implications, what this unlocks next, or open questions) instead of restating earlier points or summarizing takeaways.
- NARRATION MUST NOT BEGIN WITH THE TITLE:
  * Narration must NEVER start by repeating or paraphrasing the slide title (e.g., do NOT start with "In [Title]...", "[Title] explains...", or "[Title] is...").
  * Dive directly into the conversational narrative or explanation.
- BULLETS MUST BE CONCRETE FACTS:
  * Every bullet must contain concrete facts from the paper (specific names, numbers, datasets, models, metrics, or case studies).
  * NEVER use generic filler bullets like "Limited empirical data", "Future work is needed", "Better performance", or "Novel approach".
- HONEST NUMBERS ONLY:
  * Quote exact numbers if they appear in the paper. If the paper has no numbers or is theoretical/qualitative, explicitly say so instead of inventing any numbers or statistics.
- SOURCE ATTRIBUTION:
  * Include a "source" field for every scene citing the specific section or page range in the paper it is drawn from (e.g., "Section 1", "Section 3.2", "Section 5.1 & Table 2", "Pages 4-6").
- Keep all diagram node labels, bar labels, cycle steps, and compare items under 4 words.
- Narration is spoken aloud: no bullet symbols, no markdown, no formulas. Say numbers in words where natural.
- Give 3 takeaways and 4 to 6 glossary terms.
- Output the JSON object and nothing else.

PAPER MATERIAL:
{material}
"""


def _short_label(val, max_words: int = 4, max_chars: int = 28) -> str:
    if isinstance(val, dict):
        val = val.get("label") or val.get("name") or val.get("text") or val.get("title") or ""
    text = re.sub(r"[*_#`]", "", str(val)).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def clean_visual(v) -> dict:
    """Validate and normalize a scene's visual object, falling back to {'type': 'none'}."""
    if not isinstance(v, dict):
        return {"type": "none"}
    vtype = str(v.get("type", "none")).strip().lower()

    if vtype == "flow":
        raw_nodes = v.get("nodes")
        if not isinstance(raw_nodes, list):
            return {"type": "none"}
        nodes, id_map = [], {}
        for idx, item in enumerate(raw_nodes[:6]):
            lbl = _short_label(item, 4, 26)
            if not lbl:
                continue
            node_idx = len(nodes)
            nodes.append(lbl)
            id_map[str(idx)] = node_idx
            id_map[lbl.lower()] = node_idx
            if isinstance(item, dict) and item.get("id") is not None:
                id_map[str(item["id"]).lower()] = node_idx
        if len(nodes) < 2:
            return {"type": "none"}

        def _resolve(ref):
            if isinstance(ref, int) and 0 <= ref < len(nodes):
                return ref
            key = str(ref).strip().lower()
            if key in id_map:
                return id_map[key]
            if key.isdigit() and 0 <= int(key) < len(nodes):
                return int(key)
            return None

        edges = []
        raw_edges = v.get("edges") if isinstance(v.get("edges"), list) else []
        for e in raw_edges[:8]:
            if not isinstance(e, dict):
                continue
            u = _resolve(e.get("from"))
            w = _resolve(e.get("to"))
            if u is not None and w is not None and u != w:
                elbl = _short_label(e.get("label", ""), 3, 18)
                edges.append({"from": u, "to": w, "label": elbl})
        if not edges:
            edges = [{"from": i, "to": i + 1, "label": ""} for i in range(len(nodes) - 1)]
        return {"type": "flow", "nodes": nodes, "edges": edges}

    if vtype == "compare":
        left_title = _short_label(v.get("left_title") or "Existing Approach", 4, 24)
        right_title = _short_label(v.get("right_title") or "Paper's Approach", 4, 24)
        raw_left = v.get("left_items")
        raw_right = v.get("right_items")
        if isinstance(v.get("columns"), list) and len(v["columns"]) >= 2:
            c0, c1 = v["columns"][0], v["columns"][1]
            if isinstance(c0, dict) and isinstance(c1, dict):
                left_title = _short_label(c0.get("title") or left_title, 4, 24)
                right_title = _short_label(c1.get("title") or right_title, 4, 24)
                raw_left = c0.get("items")
                raw_right = c1.get("items")
        if not isinstance(raw_left, list) or not isinstance(raw_right, list):
            return {"type": "none"}
        left_items = [_short_label(x, 4, 28) for x in raw_left if _short_label(x, 4, 28)][:4]
        right_items = [_short_label(x, 4, 28) for x in raw_right if _short_label(x, 4, 28)][:4]
        if len(left_items) < 2 or len(right_items) < 2:
            return {"type": "none"}
        return {
            "type": "compare",
            "left_title": left_title,
            "left_items": left_items,
            "right_title": right_title,
            "right_items": right_items,
        }

    if vtype == "bar_chart":
        raw_bars = v.get("bars") or v.get("items") or v.get("data")
        if not isinstance(raw_bars, list):
            return {"type": "none"}
        bars = []
        for b in raw_bars[:6]:
            if not isinstance(b, dict):
                continue
            lbl = _short_label(b.get("label") or b.get("name", ""), 4, 22)
            raw_val = b.get("value")
            if raw_val is None or not lbl:
                continue
            try:
                num_str = re.sub(r"[^0-9.\-]", "", str(raw_val))
                val = float(num_str)
            except ValueError:
                continue
            if val == val and abs(val) != float("inf"):
                bars.append({"label": lbl, "value": val})
        if len(bars) < 3:
            return {"type": "none"}
        unit = _short_label(v.get("unit", ""), 2, 8)
        return {"type": "bar_chart", "unit": unit, "bars": bars}

    if vtype == "cycle":
        raw_steps = v.get("steps") or v.get("nodes") or v.get("items")
        if not isinstance(raw_steps, list):
            return {"type": "none"}
        steps = [_short_label(s, 4, 24) for s in raw_steps if _short_label(s, 4, 24)][:5]
        if len(steps) < 3:
            return {"type": "none"}
        return {"type": "cycle", "steps": steps}

    return {"type": "none"}


def clean_script(data: dict) -> dict:
    scenes = []
    for i, s in enumerate(data.get("scenes", [])):
        title = re.sub(r"[*_#`]", "", str(s.get("title", ""))).strip()
        source = re.sub(r"[*_#`]", "", str(s.get("source", ""))).strip()
        narration = re.sub(r"[*_#`]", "", str(s.get("narration", ""))).strip()
        bullets = [re.sub(r"[*_#`]", "", str(b)).strip() for b in s.get("bullets", []) if str(b).strip()][:4]
        visual = clean_visual(s.get("visual"))
        if title and narration:
            scenes.append(
                {
                    "title": title,
                    "source": source or f"Section {i + 1}",
                    "bullets": bullets or [title],
                    "narration": narration,
                    "visual": visual,
                }
            )
    if len(scenes) < 3:
        raise ValueError("The script had too few scenes")

    raw_takeaways = data.get("takeaways", []) if isinstance(data.get("takeaways"), list) else []
    takeaways = [re.sub(r"[*_#`]", "", str(t)).strip() for t in raw_takeaways if str(t).strip()][:3]
    if len(takeaways) < 3:
        for b in scenes[-1]["bullets"] + scenes[0]["bullets"]:
            if b not in takeaways and len(takeaways) < 3:
                takeaways.append(b)
    while len(takeaways) < 3:
        takeaways.append(str(data.get("title", "Key insight from this research")).strip())

    glossary = []
    for g in data.get("glossary", [])[:6]:
        term, meaning = str(g.get("term", "")).strip(), str(g.get("meaning", "")).strip()
        if term and meaning:
            glossary.append({"term": term, "meaning": meaning})
    return {
        "title": str(data.get("title", "Paper explained")).strip() or "Paper explained",
        "summary": str(data.get("summary", "")).strip(),
        "takeaways": takeaways[:3],
        "scenes": scenes[:10],
        "glossary": glossary,
    }


# ---------------------------------------------------------------- validation & repair

GENERIC_TITLES = {
    "the results", "results", "key results", "the result",
    "takeaway", "takeaways", "key takeaways", "the takeaway", "3 key takeaways",
    "the limitations", "limitations", "limitation",
    "conclusion", "conclusions", "the conclusion",
    "the method", "method", "methods", "the methodology", "methodology",
    "how it works", "the key idea", "key idea", "the problem", "problem",
    "why it matters", "overview", "introduction", "background", "summary",
}

GENERIC_BULLET_PHRASES = [
    "limited empirical data",
    "more research needed",
    "future work is needed",
    "future work needed",
    "further research needed",
    "better performance",
    "novel approach",
]


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/markdown, and collapse whitespace."""
    t = re.sub(r"[*_#`\"'.,;:!?()[\]{}]", " ", str(text).lower())
    return " ".join(t.split())


def text_similarity(s1: str, s2: str) -> float:
    """Compute token Jaccard similarity and character sequence ratio."""
    import difflib
    n1, n2 = normalize_text(s1), normalize_text(s2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    w1, w2 = set(n1.split()), set(n2.split())
    jaccard = len(w1 & w2) / max(1, len(w1 | w2))
    return max(ratio, jaccard)


def detect_duplicate_scenes(scenes: list[dict], paper_title: str = "") -> dict[int, list[str]]:
    """Detect duplicate or near-duplicate titles and bullets, generic titles/bullets,
    narration repeating the slide title, and cross-scene repeated sentences.
    Returns a mapping of scene_index -> list of issue descriptions.
    """
    issues: dict[int, list[str]] = {}

    def _flag(idx: int, reason: str):
        issues.setdefault(idx, []).append(reason)

    norm_paper_title = normalize_text(paper_title)

    # 1. Title validation
    for i, s in enumerate(scenes):
        t_i = s.get("title", "")
        norm_ti = normalize_text(t_i)

        # Check generic titles
        if norm_ti in GENERIC_TITLES or norm_ti.startswith("the results") or norm_ti.endswith("takeaways"):
            _flag(i, f"Title '{t_i}' is a generic label; must name a specific concept, place, method or number.")

        # Check repeating overall paper title
        if norm_paper_title and (norm_ti == norm_paper_title or text_similarity(t_i, paper_title) >= 0.85):
            _flag(i, f"Title '{t_i}' repeats the paper title.")

        # Check exact or near duplicates with previous or next scenes
        for j in range(i + 1, len(scenes)):
            t_j = scenes[j].get("title", "")
            norm_tj = normalize_text(t_j)
            if norm_ti == norm_tj:
                _flag(j, f"Title is an exact duplicate of scene {i + 1}: '{t_j}'")
            elif text_similarity(t_i, t_j) >= 0.75:
                _flag(j, f"Title '{t_j}' is a near-duplicate of scene {i + 1} '{t_i}'")

    # 2. Narration repeating the slide title at the start
    for i, s in enumerate(scenes):
        t_i = s.get("title", "")
        narr = s.get("narration", "")
        norm_ti = normalize_text(t_i)
        norm_narr = normalize_text(narr)
        if norm_ti and norm_narr:
            if norm_narr.startswith(norm_ti):
                _flag(i, f"Narration starts by repeating slide title '{t_i}'.")
            elif any(norm_narr.startswith(p + norm_ti) for p in ["in ", "on ", "about ", "this section covers ", "this scene covers "]):
                _flag(i, f"Narration starts by repeating slide title '{t_i}'.")

    # 3. Bullets validation (generic bullets & duplicate bullets)
    all_bullets = []
    for i, s in enumerate(scenes):
        for b in s.get("bullets", []):
            norm_b = normalize_text(b)
            all_bullets.append((i, norm_b, str(b).strip()))
            for g_phrase in GENERIC_BULLET_PHRASES:
                if g_phrase in norm_b or norm_b == g_phrase:
                    _flag(i, f"Bullet '{b}' is generic filler; must contain concrete facts from the paper.")
                    break

    for idx1 in range(len(all_bullets)):
        s1, norm1, raw1 = all_bullets[idx1]
        for idx2 in range(idx1 + 1, len(all_bullets)):
            s2, norm2, raw2 = all_bullets[idx2]
            if s1 == s2:
                if norm1 == norm2:
                    _flag(s2, f"Duplicate bullet within same scene: '{raw2}'")
            else:
                if norm1 == norm2:
                    _flag(s2, f"Bullet in scene {s2 + 1} duplicates scene {s1 + 1}: '{raw2}'")
                elif text_similarity(raw1, raw2) >= 0.8 and len(norm1.split()) >= 4:
                    _flag(s2, f"Bullet in scene {s2 + 1} is near-duplicate of scene {s1 + 1}: '{raw2}'")

    # 4. Cross-scene narration sentence duplication (5+ words)
    scene_sentences = []
    for i, s in enumerate(scenes):
        raw_narr = s.get("narration", "")
        sents = [sent.strip() for sent in re.split(r"[.!?]+", raw_narr) if len(sent.strip().split()) >= 5]
        for sent in sents:
            scene_sentences.append((i, normalize_text(sent), sent))

    for idx1 in range(len(scene_sentences)):
        s1, norm1, raw1 = scene_sentences[idx1]
        for idx2 in range(idx1 + 1, len(scene_sentences)):
            s2, norm2, raw2 = scene_sentences[idx2]
            if s1 != s2:
                if norm1 == norm2 or text_similarity(raw1, raw2) >= 0.85:
                    _flag(s2, f"Sentence in scene {s2 + 1} narration repeats scene {s1 + 1}: '{raw2}'")

    return issues


REPAIR_SYSTEM = (
    "You are a meticulous research paper editor. You repair specific scenes in an explainer video "
    "script that failed quality checks (such as duplicate titles, generic bullets, or narration repeating titles). "
    "Every repaired scene must be concrete, unique, and strictly grounded in the paper. Reply with JSON only."
)

REPAIR_PROMPT = """We have an explainer video script for the paper "{paper_title}".
The following scenes have quality issues (duplicates, generic titles/bullets, or narration repeating the title):

VALID ACCEPTED SCENES (do NOT duplicate their titles, bullets, or sentences):
{accepted_scenes_summary}

SCENES REQUIRING REPAIR:
{scenes_to_repair}

PAPER MATERIAL:
{material}

INSTRUCTIONS:
1. Return a JSON object containing replacements ONLY for the scenes requiring repair.
2. For each repaired scene:
   - "index": the exact scene index (integer)
   - "title": a unique, specific title naming the exact concept, method, dataset, or finding (NEVER generic like 'The Results' or 'Takeaways')
   - "source": section or page in the paper (e.g. 'Section 3.2')
   - "bullets": 3 or 4 concrete facts (names, numbers, specific cases from the paper)
   - "narration": 2-4 spoken sentences (40-70 words) that do NOT start with the title
   - "visual": diagram object ({{"type": "flow | compare | bar_chart | cycle | none"}})
3. If repairing the final scene, add practical or big-picture insights instead of restating earlier takeaways.

Return JSON with this exact shape:
{{
  "repaired_scenes": [
    {{
      "index": 1,
      "title": "...",
      "source": "...",
      "bullets": ["...", "...", "..."],
      "narration": "...",
      "visual": {{"type": "none"}}
    }}
  ]
}}
"""


def _programmatic_sanitize(scenes: list[dict], paper_title: str = "") -> None:
    """Fallback defensive pass ensuring zero lingering duplicates or title-repeating narrations."""
    seen_titles: dict[str, int] = {}
    for i, s in enumerate(scenes):
        title = s.get("title", "").strip()
        norm_t = normalize_text(title)

        if norm_t in seen_titles:
            seen_titles[norm_t] += 1
            s["title"] = f"{title} (Part {seen_titles[norm_t]})"
        else:
            seen_titles[norm_t] = 1

        narr = s.get("narration", "").strip()
        norm_narr = normalize_text(narr)
        if norm_narr.startswith(norm_t) and len(narr) > len(title):
            stripped = narr[len(title):].lstrip(" :,.-—")
            if stripped and len(stripped.split()) >= 8:
                s["narration"] = stripped[0].upper() + stripped[1:]

        if not s.get("source"):
            s["source"] = f"Section {i + 1}"


def validate_and_repair_script(script: dict, material: str) -> dict:
    """Post-validation: detect duplicate/generic titles & bullets, and send ONE repair call
    using the stronger model (NVIDIA_MODEL) fixing only the flagged scenes."""
    scenes = script.get("scenes", [])
    paper_title = script.get("title", "")
    issues = detect_duplicate_scenes(scenes, paper_title)
    if not issues:
        return script

    bad_indices = sorted(issues.keys())
    print(f"[llm] Quality issues detected in {len(bad_indices)} scene(s) {bad_indices}; running targeted repair call.", flush=True)

    accepted = []
    for i, s in enumerate(scenes):
        if i not in issues:
            bullets_preview = ", ".join(s.get("bullets", [])[:2])
            accepted.append(f"- Scene {i} [title: '{s.get('title')}']: {bullets_preview}")
    accepted_summary = "\n".join(accepted) if accepted else "(none accepted yet)"

    to_repair = []
    for i in bad_indices:
        s = scenes[i]
        reasons = "; ".join(issues[i])
        to_repair.append(
            f"Scene {i}:\n"
            f"  Current Title: {s.get('title')}\n"
            f"  Current Bullets: {s.get('bullets')}\n"
            f"  Current Narration: {s.get('narration')}\n"
            f"  Current Source: {s.get('source', '')}\n"
            f"  Issues to fix: {reasons}"
        )
    to_repair_text = "\n\n".join(to_repair)

    context = material[:25_000]
    prompt = REPAIR_PROMPT.format(
        paper_title=paper_title,
        accepted_scenes_summary=accepted_summary,
        scenes_to_repair=to_repair_text,
        material=context,
    )

    try:
        reply = chat(
            REPAIR_SYSTEM,
            prompt,
            temperature=0.3,
            max_tokens=4000,
            model=model_name(),  # Stronger model
        )
        parsed = parse_json(reply)
        repaired_list = parsed.get("repaired_scenes") or parsed.get("scenes") or []
        for rep in repaired_list:
            if not isinstance(rep, dict):
                continue
            idx = rep.get("index")
            if idx is not None and isinstance(idx, int) and 0 <= idx < len(scenes):
                t = str(rep.get("title", "")).strip()
                n = re.sub(r"[*_#`]", "", str(rep.get("narration", ""))).strip()
                b = [str(x).strip() for x in rep.get("bullets", []) if str(x).strip()][:4]
                src = str(rep.get("source", "")).strip() or scenes[idx].get("source", f"Section {idx + 1}")
                v = clean_visual(rep.get("visual"))
                if t and n:
                    scenes[idx] = {
                        "title": t,
                        "source": src,
                        "bullets": b or [t],
                        "narration": n,
                        "visual": v,
                    }
    except Exception as e:
        print(f"[llm] Repair call encountered error ({e}); applying programmatic sanitization.", flush=True)

    _programmatic_sanitize(scenes, paper_title)
    script["scenes"] = scenes
    return script


def make_script(material: str, style: str = "default") -> dict:
    style_rules = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["default"])
    prompt = SCRIPT_PROMPT.format(material=material, style_rules=style_rules)
    problem = ""
    for _ in range(3):
        # Explicitly use the stronger model (NVIDIA_MODEL) for script generation
        reply = chat(SCRIPT_SYSTEM, prompt + problem, temperature=0.4, max_tokens=8000, model=model_name())
        try:
            raw_script = clean_script(parse_json(reply))
            return validate_and_repair_script(raw_script, material)
        except (ValueError, json.JSONDecodeError, AttributeError) as e:
            problem = (
                f"\n\nYour previous reply could not be used ({e}). "
                "Reply again with ONLY the valid JSON object."
            )
    raise RuntimeError("The AI did not return a usable script after 3 tries. Please try again.")


# ---------------------------------------------------------------- paper chat

CHAT_SYSTEM = (
    "You are a helpful research assistant answering questions about a specific paper. "
    "Use ONLY the provided paper text to answer. Never invent facts, numbers, or claims "
    "that are not in the paper text. If the paper does not contain the answer, clearly "
    "state that the paper does not mention it. Keep answers clear, friendly, and concise."
)


def _select_paper_context(paper_text: str, question: str, max_chars: int = 32_000) -> str:
    if len(paper_text) <= max_chars:
        return paper_text
    words = {w for w in re.findall(r"[a-zA-Z0-9]{3,}", question.lower())}
    head = paper_text[:8_000]
    rest = paper_text[8_000:]
    chunk_size = 4_000
    chunks = [rest[i : i + chunk_size] for i in range(0, len(rest), chunk_size)]
    scored = sorted(
        enumerate(chunks),
        key=lambda item: sum(item[1].lower().count(w) for w in words),
        reverse=True,
    )
    chosen = [c for _, c in sorted(scored[:6], key=lambda x: x[0])]
    return head + "\n\n...\n\n" + "\n\n...\n\n".join(chosen)


def answer_question(paper_text: str, question: str, history: list | None = None) -> str:
    context = _select_paper_context(paper_text, question)
    conv = ""
    if history:
        recent = history[-6:]
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            conv = "RECENT CONVERSATION:\n" + "\n".join(lines) + "\n\n"

    user_prompt = (
        f"PAPER TEXT:\n{context}\n\n"
        f"{conv}"
        f"QUESTION: {question.strip()}\n\n"
        "Answer using only the paper text above:"
    )
    reply = chat(CHAT_SYSTEM, user_prompt, temperature=0.2, max_tokens=2000).strip()
    if not reply:
        raise RuntimeError("The AI returned an empty response. Please try asking again.")
    return reply

