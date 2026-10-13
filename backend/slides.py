"""Draws each scene as a 1280x720 slide image with Pillow."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (15, 27, 51)
INK = (238, 243, 250)
MUTED = (139, 157, 186)
MARK = (255, 225, 77)
TRACK = (30, 45, 78)

_FONT_FILES = {
    True: [  # bold
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    False: [
        "DejaVuSans.ttf",
        "arial.ttf",
        "segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}
_font_cache = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _font_cache:
        loaded = None
        for name in _FONT_FILES[bold]:
            try:
                loaded = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        _font_cache[key] = loaded or ImageFont.load_default(size=size)
    return _font_cache[key]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def shorten(draw, text: str, fnt, max_width: int) -> str:
    if draw.textlength(text, font=fnt) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=fnt) > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


def render_slide(scene: dict, index: int, total: int, paper_title: str, out_path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, H], fill=MARK)

    margin = 96
    max_w = W - margin * 2

    # Title: shrink until it fits on two lines
    size = 58
    while True:
        title_font = font(size, bold=True)
        lines = wrap(d, scene["title"], title_font, max_w)
        if len(lines) <= 2 or size <= 40:
            break
        size -= 4
    y = 84
    for line in lines:
        d.text((margin, y), line, font=title_font, fill=INK)
        y += int(size * 1.2)
    y += 26
    d.rectangle([margin, y, margin + 90, y + 6], fill=MARK)
    y += 46

    visual = scene.get("visual") or {}
    fig_path = Path(visual["path"]) if visual.get("type") == "figure" and visual.get("path") else None
    has_fig = fig_path is not None and fig_path.exists()
    bullet_max_w = 420 if has_fig else max_w

    # Bullets: shrink until they all fit above the footer
    bottom = H - 110
    bsize = 30 if has_fig else 36
    while True:
        bullet_font = font(bsize)
        line_h = int(bsize * 1.35)
        gap = int(bsize * 0.6)
        layout, yy = [], y
        for bullet in scene["bullets"]:
            bl = wrap(d, bullet, bullet_font, bullet_max_w - 48)
            layout.append((yy, bl))
            yy += len(bl) * line_h + gap
        if yy - gap <= bottom or bsize <= 22:
            break
        bsize -= 2

    for top, bl in layout:
        r = max(5, bsize // 6)
        cy = top + int(bsize * 0.7)
        d.ellipse([margin, cy - r, margin + 2 * r, cy + r], fill=MARK)
        for i, text in enumerate(bl):
            d.text((margin + 48, top + i * line_h), text, font=bullet_font, fill=INK)

    if has_fig:
        try:
            with Image.open(fig_path) as raw_fig:
                fig_img = raw_fig.convert("RGB")
                box_x, box_y, box_w, box_h = 560, 190, 624, 430
                d.rounded_rectangle(
                    [box_x, box_y, box_x + box_w, box_y + box_h],
                    radius=10,
                    fill=(245, 248, 252),
                    outline=MARK,
                    width=3,
                )
                fig_img.thumbnail((box_w - 28, box_h - 54), Image.Resampling.LANCZOS)
                fx = box_x + (box_w - fig_img.width) // 2
                fy = box_y + 14 + (box_h - 54 - fig_img.height) // 2
                img.paste(fig_img, (fx, fy))
                cap = shorten(d, visual.get("caption", "Figure from paper"), font(18, bold=True), box_w - 32)
                cw = d.textlength(cap, font=font(18, bold=True))
                d.text((box_x + (box_w - cw) / 2, box_y + box_h - 32), cap, font=font(18, bold=True), fill=BG)
        except Exception:
            pass

    # Footer: scene counter, paper title, progress bar
    small = font(22)
    d.text((margin, H - 62), f"{index + 1} / {total}", font=small, fill=MUTED)
    label = shorten(d, paper_title, small, 700)
    d.text((W - margin - d.textlength(label, font=small), H - 62), label, font=small, fill=MUTED)
    d.rectangle([0, H - 10, W, H], fill=TRACK)
    d.rectangle([0, H - 10, int(W * (index + 1) / total), H], fill=MARK)

    img.save(out_path, "PNG")


def render_all(script: dict, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = script["scenes"]
    paths = []
    for i, scene in enumerate(scenes):
        path = out_dir / f"slide_{i:02d}.png"
        render_slide(scene, i, len(scenes), script["title"], path)
        paths.append(path)
    return paths
