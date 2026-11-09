"""Builds the wsplab logo files in this folder.

The mark is a 5x5 grid of weights: the lit ones draw a W and the centre one is the edited weight. The wordmark is
DM Mono (SIL Open Font License), converted to outlines so the SVGs need no font. Fetch the two font files first:

    https://github.com/google/fonts/raw/main/ofl/dmmono/DMMono-Medium.ttf
    https://github.com/google/fonts/raw/main/ofl/dmmono/DMMono-Light.ttf

Run: python docs/logo/make_logo.py <folder with the two .ttf files>   (needs fontTools and Pillow)
"""

from __future__ import annotations

import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
THEMES = {"light": {"ink": "#16181D", "bg": "#F6F6F3", "accent": "#2563EB"},
          "dark": {"ink": "#F3F3EF", "bg": "#16181D", "accent": "#3B82F6"}}  # a lighter blue reads better on dark
LIT = {(0, 0), (0, 4), (1, 0), (1, 4), (2, 0), (2, 4), (3, 0), (3, 1), (3, 3), (3, 4), (4, 0), (4, 4)}
EDITED = (2, 2)
FADED = 0.12  # opacity of the weights that are off
LAB = 0.6  # opacity of "lab" in the wordmark
TRACKING = -0.05  # letter spacing, in em
CELL, STEP, RADIUS = 10, 13, 2.5  # on a 64-unit grid


def cells():
    for r in range(5):
        for c in range(5):
            yield (r, c), 1 + STEP * c, 1 + STEP * r


def mark_svg(ink: str, accent: str, x: float = 0, y: float = 0) -> str:
    rects = []
    for rc, cx, cy in cells():
        if rc == EDITED:
            style = f'fill="{accent}"'
        elif rc in LIT:
            style = f'fill="{ink}"'
        else:
            style = f'fill="{ink}" fill-opacity="{FADED}"'
        rects.append(f'<rect x="{x + cx:g}" y="{y + cy:g}" width="{CELL}" height="{CELL}" rx="{RADIUS}" {style}/>')
    return "\n  ".join(rects)


def svg(width: float, height: float, body: str, label: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" width="{width:g}" '
            f'height="{height:g}" role="img" aria-label="{label}">\n  <title>{label}</title>\n  {body}\n</svg>\n')


def text_paths(runs, size: float, x: float):
    """Outlines for [(font, text), ...] on baseline y=0. Returns [(d, bounds)] per run and the end x."""
    out, cursor = [], x
    for font, text in runs:
        glyphs, cmap, upem = font.getGlyphSet(), font.getBestCmap(), font["head"].unitsPerEm
        scale = size / upem
        svg_pen, bounds_pen = SVGPathPen(glyphs), BoundsPen(glyphs)
        for ch in text:
            name = cmap[ord(ch)]
            transform = (scale, 0, 0, -scale, cursor, 0)
            glyphs[name].draw(TransformPen(svg_pen, transform))
            glyphs[name].draw(TransformPen(bounds_pen, transform))
            cursor += (font["hmtx"][name][0] + TRACKING * upem) * scale
        out.append((svg_pen.getCommands(), bounds_pen.bounds))
    return out, cursor - TRACKING * size


def lockup(medium: TTFont, light: TTFont, ink: str, accent: str) -> str:
    size, gap, pad = 36, 15, 2
    runs, end = text_paths([(medium, "wsp"), (light, "lab")], size, 64 + gap)
    top = min(b[1] for _, b in runs)
    bottom = max(b[3] for _, b in runs)
    shift = 32 - (top + bottom) / 2  # centre the word's ink on the mark
    (wsp, _), (lab, _) = runs
    body = (mark_svg(ink, accent, pad, pad) + "\n  "
            f'<g transform="translate({pad:g} {pad + shift:.3f})" fill="{ink}">'
            f'<path d="{wsp}"/><path d="{lab}" fill-opacity="{LAB}"/></g>')
    return svg(round(end + 2 * pad, 2), 64 + 2 * pad, body, "wsplab")


def hex_rgba(color: str, alpha: float = 1.0):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)) + (round(alpha * 255),)


def draw_mark(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, ink: str, accent: str) -> None:
    """Each colour is drawn as an opaque mask, then composited once: drawing translucent rounded rectangles
    directly double-paints their edges."""
    image, k = draw._image, size / 64
    groups = {"edited": (accent, 1.0), "lit": (ink, 1.0), "faded": (ink, FADED)}
    masks = {name: Image.new("L", image.size, 0) for name in groups}
    for rc, cx, cy in cells():
        name = "edited" if rc == EDITED else "lit" if rc in LIT else "faded"
        box = [x + cx * k, y + cy * k, x + (cx + CELL) * k, y + (cy + CELL) * k]
        ImageDraw.Draw(masks[name]).rounded_rectangle(box, radius=RADIUS * k, fill=255)
    for name, (color, alpha) in groups.items():
        layer = Image.new("RGBA", image.size, hex_rgba(color, 0))
        layer.putalpha(masks[name].point(lambda v, a=alpha: round(v * a)))
        image.alpha_composite(layer)


def png(width: int, height: int, paint, path: Path, bg: str | None, ss: int = 4) -> None:
    image = Image.new("RGBA", (width * ss, height * ss), hex_rgba(bg) if bg else (0, 0, 0, 0))
    paint(ImageDraw.Draw(image, "RGBA"), ss)
    image.resize((width, height), Image.LANCZOS).save(path, optimize=True)


def word(draw, x, y, text, font, color, size):
    """Draws text with the wordmark's tracking, one character at a time. Returns the end x."""
    for ch in text:
        draw.text((x, y), ch, font=font, fill=color, anchor="ls")
        x += font.getlength(ch) + TRACKING * size
    return x


def main(fonts: Path) -> None:
    medium, light = TTFont(fonts / "DMMono-Medium.ttf"), TTFont(fonts / "DMMono-Light.ttf")
    for theme, colors in THEMES.items():
        (HERE / f"mark-{theme}.svg").write_text(svg(64, 64, mark_svg(colors["ink"], colors["accent"]), "wsplab"))
        (HERE / f"logo-{theme}.svg").write_text(lockup(medium, light, colors["ink"], colors["accent"]))

    ink, bg, accent = (THEMES["light"][k] for k in ("ink", "bg", "accent"))
    png(512, 512, lambda d, s: draw_mark(d, 96 * s, 96 * s, 320 * s, ink, accent), HERE / "avatar.png", bg)

    def social(d, s):  # GitHub's social preview size
        mark, size, gap = 200 * s, 120 * s, 44 * s
        f_med = ImageFont.truetype(str(fonts / "DMMono-Medium.ttf"), size)
        f_light = ImageFont.truetype(str(fonts / "DMMono-Light.ttf"), size)
        f_tag = ImageFont.truetype(str(fonts / "DMMono-Light.ttf"), 34 * s)
        text_w = sum(f.getlength(t) for f, t in ((f_med, "wsp"), (f_light, "lab"))) + 5 * TRACKING * size
        x = (1280 * s - (mark + gap + text_w)) / 2
        y = 230 * s
        draw_mark(d, x, y - mark / 2, mark, ink, accent)
        _, top, _, bottom = f_med.getbbox("wsplab", anchor="ls")
        baseline = y - (top + bottom) / 2
        end = word(d, x + mark + gap, baseline, "wsp", f_med, hex_rgba(ink), size)
        word(d, end, baseline, "lab", f_light, hex_rgba(ink, LAB), size)
        d.text((640 * s, 470 * s), "weight-space programming, end to end", font=f_tag,
               fill=hex_rgba(ink, 0.7), anchor="ms")

    png(1280, 640, social, HERE / "social-preview.png", bg)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
