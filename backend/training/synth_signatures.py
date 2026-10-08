"""
Synthetic Signature Generator
=============================
Generates CEDAR-format synthetic signature data from handwriting fonts.

Idea:
  - A "writer" = (romanized Indian name) + (script font) + (personal style:
    slant, stroke thickness, size, underline flourish).
  - GENUINE samples: the writer's own font/style with natural per-signing
    variation (small rotation, elastic warp, ink pressure, stroke wobble).
  - FORGERIES: the SAME name written in a DIFFERENT font/style (a different
    "hand" copying the name), plus tracing-style heavy distortions.

Output matches the CEDAR layout so training/dataset.py works unchanged:
  <out>/<writer_id>/original_<writer_id>_<n>.png
  <out>/<writer_id>/forgeries_<writer_id>_<n>.png

Usage (from backend folder):
  python -m training.synth_signatures --out ../dataset/SYNTH --writers 100
  python -m training.synth_signatures --preview          # 3 writers, quick look
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path("C:/Windows/Fonts")

# Script / handwriting fonts that ship with Windows
SCRIPT_FONTS = [
    "BRADHITC.TTF",   # Bradley Hand ITC
    "BRUSHSCI.TTF",   # Brush Script MT
    "CHILLER.TTF",    # Chiller
    "FREESCPT.TTF",   # Freestyle Script
    "FRSCRIPT.TTF",   # French Script MT
    "GIGI.TTF",       # Gigi
    "Gabriola.ttf",   # Gabriola
    "HARLOWSI.TTF",   # Harlow Solid
    "Inkfree.ttf",    # Ink Free
    "ITCBLKAD.TTF",   # Blackadder ITC
    "ITCEDSCR.TTF",   # Edwardian Script ITC
    "ITCKRIST.TTF",   # Kristen ITC
    "KUNSTLER.TTF",   # Kunstler Script
    "LCALLIG.TTF",    # Lucida Calligraphy
    "LHANDW.TTF",     # Lucida Handwriting
    "MATURASC.TTF",   # Matura MT Script
    "MISTRAL.TTF",    # Mistral
    "MTCORSVA.TTF",   # Monotype Corsiva
    "mvboli.ttf",     # MV Boli
    "PALSCRI.TTF",    # Palace Script MT
    "PRISTINA.TTF",   # Pristina
    "RAGE.TTF",       # Rage Italic
    "SCRIPTBL.TTF",   # Script MT Bold
    "segoepr.ttf",    # Segoe Print
    "segoeprb.ttf",   # Segoe Print Bold
    "segoesc.ttf",    # Segoe Script
    "segoescb.ttf",   # Segoe Script Bold
    "TEMPSITC.TTF",   # Tempus Sans ITC
    "VINERITC.TTF",   # Viner Hand ITC
    "VIVALDII.TTF",   # Vivaldi
    "VLADIMIR.TTF",   # Vladimir Script
]

FIRST_NAMES = [
    "Navya", "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Krishna", "Ishaan", "Rohan", "Kabir", "Ananya", "Diya", "Aadhya", "Kavya",
    "Aanya", "Pari", "Anika", "Riya", "Ishita", "Priya", "Sneha", "Pooja",
    "Neha", "Shreya", "Tanvi", "Meera", "Sanya", "Nisha", "Rahul", "Amit",
    "Vikram", "Suresh", "Rajesh", "Deepak", "Manish", "Sanjay", "Anil", "Ravi",
    "Kiran", "Sunita", "Anjali", "Swati", "Rekha", "Geeta", "Lakshmi", "Sarita",
    "Nikhil", "Varun", "Karan", "Siddharth", "Harsh", "Yash", "Dev", "Om",
    "Tara", "Zara", "Myra", "Ira",
]

LAST_NAMES = [
    "Singhal", "Sharma", "Verma", "Gupta", "Agarwal", "Jain", "Mehta", "Shah",
    "Patel", "Reddy", "Rao", "Nair", "Iyer", "Menon", "Das", "Bose",
    "Chatterjee", "Mukherjee", "Banerjee", "Sen", "Kapoor", "Khanna", "Malhotra",
    "Chopra", "Bhatia", "Arora", "Saxena", "Srivastava", "Tiwari", "Mishra",
    "Pandey", "Dubey", "Joshi", "Kulkarni", "Deshpande", "Patil", "Naidu",
    "Chauhan", "Rathore", "Yadav",
]


def make_signature_text(rng: random.Random) -> str:
    """Compose a name the way people actually sign: varied formats."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    style = rng.random()
    if style < 0.35:
        return f"{first} {last}"
    elif style < 0.55:
        return first
    elif style < 0.70:
        return f"{first[0]}. {last}"
    elif style < 0.85:
        return f"{first.lower()} {last.lower()}"
    else:
        return f"{first} {last[0]}."


def _render_piece(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render a text fragment tightly cropped, white bg / dark ink."""
    dummy = Image.new("L", (8, 8), 255)
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
    img = Image.new("L", (tw + 8, th + 8), 255)
    ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=0)
    return np.array(img)


def _bezier(pts: np.ndarray, n: int = 150) -> np.ndarray:
    """Evaluate a cubic Bezier curve through 4 control points."""
    t = np.linspace(0, 1, n)[:, None]
    return (((1 - t) ** 3) * pts[0] + 3 * ((1 - t) ** 2) * t * pts[1]
            + 3 * (1 - t) * (t ** 2) * pts[2] + (t ** 3) * pts[3])


def _ink_bbox(arr: np.ndarray) -> tuple:
    """Bounding box (x0, y0, x1, y1) of dark pixels."""
    ys, xs = np.where(arr < 200)
    if len(xs) == 0:
        return 0, 0, arr.shape[1], arr.shape[0]
    return xs.min(), ys.min(), xs.max(), ys.max()


def draw_flourish(arr: np.ndarray, kind: str, thickness: int,
                  rng: random.Random) -> np.ndarray:
    """Draw a signature flourish: the sweeps, cuts, and loops real signers add."""
    x0, y0, x1, y1 = _ink_bbox(arr)
    w = max(1, x1 - x0)
    cy = (y0 + y1) // 2

    if kind == "sweep":
        # Long tail: starts under the left, sweeps beneath the name,
        # rises and cuts past the end toward the top-right.
        pts = np.array([
            [x0 + w * 0.05, y1 + rng.randint(5, 25)],
            [x0 + w * 0.45, y1 + rng.randint(25, 55)],
            [x1 - w * 0.10, y1 + rng.randint(0, 25)],
            [x1 + rng.randint(40, 130), y0 - rng.randint(0, 25)],
        ], dtype=np.float64)
        curve = _bezier(pts).astype(np.int32)
        cv2.polylines(arr, [curve], False, 0, thickness, cv2.LINE_AA)

    elif kind == "underline":
        pts = np.array([
            [x0 - rng.randint(5, 30), y1 + rng.randint(8, 20)],
            [x0 + w * 0.35, y1 + rng.randint(15, 35)],
            [x0 + w * 0.65, y1 + rng.randint(5, 25)],
            [x1 + rng.randint(5, 40), y1 + rng.randint(0, 15)],
        ], dtype=np.float64)
        curve = _bezier(pts).astype(np.int32)
        cv2.polylines(arr, [curve], False, 0, thickness, cv2.LINE_AA)

    elif kind == "loop":
        # Looping paraph under the name center
        lx, ly = x0 + w // 2, y1 + rng.randint(10, 22)
        axes = (int(w * rng.uniform(0.2, 0.35)), rng.randint(8, 16))
        cv2.ellipse(arr, (lx, ly), axes, rng.randint(-10, 10),
                    0, rng.randint(280, 360), 0, thickness, cv2.LINE_AA)

    elif kind == "cut":
        # Stroke slashing through the middle of the name
        tilt = rng.randint(10, 35)
        cv2.line(arr, (x0 - rng.randint(5, 25), cy + tilt),
                 (x1 + rng.randint(10, 50), cy - tilt),
                 0, thickness, cv2.LINE_AA)

    return arr


def render_base(writer: dict, rng: random.Random) -> np.ndarray:
    """
    Render a signature the way people actually sign:
      - exaggerated large capital letter
      - rest of the name horizontally squeezed (scrawl effect), overlapping the capital
      - optional truncation (people rarely sign every letter)
      - slant, then a flourish stroke (sweep / underline / loop / cut)
    """
    text = writer["text"]
    cap, rest = text[0], text[1:]
    if writer["truncate"] and len(rest) > 5:
        rest = rest[:rng.randint(3, 5)]

    cap_font = ImageFont.truetype(writer["font"],
                                  int(writer["font_size"] * writer["cap_scale"]))
    cap_img = _render_piece(cap, cap_font)

    pieces = [cap_img]
    if rest.strip():
        rest_font = ImageFont.truetype(writer["font"], writer["font_size"])
        rest_img = _render_piece(rest, rest_font)
        # Horizontal squeeze → compressed, half-legible scrawl
        new_w = max(10, int(rest_img.shape[1] * writer["squeeze"]))
        rest_img = cv2.resize(rest_img, (new_w, rest_img.shape[0]),
                              interpolation=cv2.INTER_AREA)
        pieces.append(rest_img)

    # ── Composite: rest overlaps into the capital, baselines roughly aligned ──
    ch, cw = pieces[0].shape
    pad = 90  # room for flourishes
    if len(pieces) == 2:
        rh, rw = pieces[1].shape
        overlap = int(cw * writer["overlap"])
        total_w = cw + rw - overlap + 2 * pad
        total_h = max(ch, rh) + 2 * pad
        canvas = np.full((total_h, total_w), 255, dtype=np.uint8)
        # capital sits lower-left; rest aligned to its baseline area
        cy0 = pad + max(0, rh - ch) if rh > ch else pad
        canvas[cy0:cy0 + ch, pad:pad + cw] = np.minimum(
            canvas[cy0:cy0 + ch, pad:pad + cw], pieces[0])
        ry0 = pad + max(0, ch - rh) - rng.randint(0, 8) if ch > rh else pad
        ry0 = max(0, ry0)
        rx0 = pad + cw - overlap
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = np.minimum(
            canvas[ry0:ry0 + rh, rx0:rx0 + rw], pieces[1])
    else:
        canvas = np.full((ch + 2 * pad, cw + 2 * pad), 255, dtype=np.uint8)
        canvas[pad:pad + ch, pad:pad + cw] = pieces[0]

    # ── Slant shear ──
    slant = writer["slant"]
    if abs(slant) > 1e-3:
        h, w = canvas.shape
        extra = int(abs(slant) * h) + 1
        M = np.float32([[1, slant, extra if slant < 0 else 0], [0, 1, 0]])
        canvas = cv2.warpAffine(canvas, M, (w + extra, h),
                                flags=cv2.INTER_LINEAR, borderValue=255)

    # ── Flourish (part of the writer's identity) ──
    if writer["flourish"] != "none":
        canvas = draw_flourish(canvas, writer["flourish"],
                               writer["flourish_thickness"], rng)

    return canvas


def elastic_warp(img: np.ndarray, alpha: float, sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Elastic distortion — simulates natural hand wobble."""
    h, w = img.shape
    dx = cv2.GaussianBlur(
        (rng.random((h, w), dtype=np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(
        (rng.random((h, w), dtype=np.float32) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    return cv2.remap(img, x + dx, y + dy,
                     cv2.INTER_LINEAR, borderValue=255)


def vary_sample(base: np.ndarray, rng: np.random.Generator,
                heavy: bool = False) -> np.ndarray:
    """Apply per-signing natural variation to a rendered signature."""
    img = base.copy()

    # Elastic wobble (heavier for traced forgeries)
    alpha = rng.uniform(8, 18) if not heavy else rng.uniform(18, 35)
    img = elastic_warp(img, alpha=alpha, sigma=8.0, rng=rng)

    if heavy:
        # Tracing is slow and hesitant: add high-frequency shake so a
        # traced forgery is distinguishable from a genuine signing's
        # smooth natural variation (small sigma = jittery, not wavy).
        img = elastic_warp(img, alpha=rng.uniform(3, 7), sigma=2.5, rng=rng)

    # Small rotation
    angle = rng.uniform(-2.5, 2.5) if not heavy else rng.uniform(-5, 5)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, rng.uniform(0.95, 1.05))
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

    # Stroke thickness variation (pen + pressure differences)
    k = rng.integers(0, 3)
    if k == 1:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))    # thicker ink
    elif k == 2:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))   # thinner ink

    # Ink pressure / darkness variation
    ink_strength = rng.uniform(0.55, 1.0)
    img = 255 - (255 - img.astype(np.float32)) * ink_strength

    # Pen softness + paper noise
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.8))
    img = img + rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


FLOURISHES = ["sweep", "underline", "loop", "cut", "none"]
FLOURISH_WEIGHTS = [0.35, 0.20, 0.15, 0.10, 0.20]


def build_writer(writer_id: int, rng: random.Random,
                 fonts: list[str], text: str = None) -> dict:
    """Create a writer identity: name + font + personal signing style."""
    return {
        "id": writer_id,
        "text": text or make_signature_text(rng),
        "font": rng.choice(fonts),
        "font_size": rng.randint(70, 110),
        "slant": rng.uniform(-0.05, 0.30),
        "cap_scale": rng.uniform(1.2, 2.1),      # exaggerated capital
        "squeeze": rng.uniform(0.55, 0.95),      # scrawl compression
        "overlap": rng.uniform(0.15, 0.45),      # rest tucks into the capital
        "truncate": rng.random() < 0.30,         # sign only part of the name
        "flourish": rng.choices(FLOURISHES, weights=FLOURISH_WEIGHTS)[0],
        "flourish_thickness": rng.randint(2, 4),
    }


def generate(out_dir: Path, num_writers: int, n_genuine: int,
             n_forgeries: int, seed: int):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    fonts = [str(FONTS_DIR / f) for f in SCRIPT_FONTS if (FONTS_DIR / f).exists()]
    if len(fonts) < 5:
        raise RuntimeError(f"Only {len(fonts)} script fonts found — need at least 5")
    print(f"[Synth] Using {len(fonts)} script fonts")

    out_dir.mkdir(parents=True, exist_ok=True)

    for wid in range(1, num_writers + 1):
        writer = build_writer(wid, rng, fonts)
        wdir = out_dir / str(wid)
        wdir.mkdir(exist_ok=True)

        # ── Genuine: writer's own hand, natural variation ──
        base = render_base(writer, rng)
        for i in range(1, n_genuine + 1):
            sample = vary_sample(base, np_rng)
            cv2.imwrite(str(wdir / f"original_{wid}_{i}.png"), sample)

        # ── Forgeries: same name, different "hand" ──
        other_fonts = [f for f in fonts if f != writer["font"]]
        for i in range(1, n_forgeries + 1):
            r = np_rng.random()
            if r < 0.33:
                # random forgery: another hand, own letterforms + habits
                forger = build_writer(-1, rng, other_fonts, text=writer["text"])
                forger["truncate"] = writer["truncate"]
                fbase = render_base(forger, rng)
                sample = vary_sample(fbase, np_rng)
            elif r < 0.67:
                # skilled forgery (HARD): identical letterforms (same font)
                # but the forger's own style — different capital size,
                # slant, compression, flourish. This is the analog of a
                # family member imitating the same name: shapes match,
                # personal writing habits don't.
                forger = build_writer(-1, rng, [writer["font"]],
                                      text=writer["text"])
                forger["truncate"] = writer["truncate"]
                fbase = render_base(forger, rng)
                sample = vary_sample(fbase, np_rng)
            else:
                # traced forgery: right shape, distorted dynamics
                sample = vary_sample(base, np_rng, heavy=True)
            cv2.imwrite(str(wdir / f"forgeries_{wid}_{i}.png"), sample)

        if wid % 10 == 0 or wid == num_writers:
            print(f"[Synth] {wid}/{num_writers} writers done "
                  f"(last: '{writer['text']}' in {Path(writer['font']).stem})")

    total = num_writers * (n_genuine + n_forgeries)
    print(f"\n[Synth] DONE — {num_writers} writers, {total} images at {out_dir}")
    print(f"[Synth] Train with:")
    print(f'  python -m training.train_siamese --dataset_path "../dataset/CEDAR,{out_dir}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CEDAR-format signatures")
    parser.add_argument("--out", type=str, default="../dataset/SYNTH")
    parser.add_argument("--writers", type=int, default=100)
    parser.add_argument("--genuine", type=int, default=24, help="genuine samples per writer")
    parser.add_argument("--forgeries", type=int, default=24, help="forgeries per writer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="quick 3-writer test run")
    args = parser.parse_args()

    if args.preview:
        generate(Path(args.out), 3, 5, 5, args.seed)
    else:
        generate(Path(args.out), args.writers, args.genuine, args.forgeries, args.seed)
