"""
PR-G1: zdjęcie / skan → PhiJSON glifów (ten sam kontrakt co G0).

Segmentacja connected-components → wiersze → atomy (S,E,T,state).
T z fokusa czytania / pewności tuszu — NIE z XY jako zamiennik T.
Etykiety: --text w kolejności czytania (MVP bez zewnętrznego OCR).

  python glyph_transcribe.py --demo --out dumps/from_photo
  python glyph_transcribe.py --src foto.png --text MOST --script lat --out dumps/from_photo
  python phimost_learn.py --src dumps/from_photo --no-warm --no-promote --baselines
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "dumps" / "from_photo"


# --- wspólne z G0 (stan / T) -------------------------------------------------

def state_for_T(T: float) -> str:
    if T >= 0.75:
        return "HOT"
    if T >= 0.45:
        return "WARM"
    if T >= 0.2:
        return "COLD"
    return "TOMB"


def glyph_T(i: int, n: int, focus: int) -> float:
    dist = abs(i - focus)
    if dist == 0:
        return 0.95
    if dist == 1:
        return 0.70
    if dist == 2:
        return 0.48
    return 0.28


def bind(edges: list, a: str, b: str, name: str) -> None:
    edges.append({"kind": "bind", "src": a, "dst": b, "name": name})
    edges.append({"kind": "bind", "src": b, "dst": a, "name": name})


# --- skrypt / etykiety -------------------------------------------------------

_HIRA_RE = re.compile(r"[\u3040-\u309F]")
_KATA_RE = re.compile(r"[\u30A0-\u30FF]")
_KANJI_RE = re.compile(r"[\u4E00-\u9FFF]")
_LAT_RE = re.compile(r"[A-Za-z0-9À-ÿ]")


def detect_script_char(ch: str) -> str:
    if _HIRA_RE.match(ch):
        return "hira"
    if _KATA_RE.match(ch):
        return "kata"
    if _KANJI_RE.match(ch):
        return "kanji"
    if _LAT_RE.match(ch):
        return "lat"
    return "sym"


def script_prefix(script: str, ch: str | None, idx: int) -> tuple[str, str]:
    """Zwraca (id_prefix_script, S). Znak z --text zawsze dyktuje skrypt glifu."""
    if ch is None or ch == "":
        sc = "sym" if script == "auto" else script
        return sc, f"{sc}:#{idx}"
    sc = detect_script_char(ch)
    return sc, f"{sc}:{ch}"


# --- segmentacja -------------------------------------------------------------

@dataclass
class GlyphBox:
    label: int
    y0: int
    x0: int
    y1: int
    x1: int
    area: int
    ink: float  # średnia intensywność tuszu [0,1]

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def _to_ink_mask(img: Image.Image, thresh: float = 0.55) -> np.ndarray:
    """True = tusz (ciemny na jasnym tle)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    # zakładamy jasne tło; odwróć jeśli średnia ciemna
    if gray.mean() < 0.45:
        ink = gray > thresh
    else:
        ink = gray < thresh
    return ink


def segment_glyphs(
    img: Image.Image,
    *,
    min_area: int = 40,
    thresh: float = 0.55,
    max_aspect: float = 8.0,
) -> list[GlyphBox]:
    ink = _to_ink_mask(img, thresh=thresh)
    # lekkie domknięcie — za mocne skleja sąsiednie litery
    ink = ndimage.binary_closing(ink, structure=np.ones((2, 2)), iterations=1)
    labeled, n = ndimage.label(ink)
    if n == 0:
        return []
    boxes: list[GlyphBox] = []
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    for lab in range(1, n + 1):
        ys, xs = np.where(labeled == lab)
        if ys.size < min_area:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w < 2 or h < 2:
            continue
        aspect = max(w / h, h / w)
        if aspect > max_aspect:
            continue  # linie / szum
        patch = gray[y0:y1, x0:x1]
        ink_score = float(1.0 - patch.mean()) if patch.mean() > 0.45 else float(patch.mean())
        ink_score = float(np.clip(ink_score, 0.05, 1.0))
        boxes.append(
            GlyphBox(
                label=lab,
                y0=y0,
                x0=x0,
                y1=y1,
                x1=x1,
                area=int(ys.size),
                ink=ink_score,
            )
        )
    return boxes


def _merge_two(a: GlyphBox, b: GlyphBox) -> GlyphBox:
    return GlyphBox(
        label=a.label,
        y0=min(a.y0, b.y0),
        x0=min(a.x0, b.x0),
        y1=max(a.y1, b.y1),
        x1=max(a.x1, b.x1),
        area=a.area + b.area,
        ink=max(a.ink, b.ink),
    )


def merge_nearby_boxes(
    boxes: list[GlyphBox], *, gap_frac: float = 0.35
) -> list[GlyphBox]:
    """Sklej części tego samego znaku (np. dakuten / rozbite kreski)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.cx)
    med_w = float(np.median([b.w for b in boxes])) or 10.0
    gap_max = gap_frac * med_w
    out: list[GlyphBox] = [boxes[0]]
    for b in boxes[1:]:
        prev = out[-1]
        gap = b.x0 - prev.x1
        # pionowe pokrycie
        y_overlap = min(prev.y1, b.y1) - max(prev.y0, b.y0)
        min_h = max(min(prev.h, b.h), 1)
        if gap <= gap_max and y_overlap >= 0.35 * min_h:
            out[-1] = _merge_two(prev, b)
        else:
            out.append(b)
    return out


def filter_boxes(boxes: list[GlyphBox]) -> list[GlyphBox]:
    if len(boxes) < 2:
        return boxes
    areas = [b.area for b in boxes]
    med = float(np.median(areas))
    return [b for b in boxes if b.area >= 0.25 * med]


def group_lines(boxes: list[GlyphBox], *, y_tol: float | None = None) -> list[list[GlyphBox]]:
    if not boxes:
        return []
    boxes = filter_boxes(boxes)
    heights = [b.h for b in boxes]
    med_h = float(np.median(heights)) if heights else 10.0
    tol = y_tol if y_tol is not None else max(0.55 * med_h, 8.0)
    ordered = sorted(boxes, key=lambda b: (b.cy, b.cx))
    lines: list[list[GlyphBox]] = []
    for b in ordered:
        placed = False
        for line in lines:
            ref_cy = sum(g.cy for g in line) / len(line)
            if abs(ref_cy - b.cy) <= tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    lines.sort(key=lambda ln: sum(g.cy for g in ln) / len(ln))
    merged_lines: list[list[GlyphBox]] = []
    for ln in lines:
        ln = merge_nearby_boxes(sorted(ln, key=lambda g: g.cx))
        # odrzuć „linie” z jednym mikro-boxem vs inne
        if ln:
            merged_lines.append(ln)
    # jeśli jedna linia ma 1 glif a inne dużo — zostaw (może być OK)
    return merged_lines


def split_line_by_projection(
    img: Image.Image,
    line_boxes: list[GlyphBox],
    n_chars: int,
) -> list[GlyphBox]:
    """Gdy znamy liczbę znaków w wierszu — tnij po profilu tuszu (MVP OCR-less)."""
    if n_chars <= 0 or not line_boxes:
        return line_boxes
    y0 = min(b.y0 for b in line_boxes)
    y1 = max(b.y1 for b in line_boxes)
    x0 = min(b.x0 for b in line_boxes)
    x1 = max(b.x1 for b in line_boxes)
    ink = _to_ink_mask(img)
    strip = ink[y0:y1, x0:x1]
    if strip.size == 0:
        return line_boxes
    profile = strip.sum(axis=0).astype(np.float64)
    # wygładź
    if profile.size >= 5:
        k = np.ones(5) / 5.0
        profile = np.convolve(profile, k, mode="same")
    width = profile.size
    # równe segmenty jako start; dociągnij do dolin
    edges = [0]
    for i in range(1, n_chars):
        target = int(round(i * width / n_chars))
        lo = max(edges[-1] + 1, target - max(width // (n_chars * 2), 2))
        hi = min(width - (n_chars - i), target + max(width // (n_chars * 2), 2))
        if hi <= lo:
            edges.append(target)
        else:
            edges.append(int(lo + np.argmin(profile[lo:hi])))
    edges.append(width)
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    out: list[GlyphBox] = []
    for i in range(n_chars):
        xa, xb = x0 + edges[i], x0 + edges[i + 1]
        if xb <= xa + 1:
            continue
        patch = gray[y0:y1, xa:xb]
        ink_score = float(1.0 - patch.mean()) if patch.mean() > 0.45 else float(patch.mean())
        out.append(
            GlyphBox(
                label=i + 1,
                y0=y0,
                x0=xa,
                y1=y1,
                x1=xb,
                area=int((y1 - y0) * (xb - xa)),
                ink=float(np.clip(ink_score, 0.05, 1.0)),
            )
        )
    return out if len(out) == n_chars else line_boxes


def pick_focus(
    boxes_flat: list[GlyphBox],
    *,
    focus: int | None,
    focus_center: bool,
    img_w: int,
    img_h: int,
) -> int:
    n = len(boxes_flat)
    if n == 0:
        return 0
    if focus is not None:
        return int(np.clip(focus, 0, n - 1))
    if focus_center:
        cx, cy = img_w / 2.0, img_h / 2.0
        dists = [(i, (b.cx - cx) ** 2 + (b.cy - cy) ** 2) for i, b in enumerate(boxes_flat)]
        return min(dists, key=lambda t: t[1])[0]
    # domyślnie: pierwszy glif pierwszego wiersza (początek czytania)
    return 0


def confidence_boost(T: float, ink: float) -> float:
    """Lekka korekta T pewnością tuszu — nadal fokus rządzi."""
    return float(np.clip(0.85 * T + 0.15 * ink, 0.05, 1.0))


# --- budowa Φ ----------------------------------------------------------------

def boxes_to_phi(
    lines: list[list[GlyphBox]],
    *,
    script: str,
    text: str | None,
    word_id: str,
    focus_index: int,
    source_name: str,
) -> tuple[list[dict], list[dict]]:
    atoms: list[dict] = []
    edges: list[dict] = []

    chars: list[str | None]
    if text:
        # usuń spacje — spacje = granice słów w przyszłości; MVP: ignoruj
        chars = [c for c in text if not c.isspace()]
    else:
        chars = []

    flat: list[tuple[int, int, GlyphBox]] = []  # line_i, pos_in_line, box
    for li, line in enumerate(lines):
        for j, b in enumerate(line):
            flat.append((li, j, b))

    atoms.append(
        {
            "id": "corpus",
            "S": "glyphs/corpus",
            "E": f"transcribed from image:{source_name}",
            "T": 1.0,
            "state": "HOT",
            "bubble": "glyphs",
            "depth": 0,
            "payload": f"PR-G1 source={source_name}",
            "mazur_tracer": 1.0,
            "topic": "corpus",
        }
    )

    page_id = f"page_{word_id}"
    atoms.append(
        {
            "id": page_id,
            "S": f"{script}/page/{word_id}",
            "E": f"page from photo; n_glyphs={len(flat)}",
            "T": 0.9,
            "state": "HOT",
            "bubble": f"glyphs/{script}/{word_id}",
            "depth": 0,
            "payload": text or None,
            "mazur_tracer": 0.9,
            "topic": word_id,
        }
    )
    bind(edges, "corpus", page_id, f"has_page:{word_id}")

    global_i = 0
    all_glyph_ids: list[str] = []
    for li, line in enumerate(lines):
        line_id = f"line_{word_id}_{li}"
        atoms.append(
            {
                "id": line_id,
                "S": f"{script}/line/{word_id}/{li}",
                "E": f"line/{li} n={len(line)}",
                "T": 0.8,
                "state": "WARM",
                "bubble": f"glyphs/{script}/{word_id}",
                "depth": 1,
                "payload": None,
                "mazur_tracer": 0.7,
                "topic": word_id,
            }
        )
        bind(edges, page_id, line_id, "has_line")
        edges.append(
            {"kind": "parent", "src": line_id, "dst": page_id, "name": "line_of"}
        )

        line_ids: list[str] = []
        n_line = len(line)
        for j, box in enumerate(line):
            ch = chars[global_i] if global_i < len(chars) else None
            sc, S = script_prefix(script, ch, global_i)
            # id bezpieczny dla FS / JSON (unikaj dziwnych slash w znaku)
            ch_tag = ch if ch is not None else f"g{global_i}"
            gid = f"{sc}/{word_id}/{global_i}/{ch_tag}"
            T_raw = glyph_T(global_i, max(len(flat), 1), focus_index)
            T = confidence_boost(T_raw, box.ink)
            st = state_for_T(T)
            mazur = (j + 0.5) / max(n_line, 1)  # pozycja w wierszu — nie T
            # bbox w E (metadane), nie w T
            atoms.append(
                {
                    "id": gid,
                    "S": S,
                    "E": (
                        f"glyph/{sc} pos={global_i} line={li} "
                        f"bbox=[{box.x0},{box.y0},{box.x1},{box.y1}] ink={box.ink:.2f}"
                    ),
                    "T": T,
                    "state": st,
                    "bubble": f"glyphs/{sc}/{word_id}",
                    "depth": 2,
                    "payload": (ch if st == "HOT" and ch else None),
                    "mazur_tracer": mazur,
                    "topic": word_id,
                }
            )
            line_ids.append(gid)
            all_glyph_ids.append(gid)
            edges.append(
                {"kind": "parent", "src": gid, "dst": line_id, "name": "glyph_of"}
            )
            bind(edges, line_id, gid, f"contains:{S}")
            global_i += 1

        for a, b in zip(line_ids, line_ids[1:]):
            bind(edges, a, b, "next")
        for i, a in enumerate(line_ids):
            for b in line_ids[i + 1 :]:
                bind(edges, a, b, f"word:{word_id}")

    # same-script HOT peers
    hots = [
        a
        for a in atoms
        if a["depth"] == 2 and a["state"] == "HOT" and a["id"] in all_glyph_ids
    ]
    for i, a in enumerate(hots):
        for b in hots[i + 1 :]:
            bind(edges, a["id"], b["id"], f"script:{a['S'].split(':')[0]}")

    if text and len(chars) != len(flat):
        # ostrzeżenie w corpus E
        atoms[0]["E"] += f" WARN text_len={len(chars)} glyphs={len(flat)}"

    return atoms, edges


def write_dump(
    out: Path,
    atoms: list[dict],
    edges: list[dict],
    *,
    meta: dict,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    n_g = sum(1 for a in atoms if a["depth"] == 2)
    man = {
        "name": out.name,
        "version": "1.0",
        "description": "PR-G1: image→glyph atoms (T from focus/ink; not screenshot-learn)",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_glyphs": n_g,
        "n_hot": n_hot,
        "source": "glyph_transcribe.py",
        "canon": "PhiJSON; T≠XY; Bridge (x,T)",
        **meta,
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    case = f"""# CASE: {out.name} (PR-G1)

atoms={len(atoms)} edges={len(edges)} glyphs={n_g} HOT={n_hot}
source={meta.get('image')} text={meta.get('text')!r} script={meta.get('script')}

```bat
python glyph_transcribe.py --src {meta.get('image', 'foto.png')} --text \"...\" --out {out}
python phimost_learn.py --src {out} --no-warm --no-promote --baselines
```
"""
    (out / "CASE_PHOTO.md").write_text(case, encoding="utf-8")


# --- demo obraz --------------------------------------------------------------

def _load_font(size: int, *, prefer_cjk: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if prefer_cjk:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msgothic.ttc",
                r"C:\Windows\Fonts\meiryo.ttc",
                r"C:\Windows\Fonts\YuGothM.ttc",
                r"C:\Windows\Fonts\malgun.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\consola.ttf",
        ]
    )
    for path in candidates:
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_demo(
    path: Path,
    *,
    lines: list[str] | None = None,
    char_gap: int = 18,
) -> tuple[Path, str]:
    """Rysuje jasne tło + znaki z odstępem (łatwiejsza segmentacja CC)."""
    if lines is None:
        lines = ["MOST", "BRIDGE"]  # tylko łacina; hira = osobny plik (--demo-script hira)
    path.parent.mkdir(parents=True, exist_ok=True)
    pad = 28
    line_h = 80
    # szerokość na zapas
    w = pad * 2 + max(len(ln) for ln in lines) * (48 + char_gap) + 40
    h = pad * 2 + line_h * len(lines)
    img = Image.new("RGB", (w, h), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        prefer_cjk = bool(
            _HIRA_RE.search(line) or _KANJI_RE.search(line) or _KATA_RE.search(line)
        )
        font = _load_font(48 if prefer_cjk else 52, prefer_cjk=prefer_cjk)
        x = pad
        for ch in line:
            draw.text((x, y), ch, fill=(20, 20, 25), font=font)
            bbox = draw.textbbox((x, y), ch, font=font)
            x = bbox[2] + char_gap
        y += line_h
    img.save(path)
    text = "".join(lines)
    return path, text


# --- CLI ---------------------------------------------------------------------

def transcribe_image(
    src: Path,
    *,
    out: Path,
    script: str = "auto",
    text: str | None = None,
    word_id: str | None = None,
    focus: int | None = None,
    focus_center: bool = False,
    min_area: int = 40,
    thresh: float = 0.55,
) -> dict:
    img = Image.open(src).convert("RGB")
    boxes = segment_glyphs(img, min_area=min_area, thresh=thresh)
    lines = group_lines(boxes)

    line_texts: list[str] | None = None
    if text and "\n" in text:
        line_texts = [ln.strip() for ln in text.splitlines() if ln.strip()]
    elif text and len(lines) == 1:
        line_texts = ["".join(c for c in text if not c.isspace())]

    # Gdy znamy N linii tekstu, a CC dało inną liczbę — utnij obraz na N pasów
    if line_texts is not None and len(line_texts) != len(lines) and boxes:
        n_lines = len(line_texts)
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        band_h = max((y1 - y0) / n_lines, 1.0)
        lines = []
        for i in range(n_lines):
            ya = y0 + int(i * band_h)
            yb = y0 + int((i + 1) * band_h) if i < n_lines - 1 else y1
            band = [b for b in boxes if ya <= b.cy < yb or (i == n_lines - 1 and b.cy <= yb)]
            if not band:
                # pusty pas — syntetyczny bbox z profilu całej szerokości
                xa = min(b.x0 for b in boxes)
                xb = max(b.x1 for b in boxes)
                band = [
                    GlyphBox(
                        label=i,
                        y0=ya,
                        x0=xa,
                        y1=max(ya + 1, yb),
                        x1=xb,
                        area=(yb - ya) * (xb - xa),
                        ink=0.5,
                    )
                ]
            lines.append(sorted(merge_nearby_boxes(band), key=lambda g: g.cx))

    if line_texts is not None and len(line_texts) == len(lines):
        new_lines: list[list[GlyphBox]] = []
        for ln_boxes, ln_txt in zip(lines, line_texts):
            n_ch = len([c for c in ln_txt if not c.isspace()])
            if n_ch > 0 and len(ln_boxes) != n_ch:
                ln_boxes = split_line_by_projection(img, ln_boxes, n_ch)
            new_lines.append(ln_boxes)
        lines = new_lines

    flat = [b for ln in lines for b in ln]
    if not flat:
        raise RuntimeError(f"brak glifów w {src} (min_area={min_area}, thresh={thresh})")

    fi = pick_focus(
        flat,
        focus=focus,
        focus_center=focus_center,
        img_w=img.size[0],
        img_h=img.size[1],
    )
    wid = word_id or re.sub(r"[^\w\-]+", "_", src.stem)[:40] or "photo"
    sc = script
    if sc == "auto" and text:
        votes = [detect_script_char(c) for c in text if not c.isspace()]
        if votes:
            sc = max(set(votes), key=votes.count)

    # złóż text bez spacji; jeśli line_texts — zachowaj kolejność
    text_flat = None
    if line_texts is not None:
        text_flat = "".join(line_texts)
    elif text:
        text_flat = "".join(c for c in text if not c.isspace())

    atoms, edges = boxes_to_phi(
        lines,
        script=sc,
        text=text_flat,
        word_id=wid,
        focus_index=fi,
        source_name=src.name,
    )
    meta = {
        "image": str(src),
        "text": text,
        "script": sc,
        "word_id": wid,
        "focus_index": fi,
        "n_lines": len(lines),
        "n_boxes": len(flat),
    }
    write_dump(out, atoms, edges, meta=meta)
    return meta | {"n_atoms": len(atoms), "n_edges": len(edges), "out": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PR-G1: image → glyph PhiJSON (learn from atom state)"
    )
    ap.add_argument("--src", type=Path, help="PNG/JPEG skanu / zdjęcia znaków")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="katalog PhiJSON")
    ap.add_argument("--text", type=str, default=None, help="etykiety w kolejności czytania")
    ap.add_argument(
        "--script",
        default="auto",
        choices=["auto", "lat", "hira", "kata", "kanji", "sym"],
    )
    ap.add_argument("--word-id", default=None)
    ap.add_argument("--focus", type=int, default=None, help="indeks glifu HOT (kolejność czytania)")
    ap.add_argument(
        "--focus-center",
        action="store_true",
        help="fokus = glif najbliżej środka kadru",
    )
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--thresh", type=float, default=0.55)
    ap.add_argument(
        "--demo",
        action="store_true",
        help="wygeneruj syntetyczny PNG (domyślnie tylko łacina) i transkrybuj",
    )
    ap.add_argument(
        "--demo-script",
        default="lat",
        choices=["lat", "hira", "both"],
        help="lat=MOST/BRIDGE (default); hira=osobny plik; both=dwa osobne pliki, bez mieszania na jednym PNG",
    )
    args = ap.parse_args(argv)

    if args.demo:
        demo_dir = ROOT / "ZdjęcieW" / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        # NIGDY nie mieszaj skryptów na jednym PNG — osobne pliki.
        jobs: list[tuple[Path, list[str]]] = []
        if args.demo_script in ("lat", "both"):
            jobs.append((demo_dir / "demo_lat.png", ["MOST", "BRIDGE"]))
        if args.demo_script in ("hira", "both"):
            jobs.append((demo_dir / "demo_hira.png", ["こんにちは", "あいうえお"]))
        # transkrybuj pierwszy (główny); przy both — złóż dwa dump-y? MVP: pierwszy + info
        src = None
        text = args.text
        for path, lines in jobs:
            path, _ = render_demo(path, lines=lines)
            print(f"[demo] wrote {path} lines={lines!r}")
            if src is None:
                src = path
                if text is None:
                    text = "\n".join(lines)
        assert src is not None
        if args.demo_script == "both" and args.text is None:
            print(
                "[demo] both: transkrypcja startuje od demo_lat.png; "
                "hira osobno: --src ZdjęcieW/demo/demo_hira.png --text \"こんにちは\\nあいうえお\""
            )
    else:
        if not args.src:
            ap.error("podaj --src PATH albo --demo")
        src = args.src
        text = args.text

    info = transcribe_image(
        src,
        out=args.out,
        script=args.script,
        text=text,
        word_id=args.word_id,
        focus=args.focus,
        focus_center=args.focus_center,
        min_area=args.min_area,
        thresh=args.thresh,
    )
    print(
        f"OK transcribe → {info['out']} glyphs={info['n_boxes']} lines={info['n_lines']} "
        f"atoms={info['n_atoms']} focus={info['focus_index']} script={info['script']}"
    )
    # szybki load
    from phi_dump import load_dump

    man, g = load_dump(args.out)
    ts = [a.T for a in g.atoms if a.depth == 2]
    print(
        f"load_ok atoms={len(g.atoms)} edges={len(g.edges)} "
        f"glyph_T=[{min(ts):.2f},{max(ts):.2f}]" if ts else f"load_ok atoms={len(g.atoms)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
