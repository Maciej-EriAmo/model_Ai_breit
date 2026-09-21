"""
Cutout / zdjęcie → tokeny Most: content x + tracer τ.

Kanon 1d_do_2d: T (hot/cold) ≠ XY. XY może być w τ, ale osobną osią.
Patchy (MVP) — nie pełne 256² tokenów.

DESI τ (n_dim=6) — INNA semantyka slotów 4–5 niż Φ glyph:
  Φ:     0–2 depth OH, 3=T, 4=state, 5=mazur
  DESI:  0–2 depth OH (jasność), 3=T z BARWY (g−r), 4=x, 5=y
Content = luminancja+morfologia (bez per-band color).
Nie transferować wag glify↔DESI bez adaptera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import CANON
from desi_cutouts import CACHE, DEMO_DIR, import_demo_fits, load_cached


@dataclass
class ImageBatch:
    content: torch.Tensor  # [B, N, d_x]
    tracer: torch.Tensor  # [B, N, n_dim]
    labels: torch.Tensor  # [B]
    mask: torch.Tensor  # [B, N]


def patch_features(tile: np.ndarray, d_x: int = CANON.d_x) -> np.ndarray:
    """
    Content = jasność + morfologia (BEZ barwy / per-band means).
    Barwa idzie do τ.T — nie dublować w x.
    """
    lum = tile.mean(axis=0)  # [h,w] — luminancja
    # tekstura
    dy = np.mean(np.abs(np.diff(lum, axis=0))) if lum.shape[0] > 1 else 0.0
    dx = np.mean(np.abs(np.diff(lum, axis=1))) if lum.shape[1] > 1 else 0.0
    p25, p75 = np.percentile(lum, [25.0, 75.0])
    feat = [
        float(lum.mean()),
        float(lum.std() + 1e-8),
        float(lum.max()),
        float(lum.min()),
        float(p75 - p25),
        float(dy),
        float(dx),
        float(np.median(lum)),
    ]
    out = np.zeros(d_x, dtype=np.float32)
    n = min(d_x, len(feat))
    out[:n] = np.asarray(feat[:n], dtype=np.float32)
    return out


def cutout_to_tokens(
    image: np.ndarray,
    T_map: np.ndarray,
    *,
    patch: int = 16,
    d_x: int = CANON.d_x,
    n_dim: int = CANON.n_dim,
) -> tuple[np.ndarray, np.ndarray]:
    """
    image [C,H,W], T_map [H,W] → content [N,d_x], tracer [N,n_dim].

    Tracer DESI (n_dim=6):
      0..2 depth OH z lokalnej jasności (nie z barwy)
      3    T ∈ [0,1]  ← ciepłota z BARWY (g−r), NIE XY, NIE SNR-as-T
      4    x_norm
      5    y_norm
    """
    C, H, W = image.shape
    ph = max(1, H // patch)
    pw = max(1, W // patch)
    H2, W2 = ph * patch, pw * patch
    y0, x0 = (H - H2) // 2, (W - W2) // 2
    img = image[:, y0 : y0 + H2, x0 : x0 + W2]
    Tm = T_map[y0 : y0 + H2, x0 : x0 + W2]
    lum_full = img.mean(axis=0)

    contents = []
    tracers = []
    for i in range(ph):
        for j in range(pw):
            ys, xs = i * patch, j * patch
            tile = img[:, ys : ys + patch, xs : xs + patch]
            t_tile = Tm[ys : ys + patch, xs : xs + patch]
            contents.append(patch_features(tile, d_x=d_x))

            t_val = float(np.clip(t_tile.mean(), 0.0, 1.0))
            bright = float(lum_full[ys : ys + patch, xs : xs + patch].mean())
            if bright < 0.25:
                depth = 2
            elif bright < 0.45:
                depth = 1
            else:
                depth = 0
            tr = np.zeros(n_dim, dtype=np.float32)
            if n_dim >= 3:
                tr[min(depth, 2)] = 1.0
            if n_dim >= 4:
                tr[3] = t_val
            if n_dim >= 5:
                tr[4] = (j + 0.5) / pw
            if n_dim >= 6:
                tr[5] = (i + 0.5) / ph
            tracers.append(tr)

    return np.stack(contents), np.stack(tracers)


def _augment_flip(image: np.ndarray, T_map: np.ndarray, rng: np.random.Generator):
    img, Tm = image.copy(), T_map.copy()
    if rng.random() < 0.5:
        img = np.flip(img, axis=-1).copy()
        Tm = np.flip(Tm, axis=-1).copy()
    if rng.random() < 0.5:
        img = np.flip(img, axis=-2).copy()
        Tm = np.flip(Tm, axis=-2).copy()
    return img, Tm


def _filter_files_by_size(
    files: list[Path], *, size: int | None
) -> list[Path]:
    """Zostaw tylko cutouty o boku `size` (z meta npz lub nazwy _s{size})."""
    if size is None:
        return files
    out: list[Path] = []
    for p in files:
        if f"_s{size}" in p.stem or f"_s{size}_" in p.name:
            out.append(p)
            continue
        try:
            d = load_cached(p)
            h, w = int(d["image"].shape[-2]), int(d["image"].shape[-1])
            if h == size and w == size:
                out.append(p)
        except Exception:
            continue
    return out


def _items_from_files(
    files: list[Path],
    *,
    patch: int,
    d_x: int,
    n_dim: int,
    augment: int,
    rng: np.random.Generator,
    geometry_only: bool,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    items: list[tuple[np.ndarray, np.ndarray, int]] = []
    for path in files:
        d = load_cached(path)
        base_img = np.asarray(d["image"], dtype=np.float32)
        base_T = np.asarray(d["T"], dtype=np.float32)
        y = int(d["label"])
        variants = [(base_img, base_T)]
        for _ in range(max(0, augment)):
            variants.append(_augment_flip(base_img, base_T, rng))
        for img, Tm in variants:
            content, tracer = cutout_to_tokens(
                img, Tm, patch=patch, d_x=d_x, n_dim=n_dim
            )
            if geometry_only:
                content = rng.normal(0.0, 0.05, size=content.shape).astype(np.float32)
            items.append((content, tracer, y))
    return items


def _pack_batches(
    items: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    batch_size: int,
    d_x: int,
    n_dim: int,
) -> list[ImageBatch]:
    if not items:
        return []
    # Po filtrze size wszystkie N powinny być równe; pad i tak + maska.
    N = max(c.shape[0] for c, _, _ in items)
    batches: list[ImageBatch] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        B = len(chunk)
        content = torch.zeros(B, N, d_x)
        tracer = torch.zeros(B, N, n_dim)
        labels = torch.zeros(B, dtype=torch.long)
        mask = torch.zeros(B, N)
        for b, (c, t, y) in enumerate(chunk):
            n = c.shape[0]
            content[b, :n] = torch.from_numpy(c)
            tracer[b, :n] = torch.from_numpy(t)
            labels[b] = y
            mask[b, :n] = 1.0
        batches.append(ImageBatch(content, tracer, labels, mask))
    return batches


def load_desi_train_val(
    *,
    cache_dir: Path | None = None,
    patch: int = 16,
    batch_size: int = 4,
    d_x: int = CANON.d_x,
    n_dim: int = CANON.n_dim,
    max_files: int | None = None,
    augment: int = 3,
    seed: int = 0,
    val_frac: float = 0.34,
    geometry_only: bool = False,
    from_demo: bool = False,
    demo_dir: Path | None = None,
    demo_only: bool = False,
    force_import: bool = False,
    size: int | None = 128,
) -> tuple[list[ImageBatch], list[ImageBatch], dict]:
    """
    Uczciwy split: najpierw pliki → train/val, **potem** augment tylko na train.
    Domyślnie size=128 (jednolite N). mask honorowana w Bridge/Softmax.
    """
    if from_demo:
        import_demo_fits(
            demo_dir or DEMO_DIR, force=force_import, prefer_size=size
        )
    cache_dir = Path(cache_dir or CACHE)
    if demo_only:
        files = sorted(cache_dir.glob("demo_*.npz"))
    else:
        # bez demo_ — unikaj mieszania seed+demo przy default
        files = sorted(
            p for p in cache_dir.glob("*.npz") if not p.name.startswith("demo_")
        )
        if not files:
            files = sorted(cache_dir.glob("*.npz"))
    files = _filter_files_by_size(files, size=size)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"brak cutoutów size={size} w {cache_dir} — "
            f"pobierz_paczke --size {size} + --demo-only --force-import"
        )

    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * val_frac))) if len(files) > 1 else 1
    if len(files) == 1:
        val_files = list(files)
        train_files = list(files)
    else:
        val_files = files[:n_val]
        train_files = files[n_val:]
        if not train_files:
            train_files, val_files = files[:-1], files[-1:]

    # Sprawdź brak przecieku id
    train_ids = {p.resolve() for p in train_files}
    val_ids = {p.resolve() for p in val_files}
    leak = train_ids & val_ids
    if leak and len(files) > 1:
        raise RuntimeError(f"train/val leak: {sorted(x.name for x in leak)[:5]}")

    train_items = _items_from_files(
        train_files,
        patch=patch,
        d_x=d_x,
        n_dim=n_dim,
        augment=augment,
        rng=rng,
        geometry_only=geometry_only,
    )
    val_items = _items_from_files(
        val_files,
        patch=patch,
        d_x=d_x,
        n_dim=n_dim,
        augment=0,  # NIGDY augment na val
        rng=rng,
        geometry_only=geometry_only,
    )
    rng.shuffle(train_items)

    train_b = _pack_batches(train_items, batch_size=batch_size, d_x=d_x, n_dim=n_dim)
    val_b = _pack_batches(val_items, batch_size=batch_size, d_x=d_x, n_dim=n_dim)
    meta = {
        "n_files": len(files),
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_train_items": len(train_items),
        "n_val_items": len(val_items),
        "size": size,
        "seq_len": int(train_b[0].content.shape[1]) if train_b else 0,
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
    }
    return train_b, val_b, meta


def load_desi_batches(**kwargs) -> list[ImageBatch]:
    """Kompatybilność: zwraca train+val złączone (preferuj load_desi_train_val)."""
    train, val, _ = load_desi_train_val(**kwargs)
    return train + val


def png_to_image_T(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Tor-2 helper — NIE używać jako zamiennik Φ-atomów / nauki Most z Lunety.
    PNG → [C,H,W] + T-proxy z luminancji (słaby sygnał).
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    img = np.transpose(arr, (2, 0, 1))
    gray = img.mean(axis=0)
    t = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    return img.astype(np.float32), t.astype(np.float32)
