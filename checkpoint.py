"""
M1 — zapis / odczyt wag encodera Most (PhiBridgeEncoder / twin Softmax).

Kanon: checkpoints/ w root PhiMost. Nie mieszać z Holon SE (M3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from config import CANON
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
DEFAULT_MOST = CKPT_DIR / "most_dump_recall.pt"


def ensure_dir(path: Path | None = None) -> Path:
    d = path or CKPT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_encoder(
    kind: str = "most",
    *,
    input_dim: int | None = None,
    d: int | None = None,
    h: int | None = None,
    dim_tracer: int | None = None,
    num_layers: int | None = None,
    coupling: str | None = None,
) -> nn.Module:
    kw = dict(
        input_dim=input_dim if input_dim is not None else CANON.d_x,
        d=d if d is not None else CANON.d_model,
        h=h if h is not None else CANON.n_heads,
        dim_tracer=dim_tracer if dim_tracer is not None else CANON.n_dim,
        num_layers=num_layers if num_layers is not None else CANON.num_layers,
    )
    if kind == "most":
        return PhiBridgeEncoder(
            **kw, coupling=coupling or CANON.coupling
        )
    if kind == "softmax":
        return PhiSoftmaxEncoder(**kw)
    raise ValueError(f"unknown encoder kind={kind!r}")


def save_encoder(
    model: nn.Module,
    path: str | Path | None = None,
    *,
    kind: str = "most",
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path) if path else DEFAULT_MOST
    ensure_dir(path.parent)
    arch = {
        "kind": kind,
        "input_dim": int(model.embed.in_features),
        "d": int(model.d),
        "h": int(model.h),
        "dim_tracer": int(
            model.layers[0]["e"].in_features
            if kind == "most"
            else CANON.n_dim
        ),
        "num_layers": len(model.layers),
    }
    if kind == "most":
        arch["coupling"] = getattr(model, "coupling", CANON.coupling)
    payload = {
        "protocol": "phimost-encoder-v1",
        "arch": arch,
        "state_dict": model.state_dict(),
        "meta": meta or {},
        "canon": {
            "coupling": CANON.coupling,
            "num_layers": CANON.num_layers,
            "d_model": CANON.d_model,
            "n_dim": CANON.n_dim,
            "d_x": CANON.d_x,
        },
    }
    torch.save(payload, path)
    return path


def load_encoder(
    path: str | Path | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(path) if path else DEFAULT_MOST
    if not path.is_file():
        raise FileNotFoundError(f"brak checkpointu: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"nieznany format checkpointu: {path}")
    arch = payload.get("arch") or {}
    kind = arch.get("kind", "most")
    model = build_encoder(
        kind,
        input_dim=arch.get("input_dim"),
        d=arch.get("d"),
        h=arch.get("h"),
        dim_tracer=arch.get("dim_tracer"),
        num_layers=arch.get("num_layers"),
        coupling=arch.get("coupling"),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta") or {}
