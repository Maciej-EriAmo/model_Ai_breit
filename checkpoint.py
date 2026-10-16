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

# Stare pliki .pt nie mają arch.t_seal. Te źródła uczyły się z seal wyłączonym,
# bo oś 3 to energia orbitalu albo barwa, nie temperatura archiwum.
_LEGACY_SEAL_OFF = ("struktura", "qm_orbital", "glyphs_photo", "from_photo")


def _legacy_t_seal(meta: dict[str, Any]) -> bool | None:
    blob = " ".join(str(meta.get(k, "")) for k in ("dump", "source", "task")).lower()
    if any(tok in blob for tok in _LEGACY_SEAL_OFF):
        return False
    return None


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
    t_seal: bool | None = None,
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
            **kw, coupling=coupling or CANON.coupling, t_seal=t_seal
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
        # None = kanon (seal włączony w apply_t_key_seal). False musi przeżyć load.
        arch["t_seal"] = getattr(model, "t_seal", None)
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
    t_seal = None
    if kind == "most":
        if "t_seal" in arch:
            t_seal = arch.get("t_seal")
        else:
            t_seal = _legacy_t_seal(payload.get("meta") or {})
    model = build_encoder(
        kind,
        input_dim=arch.get("input_dim"),
        d=arch.get("d"),
        h=arch.get("h"),
        dim_tracer=arch.get("dim_tracer"),
        num_layers=arch.get("num_layers"),
        coupling=arch.get("coupling"),
        t_seal=t_seal,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta") or {}
