"""
Generative head — propozycja E (embedding) warunkowana tracerem Φ.

Produkt (P2.14):
  - target = content query (S/E/payload już w x), nie topic-cheat
  - checkpoint `checkpoints/most_gen_head.pt`
  - API `propose_E` za PHIMOST_GEN=1
  - CLI: `phimost_gen.py`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CANON
from phimost_flags import gen_enabled
from recall_model import PhiBridgeEncoder, gather_idx
from recall_data import RecallBatch

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
DEFAULT_GEN = CKPT_DIR / "most_gen_head.pt"


class PhiGenHead(nn.Module):
    """tok_h + tracer_proj → pred E embedding (d_x)."""

    def __init__(self, d_model: int = 32, d_x: int = 8, dim_tracer: int = 6):
        super().__init__()
        self.d_model = d_model
        self.d_x = d_x
        self.dim_tracer = dim_tracer
        self.tracer_proj = nn.Linear(dim_tracer, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_x),
        )

    def forward(self, tok_h: torch.Tensor, tracer: torch.Tensor, idx: torch.Tensor):
        """
        tok_h [B,N,d], tracer [B,N,n], idx [B] — pozycje do przewidzenia.
        """
        h = gather_idx(tok_h, idx)  # [B,d]
        tr = gather_idx(tracer, idx)  # [B,n]
        tp = self.tracer_proj(tr)
        return self.fuse(torch.cat([h, tp], dim=-1))


def e_targets_from_batch(batch: RecallBatch) -> torch.Tensor:
    """Target = content w query_idx (uczciwy E/S/payload już w x)."""
    B = batch.content.shape[0]
    ar = torch.arange(B)
    return batch.content[ar, batch.query_idx].clone()


def gen_loss(
    encoder: nn.Module,
    head: PhiGenHead,
    batch: RecallBatch,
    *,
    zero_tracer: bool = False,
) -> torch.Tensor:
    tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
    mask = batch.mask
    tok, _ = encoder(batch.content, tr, mask)
    pred = head(tok, tr, batch.query_idx)
    target = e_targets_from_batch(batch).to(pred.device)
    return F.mse_loss(pred, target)


def train_gen(
    encoder: PhiBridgeEncoder,
    head: PhiGenHead,
    train_batches: list[RecallBatch],
    val_batches: list[RecallBatch],
    epochs: int = 8,
    lr: float = 1e-3,
) -> dict:
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        encoder.train()
        head.train()
        for batch in train_batches:
            opt.zero_grad()
            loss = gen_loss(encoder, head, batch, zero_tracer=False)
            loss.backward()
            opt.step()

    encoder.eval()
    head.eval()

    def _eval(zero: bool) -> float:
        losses = []
        with torch.no_grad():
            for batch in val_batches:
                losses.append(
                    float(gen_loss(encoder, head, batch, zero_tracer=zero).item())
                )
        return sum(losses) / max(len(losses), 1)

    mse = _eval(False)
    mse_zero = _eval(True)
    return {
        "mse": mse,
        "mse_zero": mse_zero,
        "delta_zero": mse_zero - mse,  # >0 ⇒ tracer pomaga
    }


@torch.no_grad()
def propose_E(
    encoder: PhiBridgeEncoder,
    head: PhiGenHead,
    content: torch.Tensor,
    tracer: torch.Tensor,
    idx: int = 0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Produktowe API — wymaga PHIMOST_GEN=1."""
    if not gen_enabled():
        raise RuntimeError("PHIMOST_GEN=0 — generative head wyłączony")
    encoder.eval()
    head.eval()
    if content.dim() == 2:
        content = content.unsqueeze(0)
        tracer = tracer.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
    if mask is None:
        mask = torch.ones(content.shape[0], content.shape[1], device=content.device)
    tok, _ = encoder(content, tracer, mask)
    idx_t = torch.tensor([idx], device=content.device)
    return head(tok, tracer, idx_t)[0]


def save_gen_bundle(
    encoder: PhiBridgeEncoder,
    head: PhiGenHead,
    path: str | Path | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    from checkpoint import save_encoder

    path = Path(path) if path else DEFAULT_GEN
    path.parent.mkdir(parents=True, exist_ok=True)
    # najpierw encoder w osobnym pliku-kanonie jeśli brak
    enc_meta = {"paired_gen": str(path)}
    from checkpoint import DEFAULT_MOST

    if not DEFAULT_MOST.is_file():
        save_encoder(encoder, DEFAULT_MOST, kind="most", meta=enc_meta)
    payload = {
        "protocol": "phimost-gen-v1",
        "arch": {
            "d_model": head.d_model,
            "d_x": head.d_x,
            "dim_tracer": head.dim_tracer,
            "encoder_layers": len(encoder.layers),
            "coupling": getattr(encoder, "coupling", CANON.coupling),
            # None = kanon przy apply. False musi przeżyć load, tak jak w checkpoint.py.
            "t_seal": getattr(encoder, "t_seal", None),
        },
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": head.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)
    return path


def load_gen_bundle(
    path: str | Path | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PhiBridgeEncoder, PhiGenHead, dict[str, Any]]:
    path = Path(path) if path else DEFAULT_GEN
    if not path.is_file():
        raise FileNotFoundError(f"brak gen checkpointu: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    arch = payload.get("arch") or {}
    # Stary bundel bez klucza: None, czyli seal kanoniczny w apply_t_key_seal.
    t_seal = arch.get("t_seal") if "t_seal" in arch else None
    enc = PhiBridgeEncoder(
        input_dim=arch.get("d_x", CANON.d_x),
        d=arch.get("d_model", CANON.d_model),
        dim_tracer=arch.get("dim_tracer", CANON.n_dim),
        num_layers=arch.get("encoder_layers", CANON.num_layers),
        coupling=arch.get("coupling", CANON.coupling),
        t_seal=t_seal,
    )
    head = PhiGenHead(
        d_model=arch.get("d_model", CANON.d_model),
        d_x=arch.get("d_x", CANON.d_x),
        dim_tracer=arch.get("dim_tracer", CANON.n_dim),
    )
    enc.load_state_dict(payload["encoder_state_dict"])
    head.load_state_dict(payload["head_state_dict"])
    enc.eval()
    head.eval()
    return enc, head, payload.get("meta") or {}
