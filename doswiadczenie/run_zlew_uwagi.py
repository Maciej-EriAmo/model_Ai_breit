"""Pilot: zlew uwagi (attention sink) — Softmax vs Most (L1) vs Most bez normalizacji.

Metryka jak u Gu i in. 2024: dla głowy (l,h) średni udział masy wiersza idący do
tokenu 0, liczony po zapytaniach i >= --q0. Głowa jest zlewem, gdy udział > --eps.
sink_frac = odsetek takich głów. Dla ramienia raw udział liczony z A_i0 / sum_j A_ij
(tylko do metryki; model używa A / N).

Ramiona:
  softmax    klasyczna uwaga przyczynowa
  most       Most-1, wiersz dzielony przez sumę (L1), jak w lm_tiny.py
  raw        Most-1 bez normalizacji wiersza, stała skala 1/N
  most_logn  Most-1 z wynikiem LSSA: log d · log N_i · cos(q, k)
  lssa       softplus(wynik LSSA) + L1, bez Lorentza (Gao i in. 2025, etap 1)
  lssa2      jak lssa, ale softplus²
  softmax_logn  softmax z wynikiem LSSA (kontrola skali)

Dodatkowo strata na ćwiartkach pozycji (czy raw traci na dalszych pozycjach).
Wynik: jeden wiersz JSON na (ramię, ziarno), na stdout i do --out.

  python doswiadczenie/run_zlew_uwagi.py --preset medium --steps 2000 --seq-len 256
  python doswiadczenie/run_zlew_uwagi.py --preset small --steps 6000 --seq-len 128 --bos 2
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[1]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import lm_tiny as L


def raw_forward(self, x, tracer, return_attn=False):
    """Kopia _MostCausalBlock.forward bez normalizacji wiersza; stała skala 1/N."""
    B, N, _ = x.shape
    q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
    q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
    S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
    E = self.e(tracer).transpose(1, 2).unsqueeze(-1)
    dE = self.e_scale * (E - E.transpose(-2, -1))
    A = self._most_weights(S, dE)
    w = A / N
    y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
    x = self._drop(x + self.o(y))
    x = self._drop(x + self.ff(x))
    return (x, w) if return_attn else x


def logn_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Wynik jak w LSSA (Gao i in. 2025): log d · log N_i · cos(q, k).

    N_i = i + 1, liczba kluczy widocznych z zapytania i (maska przyczynowa).
    """
    N, dh = q.shape[-2], q.shape[-1]
    n_i = torch.arange(1, N + 1, device=q.device, dtype=q.dtype).view(N, 1)
    scale = math.log(dh) * torch.log(n_i)
    return scale * (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))


def most_logn_forward(self, x, tracer, return_attn=False):
    """Most-1 (softplus², Lorentz, L1) z wynikiem LSSA zamiast QKᵀ/√d."""
    B, N, _ = x.shape
    q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
    q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
    S = logn_scores(q, k)
    E = self.e(tracer).transpose(1, 2).unsqueeze(-1)
    dE = self.e_scale * (E - E.transpose(-2, -1))
    A = self._most_weights(S, dE)
    w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
    y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
    x = self._drop(x + self.o(y))
    x = self._drop(x + self.ff(x))
    return (x, w) if return_attn else x


def make_lssa_forward(power: int):
    """LSSA etap 1: softplus(wynik LSSA)^power, maska, L1. Bez Lorentza i ΔE."""

    def forward(self, x, return_attn=False):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        A = F.softplus(logn_scores(q, k)).pow(power)
        A = A.masked_fill(L.causal_mask(N, x.device).view(1, 1, N, N), 0.0)
        w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
        y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = self._drop(x + self.o(y))
        x = self._drop(x + self.ff(x))
        return (x, w) if return_attn else x

    return forward


def softmax_logn_forward(self, x, return_attn=False):
    """Softmax z wynikiem LSSA. Kontrola: czy sama skala zamyka różnicę."""
    B, N, _ = x.shape
    q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
    q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
    S = logn_scores(q, k).masked_fill(L.causal_mask(N, x.device).view(1, 1, N, N), -1e9)
    w = F.softmax(S, dim=-1)
    y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
    x = self._drop(x + self.o(y))
    x = self._drop(x + self.ff(x))
    return (x, w) if return_attn else x


ARMS = {
    "softmax_logn": ("softmax", softmax_logn_forward),
    "softmax": ("softmax", None),
    "most": ("most", None),
    "raw": ("most", raw_forward),
    "most_logn": ("most", most_logn_forward),
    "lssa": ("softmax", make_lssa_forward(1)),
    "lssa2": ("softmax", make_lssa_forward(2)),
}


def build(arm: str, cfg: L.TinyLMConfig) -> L.TinyLM:
    kernel, fwd = ARMS[arm]
    m = L.TinyLM(cfg, attn=kernel)
    if fwd is not None:
        for b in m.blocks:
            b.forward = fwd.__get__(b)
    return m


def make_cfg(preset: str, seq_len: int) -> L.TinyLMConfig:
    if preset == "small":
        return L.TinyLMConfig(
            vocab_size=256, d=128, h=4, num_layers=4, n_dim=4, ff_mult=4, max_seq=seq_len
        )
    return L.TinyLMConfig.preset(preset)


def with_bos(batches: list[torch.Tensor], bos: int) -> list[torch.Tensor]:
    """bos >= 0: stały token na pozycji 0 każdego okna (okno skraca się o jeden znak)."""
    if bos < 0:
        return batches
    return [
        torch.cat([torch.full((b.shape[0], 1), bos, dtype=b.dtype), b[:, :-1]], 1)
        for b in batches
    ]


def run_one(arm, seed, cfg, train, evb, args) -> dict:
    torch.manual_seed(seed)
    m = build(arm, cfg)
    bt = with_bos(
        L.make_batches(train, seq_len=args.seq_len, batch_size=16, n_batches=args.steps, seed=seed),
        args.bos,
    )
    t0 = time.time()
    L.train_lm(m, bt, steps=args.steps, lr=1e-3, log_every=10**9)
    m.eval()
    seq = args.seq_len
    shares, nll = [], torch.zeros(seq - 1)
    with torch.no_grad():
        for b in evb:
            logits, ws = m(b, return_attn=True)
            nll += F.cross_entropy(
                logits[:, :-1].transpose(1, 2), b[:, 1:], reduction="none"
            ).mean(0)
            sh = torch.stack(
                [
                    (w / w.sum(-1, keepdim=True).clamp_min(1e-12))[..., args.q0 :, 0].mean((0, 2))
                    for w in ws
                ]
            )
            shares.append(sh)  # [L,h]
    sh = torch.stack(shares).mean(0)
    nll /= len(evb)
    q = (seq - 1) // 4
    return {
        "arm": arm,
        "seed": seed,
        "preset": args.preset,
        "steps": args.steps,
        "seq_len": seq,
        "bos": args.bos,
        "sink_frac": float((sh > args.eps).float().mean()),
        "share0_max": float(sh.max()),
        "share0_mean": float(sh.mean()),
        "uniform_ref": float(torch.tensor([1 / (i + 1) for i in range(args.q0, seq)]).mean()),
        "loss": float(nll.mean()),
        "loss_quarters": [round(float(nll[i * q : (i + 1) * q].mean()), 4) for i in range(4)],
        "share_per_layer_head": [[round(float(v), 3) for v in row] for row in sh],
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="medium", help="tiny | medium | small")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seeds", default="0", help="np. 0,1,2")
    ap.add_argument("--arms", default="softmax,most,raw")
    ap.add_argument("--bos", type=int, default=-1, help="kod tokenu BOS; -1 = bez BOS")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--q0", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="plik JSONL (dopisywany)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = make_cfg(args.preset, args.seq_len)
    data = L.build_char_corpus(L.load_text_corpus(), vocab_size=cfg.vocab_size)
    n_ev = data.numel() // 10
    train, ev = data[:-n_ev], data[-n_ev:]
    evb = with_bos(
        L.make_batches(ev, seq_len=args.seq_len, batch_size=16, n_batches=8, seed=999), args.bos
    )
    for arm in args.arms.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            r = run_one(arm, seed, cfg, train, evb, args)
            line = json.dumps(r)
            print(line, flush=True)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
