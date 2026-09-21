"""Tiny causal LM — Most vs Softmax (kryterium #1b smoke ≪1M params).

Uwaga Most: softplus² + ΔE(tracer); Softmax: klasyczny causal attn.
Tracer LM: cechy pozycji (+ lekki rzut z embedów) — nie Φ-dump.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MostKnobs:
    """Gałki zamka Lorentza (strojenie wag uwagi, nie lr).

    Bazowo: A = g * softplus(S/τ)^2 / (g^2 + (e_scale*dE)^2)
    dE[i,j] = E_i - E_j (query i → key j). E z Linear może być ujemne,
    ale sam mianownik dE² **gubi znak** — stąd delta_mode.
    """

    gamma_init: float = 0.0  # init log_gamma; softplus(0)≈0.69
    e_scale: float = 1.0  # skala ΔE po projekcji e()
    couple_temp: float = 1.0  # τ: softplus(S/τ); >1 → gładsze C
    # signed ΔE — A≥0 przed normalizacją
    # square: klasyczny Lorentz (bez znaku)
    # directed: * (bias + (1-bias)*σ(α·dE)) — dE>0 = E_q>E_k = w dół energii
    # onesided: g_eff = g·(1+softplus(α·dE)) — szerszy zamek „w dół”
    delta_mode: str = "square"  # square|directed|onesided
    delta_alpha: float = 1.0
    directed_bias: float = 0.25  # podłoga masy pod górę energii


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int = 128  # ASCII / byte-ish
    d: int = 64
    h: int = 2
    num_layers: int = 2
    n_dim: int = 4  # tracer
    ff_mult: int = 4
    max_seq: int = 128
    dropout: float = 0.0
    most: MostKnobs = MostKnobs()

    @staticmethod
    def preset(name: str) -> "TinyLMConfig":
        """Presety skali: tiny ≪1M (#1b), medium ~2–5M (#1c)."""
        n = name.strip().lower()
        if n in ("tiny", "1b", "default"):
            return TinyLMConfig()
        if n in ("medium", "med", "1c"):
            return TinyLMConfig(
                vocab_size=256,
                d=256,
                h=4,
                num_layers=4,
                n_dim=4,
                ff_mult=4,
                max_seq=256,
            )
        if n in ("medium_long", "med_long", "1c_long"):
            # ten sam ~3.3M, pos-emb na 2–4k (probe lost-in-middle)
            return TinyLMConfig(
                vocab_size=256,
                d=256,
                h=4,
                num_layers=4,
                n_dim=4,
                ff_mult=4,
                max_seq=4096,
            )
        raise ValueError(f"nieznany preset LM: {name!r} (tiny|medium|medium_long)")

    def approx_params(self, attn: str = "most") -> int:
        """Grube oszacowanie (bez biasów)."""
        d, h, L, V, nd = self.d, self.h, self.num_layers, self.vocab_size, self.n_dim
        emb = V * d + self.max_seq * d
        # qkv + o + ln + ff
        per = (d * 3 * d) + (d * d) + (2 * d) + (d * (self.ff_mult * d) * 2)
        if attn == "most":
            per += nd * h + h  # e + log_gamma
        head = d * V
        return emb + L * per + head


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def causal_mask(n: int, device: torch.device) -> torch.Tensor:
    """True = zablokowane (nie wolno patrzeć)."""
    return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)


def position_tracer(pos: torch.Tensor, n_dim: int) -> torch.Tensor:
    """pos [B,N] int → [B,N,n_dim] cechy „energii” pozycji."""
    B, N = pos.shape
    t = pos.float().unsqueeze(-1) / max(N - 1, 1)  # [B,N,1] in [0,1]
    feats = [t, 1.0 - t]
    if n_dim >= 3:
        feats.append(torch.sin(2 * math.pi * t))
    if n_dim >= 4:
        feats.append(torch.cos(2 * math.pi * t))
    while len(feats) < n_dim:
        feats.append(feats[len(feats) % 2] * 0.5)
    return torch.cat(feats[:n_dim], dim=-1)


class _SoftmaxCausalBlock(nn.Module):
    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        d, h = cfg.d, cfg.h
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.ff_mult * d),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * d, d),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        S = S.masked_fill(causal_mask(N, x.device).view(1, 1, N, N), -1e9)
        w = F.softmax(S, dim=-1)
        y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = x + self.o(y)
        x = x + self.ff(x)
        if return_attn:
            return x, w
        return x


class _MostCausalBlock(nn.Module):
    """Zamek Lorentza + maska przyczynowa (bez zerowania diagonali — LM)."""

    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        d, h, nd = cfg.d, cfg.h, cfg.n_dim
        assert d % h == 0
        kn = cfg.most
        self.h, self.dh = h, d // h
        self.e_scale = float(kn.e_scale)
        self.couple_temp = max(float(kn.couple_temp), 1e-3)
        self.delta_mode = str(kn.delta_mode)
        self.delta_alpha = float(kn.delta_alpha)
        self.directed_bias = float(min(max(kn.directed_bias, 0.0), 1.0))
        if self.delta_mode not in ("square", "directed", "onesided"):
            raise ValueError(f"delta_mode={self.delta_mode!r}")
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.e = nn.Linear(nd, h)
        nn.init.ones_(self.e.weight)
        nn.init.zeros_(self.e.bias)
        self.log_gamma = nn.Parameter(torch.full((h, 1, 1), float(kn.gamma_init)))
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.ff_mult * d),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * d, d),
        )

    def _most_weights(self, S: torch.Tensor, dE: torch.Tensor) -> torch.Tensor:
        """A≥0 z opcjonalnym znakiem ΔE (directed/onesided)."""
        g0 = F.softplus(self.log_gamma) + 1e-3
        C = F.softplus(S / self.couple_temp).square()
        if self.delta_mode == "square":
            g = g0
            A = g * C / (g.square() + dE.square() + 1e-12)
        elif self.delta_mode == "onesided":
            # dE>0 (E_q>E_k): szerszy g → grubszy ogon „w dół” energii
            g = g0 * (1.0 + F.softplus(self.delta_alpha * dE))
            A = g * C / (g.square() + dE.square() + 1e-12)
        else:  # directed
            g = g0
            base = g * C / (g.square() + dE.square() + 1e-12)
            # σ(α·dE): większa waga gdy key ma niższą energię niż query
            gate = torch.sigmoid(self.delta_alpha * dE)
            b = self.directed_bias
            A = base * (b + (1.0 - b) * gate)
        return A

    def forward(self, x: torch.Tensor, tracer: torch.Tensor, return_attn: bool = False):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        E = self.e(tracer).transpose(1, 2).unsqueeze(-1)  # [B,h,N,1] — może być ujemne
        dE = self.e_scale * (E - E.transpose(-2, -1))
        A = self._most_weights(S, dE)
        A = A.masked_fill(causal_mask(N, x.device).view(1, 1, N, N), 0.0)
        w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
        from precision import assert_finite

        assert_finite(S, "lm.S")
        assert_finite(dE, "lm.dE")
        assert_finite(A, "lm.A")
        assert_finite(w, "lm.w")
        y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = x + self.o(y)
        x = x + self.ff(x)
        if return_attn:
            return x, w
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig, attn: str = "softmax"):
        super().__init__()
        if attn not in ("softmax", "most"):
            raise ValueError(attn)
        self.cfg = cfg
        self.attn = attn
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d)
        if attn == "most":
            self.blocks = nn.ModuleList([_MostCausalBlock(cfg) for _ in range(cfg.num_layers)])
            self.tracer_mix = nn.Linear(cfg.d + cfg.n_dim, cfg.n_dim)
        else:
            self.blocks = nn.ModuleList([_SoftmaxCausalBlock(cfg) for _ in range(cfg.num_layers)])
            self.tracer_mix = None
        self.out_norm = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tie

    def forward(self, idx: torch.Tensor, return_attn: bool = False):
        """idx [B,N] → logits [B,N,V]; opcjonalnie lista wag [B,h,N,N] per warstwa."""
        B, N = idx.shape
        if N > self.cfg.max_seq:
            raise ValueError(f"seq {N} > max_seq {self.cfg.max_seq}")
        pos = torch.arange(N, device=idx.device).unsqueeze(0).expand(B, N)
        x = self.tok(idx) + self.pos(pos)
        attn_ws: list[torch.Tensor] = []
        if self.attn == "most":
            tr0 = position_tracer(pos, self.cfg.n_dim)
            tracer = self.tracer_mix(torch.cat([x, tr0], dim=-1))
            for blk in self.blocks:
                if return_attn:
                    x, w = blk(x, tracer, return_attn=True)
                    attn_ws.append(w)
                else:
                    x = blk(x, tracer)
        else:
            for blk in self.blocks:
                if return_attn:
                    x, w = blk(x, return_attn=True)
                    attn_ws.append(w)
                else:
                    x = blk(x)
        logits = self.head(self.out_norm(x))
        if return_attn:
            return logits, attn_ws
        return logits


def lm_loss(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Next-token CE: przewiduj idx[:,1:] z logits[:,:-1]."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        idx[:, 1:].reshape(-1),
    )


def build_char_corpus(texts: list[str], vocab_size: int = 128) -> torch.Tensor:
    """Złącz teksty → LongTensor kodów 0..vocab_size-1 (ASCII clamp)."""
    raw = "\n\n".join(texts).encode("utf-8", errors="replace")
    ids = [min(b, vocab_size - 1) for b in raw]
    if len(ids) < 8:
        ids = ids * 64
    return torch.tensor(ids, dtype=torch.long)


DEFAULT_TEXTS = [
    "PhiMost tiny LM smoke — Most vs Softmax on the same bytes.\n",
    "Uklad okresowy: H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca.\n",
    "Most Lorentza: A = gamma * softplus(S)^2 / (gamma^2 + dE^2).\n",
    "Kryterium sukcesu #1b: krzywa loss przy tej samej ilosci danych.\n",
    "Ala ma kota. Kot ma Ale. Lit i sod to grupa 1. Potas tez.\n",
    "The quick brown fox jumps over the lazy dog. 0123456789.\n",
] * 8


def load_text_corpus(root: Path | None = None, *, repeat: int = 2) -> list[str]:
    """Teksty z docs/ + doswiadczenie/glify/*.md + DEFAULT — wspólny korpus Most/SM."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    chunks: list[str] = list(DEFAULT_TEXTS)
    for folder in (root / "docs", root / "doswiadczenie" / "glify"):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.md")):
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    if repeat > 1:
        chunks = chunks * repeat
    return chunks


def make_batches(
    data: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    max_start = max(1, data.numel() - seq_len - 1)
    for _ in range(n_batches):
        starts = torch.randint(0, max_start, (batch_size,), generator=g)
        rows = torch.stack([data[s : s + seq_len] for s in starts.tolist()], dim=0)
        out.append(rows)
    return out


def train_lm(
    model: TinyLM,
    batches: list[torch.Tensor],
    *,
    steps: int,
    lr: float = 3e-3,
    device: str = "cpu",
    log_every: int = 20,
) -> list[dict]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    curve: list[dict] = []
    model.train()
    it = 0
    while it < steps:
        for batch in batches:
            if it >= steps:
                break
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = lm_loss(logits, batch)
            from precision import assert_finite_loss, finite_check_enabled

            if finite_check_enabled():
                assert_finite_loss(loss, step=f"lm_step{it}")
            loss.backward()
            opt.step()
            if it % log_every == 0 or it == steps - 1:
                curve.append({"step": it, "loss": float(loss.detach()), "ppl": float(loss.detach().exp())})
            it += 1
    return curve


def config_dict(cfg: TinyLMConfig) -> dict:
    return asdict(cfg)


def with_most_knobs(cfg: TinyLMConfig, knobs: MostKnobs) -> TinyLMConfig:
    """Kopia cfg z innymi gałkami zamka."""
    return TinyLMConfig(
        vocab_size=cfg.vocab_size,
        d=cfg.d,
        h=cfg.h,
        num_layers=cfg.num_layers,
        n_dim=cfg.n_dim,
        ff_mult=cfg.ff_mult,
        max_seq=cfg.max_seq,
        dropout=cfg.dropout,
        most=knobs,
    )
