"""Tiny causal LM — jądro Breita–Wignera vs Softmax.

Most-1: C = softplus²(QK), delta_E z pozycji, gamma = gamma0 * (1 + beta * max(C/sum C)).
Softmax: klasyczny causal attn.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from breit_kernel import breit_wigner_rows


@dataclass(frozen=True)
class MostKnobs:
    """Gałki Most-1. beta_init=0 to beta_raw, więc beta = softplus(0) ≈ 0.69, nie zero.

    Ramię β=0: beta_exact=0. Szerokość jest wtedy γ0 na każdym wierszu.
    width_scale='excess': płaski wiersz startuje od γ0, niezależnie od liczby kluczy.
    dE[i,j] = E_i - E_j. Do wagi wchodzi dE². delta_mode poza 'square' jest odrzucony.
    """

    gamma_init: float = 0.0  # init log_gamma0; softplus(0)≈0.69
    beta_init: float = 0.0  # init beta_raw, nie wartość β
    e_scale: float = 1.0  # skala ΔE po projekcji e()
    couple_temp: float = 1.0  # τ: softplus(S/τ); >1 → gładsze C
    delta_mode: str = "square"
    delta_alpha: float = 1.0
    directed_bias: float = 0.25
    beta_exact: float | None = None  # 0 = ramię bez szerokości zależnej od wiersza
    width_scale: str = "kappa"  # "kappa" historyczna, "excess" skala startowa γ0


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
    side_sources: int = 0  # 0 = brak pluginu źródła w tokenie
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
            per += nd * h + h + h  # e + log_gamma0 + beta
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
        # F.dropout, nie nn.Dropout: p=0 nie dokłada parametrów do state_dict.
        self.drop_p = float(cfg.dropout)

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_p > 0.0:
            return F.dropout(x, self.drop_p, self.training)
        return x

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        S = S.masked_fill(causal_mask(N, x.device).view(1, 1, N, N), -1e9)
        w = F.softmax(S, dim=-1)
        y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = self._drop(x + self.o(y))
        x = self._drop(x + self.ff(x))
        if return_attn:
            return x, w
        return x


class _MostCausalBlock(nn.Module):
    """Most-1 Breita–Wignera + maska przyczynowa (bez zerowania diagonali — LM)."""

    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        d, h, nd = cfg.d, cfg.h, cfg.n_dim
        assert d % h == 0
        kn = cfg.most
        self.h, self.dh = h, d // h
        self.e_scale = float(kn.e_scale)
        self.couple_temp = max(float(kn.couple_temp), 1e-3)
        delta_mode = str(kn.delta_mode)
        if delta_mode != "square":
            raise ValueError(
                "ten fork ma jądro Breita–Wignera: delta_mode musi być 'square'"
            )
        # Pola zostają w MostKnobs (skrypty archiwum je przekazują), ale nie
        # wchodzą do jądra i nie są stanem modułu.
        if float(kn.delta_alpha) != 1.0 or float(kn.directed_bias) != 0.25:
            raise ValueError(
                "delta_alpha i directed_bias nie wchodzą do jądra Breita–Wignera; "
                "zostaw wartości domyślne"
            )
        self.drop_p = float(cfg.dropout)
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.e = nn.Linear(nd, h)
        nn.init.ones_(self.e.weight)
        nn.init.zeros_(self.e.bias)
        self.log_gamma = nn.Parameter(torch.full((h, 1, 1), float(kn.gamma_init)))
        self.beta_raw = nn.Parameter(torch.full((h, 1, 1), float(kn.beta_init)))
        self.beta_exact = kn.beta_exact
        self.width_scale = str(kn.width_scale)
        if kn.beta_exact is not None:
            self.beta_raw.requires_grad_(False)
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.ff_mult * d),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * d, d),
        )

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_p > 0.0:
            return F.dropout(x, self.drop_p, self.training)
        return x

    def _most_weights(self, S: torch.Tensor, dE: torch.Tensor) -> torch.Tensor:
        """C z QK, potem Most-1. Przyszłe klucze są zerem zanim liczone jest kappa."""
        C = F.softplus(S / self.couple_temp).square()
        blocked = causal_mask(S.shape[-1], S.device).view(1, 1, S.shape[-1], S.shape[-1])
        C = C.masked_fill(blocked, 0.0)
        return breit_wigner_rows(
            C,
            dE,
            self.log_gamma,
            self.beta_raw,
            beta_exact=self.beta_exact,
            scale=self.width_scale,
        )

    def forward(self, x: torch.Tensor, tracer: torch.Tensor, return_attn: bool = False):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        E = self.e(tracer).transpose(1, 2).unsqueeze(-1)  # [B,h,N,1] — może być ujemne
        dE = self.e_scale * (E - E.transpose(-2, -1))
        A = self._most_weights(S, dE)
        w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
        from precision import assert_finite

        assert_finite(S, "lm.S")
        assert_finite(dE, "lm.dE")
        assert_finite(A, "lm.A")
        assert_finite(w, "lm.w")
        y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = self._drop(x + self.o(y))
        x = self._drop(x + self.ff(x))
        if return_attn:
            return x, w
        return x


def resolve_lm_arm(attn: str, cfg: TinyLMConfig) -> tuple[str, str, TinyLMConfig]:
    """Nazwa ramienia → (jądro most|softmax, nazwa, konfiguracja).

    most: historyczne κ i β = softplus(beta_raw).
    beta0: β = 0, γ = γ0 na każdym wierszu.
    excess: płaski wiersz startuje od γ0; β zostaje uczone.
    """
    if attn == "beta0":
        return "most", "beta0", replace(cfg, most=replace(cfg.most, beta_exact=0.0))
    if attn == "excess":
        return (
            "most",
            "excess",
            replace(cfg, most=replace(cfg.most, width_scale="excess")),
        )
    if attn in ("softmax", "most"):
        return attn, attn, cfg
    raise ValueError(attn)


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig, attn: str = "softmax"):
        super().__init__()
        kernel, arm, cfg = resolve_lm_arm(attn, cfg)
        self.cfg = cfg
        self.arm = arm
        self.attn = kernel
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d)
        n_side = int(getattr(cfg, "side_sources", 0) or 0)
        self.side = nn.Embedding(n_side, cfg.d) if n_side else None
        if kernel == "most":
            self.blocks = nn.ModuleList([_MostCausalBlock(cfg) for _ in range(cfg.num_layers)])
            self.tracer_mix = nn.Linear(cfg.d + cfg.n_dim, cfg.n_dim)
        else:
            self.blocks = nn.ModuleList([_SoftmaxCausalBlock(cfg) for _ in range(cfg.num_layers)])
            self.tracer_mix = None
        self.out_norm = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tie

    def forward(self, idx: torch.Tensor, return_attn: bool = False, side: torch.Tensor | None = None, zero_de: bool = False):
        """idx [B,N] → logits [B,N,V]; opcjonalnie lista wag [B,h,N,N] per warstwa.

        side: numer źródła [B,N], dodawany do tokenu. zero_de: Most bez ΔE (tracer zerowy).
        Domyślnie zachowanie dotychczasowe.
        """
        B, N = idx.shape
        if N > self.cfg.max_seq:
            raise ValueError(f"seq {N} > max_seq {self.cfg.max_seq}")
        pos = torch.arange(N, device=idx.device).unsqueeze(0).expand(B, N)
        x = self.tok(idx) + self.pos(pos)
        if side is not None:
            if self.side is None:
                raise ValueError("side podane, a model nie ma embeddingu źródła")
            x = x + self.side(side)
        attn_ws: list[torch.Tensor] = []
        if self.attn == "most":
            if zero_de:
                tracer = torch.zeros(B, N, self.cfg.n_dim, device=idx.device)
            else:
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
    """Kopia cfg z innymi gałkami zamka. Reszta pól (side_sources, dropout) zostaje."""
    return replace(cfg, most=knobs)
