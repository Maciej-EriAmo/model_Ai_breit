"""Precyzja treningu — kryterium sukcesu #2 (BF16).

- Domyślnie lab zostaje w FP32.
- `--bf16` / PHIMOST_BF16=1 → autocast bfloat16 + opcjonalny finite-check zamka.
- Gate: brak NaN/Inf w C, ΔE, A, w, loss, logits — bez ucieczki całego runu do FP32.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch

_FINITE_CHECK = False


def finite_check_enabled() -> bool:
    return _FINITE_CHECK


def set_finite_check(enabled: bool) -> None:
    global _FINITE_CHECK
    _FINITE_CHECK = bool(enabled)


def bf16_env_enabled() -> bool:
    raw = os.environ.get("PHIMOST_BF16", "0").strip().lower()
    return raw not in ("0", "false", "off", "no", "", "disable", "disabled")


def resolve_amp_dtype(spec: str | None) -> torch.dtype | None:
    """None → bez AMP (FP32). 'bf16'/'bfloat16' → bfloat16. 'fp16' → float16."""
    if spec is None or spec in ("", "fp32", "float32", "off", "0"):
        return None
    s = spec.strip().lower()
    if s in ("bf16", "bfloat16", "1", "true", "on"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"nieznany dtype AMP: {spec!r}")


def device_type(device: torch.device | str | None = None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.device(device)
    return d.type


def bf16_supported(device: torch.device | str | None = None) -> bool:
    """Czy bf16 da się realnie policzyć na urządzeniu (nie: czy CUDA ma tensor cores)."""
    dt = device_type(device)
    if dt == "cuda":
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False
    try:
        a = torch.randn(8, 8, dtype=torch.bfloat16)
        b = torch.randn(8, 8, dtype=torch.bfloat16)
        c = a @ b
        return bool(torch.isfinite(c).all().item())
    except Exception:
        return False


@contextmanager
def amp_context(
    dtype: torch.dtype | None,
    *,
    device: torch.device | str | None = None,
) -> Iterator[None]:
    if dtype is None:
        with nullcontext():
            yield
        return
    dt = device_type(device)
    # torch.amp.autocast — PyTorch 2.x
    with torch.amp.autocast(device_type=dt, dtype=dtype):
        yield


def assert_finite(t: torch.Tensor | None, name: str) -> None:
    if not _FINITE_CHECK or t is None:
        return
    if not torch.is_tensor(t):
        return
    if t.numel() == 0:
        return
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise RuntimeError(
            f"BF16/finite gate FAIL: non-finite in `{name}` "
            f"(bad={bad}/{t.numel()} dtype={t.dtype} shape={tuple(t.shape)})"
        )


def assert_finite_loss(loss: torch.Tensor, *, step: str = "loss") -> None:
    assert_finite(loss, step)
    if _FINITE_CHECK and (loss.detach().abs() > 1e6):
        # nie NaN, ale wybuch — też FAIL gate'u skali
        raise RuntimeError(
            f"BF16/finite gate FAIL: `{step}` abs={float(loss.detach())} > 1e6"
        )
