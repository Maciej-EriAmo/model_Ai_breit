"""Flagii PhiMost Faza E — wszystko OFF domyślnie."""

from __future__ import annotations

import os


def _on(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw not in ("0", "false", "off", "no", "", "disable", "disabled")


def live_enabled() -> bool:
    return _on("PHIMOST_LIVE", "0")


def neural_lambda_enabled() -> bool:
    return _on("PHIMOST_NEURAL_LAMBDA", "0")


def gen_enabled() -> bool:
    return _on("PHIMOST_GEN", "0")


def lambda_neural_weight() -> float:
    try:
        return float(os.environ.get("PHIMOST_LAMBDA_NEURAL", "0.2"))
    except ValueError:
        return 0.2


def phimost_root() -> str:
    return os.environ.get("PHIMOST_ROOT", r"C:\Users\drwis\model Ai")
