"""Wstaw root labu (+ archiwum/skrypty) na sys.path — testy w doswiadczenie/testy/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_ARCH_SKRYPTY = ROOT / "archiwum" / "skrypty"
for _p in (ROOT, _ARCH_SKRYPTY):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
