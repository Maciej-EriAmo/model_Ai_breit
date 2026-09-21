"""Ścieżki labu dla skryptów w archiwum/skrypty/."""
from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
ARCH_SKRYPTY = Path(__file__).resolve().parent
ARCH_DANE = LAB_ROOT / "archiwum" / "dane"
ARCH_HISTORIA = LAB_ROOT / "archiwum" / "historia"
ARCH_ZDJECIEW = LAB_ROOT / "archiwum" / "ZdjęcieW"

if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
