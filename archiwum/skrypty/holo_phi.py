"""
Holo-Φ — znak jako fala (amp, faza); słowo / rdzeń = interferencja.

P0 definicje A i T (po audycie ocena_holo):
  A = energia fali znaku z CZĘSTOŚCI w korpusie lab (+ boost rdzenia),
      NIE z hasha. Hash zostaje tylko dla fazy φ.
  T = ciepło uwagi z ENERGII FALI na wspólnej skali:
      znak:  T = f(A)
      compound: T = f(|Σ|)
      + jawny fokus czytania (pierwsza litera słowa)
      + archive T := T_TOMB (stała, nie z A)
  Seal (most_seal) bramkuje niskie T = niską energię fali / fokus —
      to jest zamierzone; nie sterujemy uwagą losowym hashem.

Mapowanie Φ: mazur = φ/2π; depth page0 / compound1 / glyph2.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from transcode.phi_build import state_for_T

# --- stałe kanonu A/T --------------------------------------------------------

A_FLOOR = 0.35  # rzadki znak nadal niesie falę
A_ROOT_BOOST = 0.85  # minimum A dla członków rodziny rdzenia
T_TOMB = 0.08  # archive / świadomy zimny
T_FOCUS_FIRST = 0.90  # pierwsza litera słowa (fokus czytania)
ALPHA_ROOT_OFF = 0.35  # offset fazy pochodnej względem φ_root


def _phase(s: str, *, salt: str = "") -> float:
    """Faza: stabilny hash etykiety (to OK — faza ≠ saliency)."""
    h = hashlib.sha256(f"{salt}:{s}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "little") / float(2**64)
    return float(u * 2.0 * math.pi)


def amp_from_freq(freq: int, freq_max: int, *, root_member: bool = False) -> float:
    """
    A ∈ [A_FLOOR, 1]: częstość znaku w korpusie lab / max.
    root_member → max(A, A_ROOT_BOOST).
    """
    if freq_max <= 0:
        a = A_FLOOR
    else:
        a = A_FLOOR + (1.0 - A_FLOOR) * (float(freq) / float(freq_max))
    if root_member:
        a = max(a, A_ROOT_BOOST)
    return float(np.clip(a, A_FLOOR, 1.0))


def energy_to_T(energy: float, *, ref: float) -> float:
    """T z energii fali (A lub |Σ|) na wspólnej skali ref."""
    if ref <= 1e-9:
        return 0.5
    return float(np.clip(0.25 + 0.7 * (energy / ref), T_TOMB, 0.98))


@dataclass
class Wave:
    label: str
    amp: float
    phase: float  # rad

    @property
    def z(self) -> complex:
        return self.amp * complex(math.cos(self.phase), math.sin(self.phase))


def wave_for_char(
    ch: str,
    *,
    amp: float,
    root_phase: float | None = None,
) -> Wave:
    if root_phase is None:
        return Wave(label=ch, amp=amp, phase=_phase(ch))
    return Wave(
        label=ch,
        amp=amp,
        phase=(root_phase + ALPHA_ROOT_OFF * _phase(ch, salt="root-off"))
        % (2.0 * math.pi),
    )


def interfere(waves: list[Wave]) -> tuple[complex, float, float]:
    s = 0j
    for w in waves:
        s += w.z
    mag = abs(s)
    ang = math.atan2(s.imag, s.real) % (2.0 * math.pi)
    return s, mag, ang


def e_signature(z: complex, n: int, kind: str, *, amp: float | None = None) -> str:
    extra = f" A={amp:.3f}" if amp is not None else ""
    return (
        f"holo/{kind} n={n} re={z.real:.4f} im={z.imag:.4f} "
        f"mag={abs(z):.4f}{extra}"
    )


def build_holo_graph(
    *,
    root_shared_phase: bool = True,
    root_binds: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """
    P3 ablacja (anty-tautologia):
      root_shared_phase — czy rodzina 女 dzieli φ_root (vs niezależny hash/znak)
      root_binds — czy krawędzie holo:root:女 są w grafie
    Porównanie phase± przy **stałych** binds izoluje wkład fazy bez „wyjęcia odpowiedzi
    z grafu”. Softmax (bez τ) powinien być prawie niewrażliwy na phase±.
    """
    atoms: list[dict] = []
    edges: list[dict] = []

    def bind(a: str, b: str, name: str) -> None:
        edges.append({"kind": "bind", "src": a, "dst": b, "name": name})
        edges.append({"kind": "bind", "src": b, "dst": a, "name": name})

    atoms.append(
        {
            "id": "corpus",
            "S": "holo/corpus",
            "E": (
                f"Holo-Φ P0/P3: A=freq; phase_shared={root_shared_phase} "
                f"root_binds={root_binds}"
            ),
            "T": 1.0,
            "state": "HOT",
            "bubble": "holo",
            "depth": 0,
            "payload": "holo_phi",
            "mazur_tracer": 0.5,
            "topic": "corpus",
        }
    )
    page = "page_holo"
    atoms.append(
        {
            "id": page,
            "S": "holo/page",
            "E": "page",
            "T": 0.92,
            "state": "HOT",
            "bubble": "holo",
            "depth": 0,
            "payload": None,
            "mazur_tracer": 0.5,
            "topic": "page",
        }
    )
    bind("corpus", page, "has_page")

    lat_words = ["MOST", "BRIDGE", "FOCUS", "TRACE", "SEAL", "WAVE", "ROOT", "PHASE"]
    hira_words = ["こんにちは", "ありがとう", "あいうえお"]
    root = "女"
    root_phase = _phase(root, salt="root") if root_shared_phase else None
    family = ["女", "安", "姿", "数", "好", "如"]
    family_set = set(family)

    # Korpus do częstości = wszystkie znaki w słowach + rodzina
    corpus_chars: list[str] = []
    for w in lat_words + hira_words:
        corpus_chars.extend(list(w))
    corpus_chars.extend(family)
    freq = Counter(corpus_chars)
    freq_max = max(freq.values()) if freq else 1

    def A_of(ch: str) -> float:
        return amp_from_freq(
            freq.get(ch, 0), freq_max, root_member=(ch in family_set)
        )

    compounds: list[tuple[str, complex, float, list[str], str]] = []

    for word in lat_words:
        waves = [wave_for_char(ch, amp=A_of(ch)) for ch in word]
        z, mag, ang = interfere(waves)
        compounds.append((f"lat/{word}", z, ang, list(word), "word"))

    for word in hira_words:
        waves = [wave_for_char(ch, amp=A_of(ch)) for ch in word]
        z, mag, ang = interfere(waves)
        compounds.append((f"hira/{word}", z, ang, list(word), "word"))

    root_waves = [
        wave_for_char(ch, amp=A_of(ch), root_phase=root_phase)  # None → niezależna φ
        for ch in family
    ]
    z_root, mag_root, ang_root = interfere(root_waves)
    compounds.append((f"root/{root}", z_root, ang_root, family, "root"))

    # wspólny ref energii: A znaków ∪ |Σ|
    energies = [A_of(ch) for ch in freq]
    energies.extend(abs(z) for _cid, z, _ang, _p, _k in compounds)
    ref = max(energies) if energies else 1.0

    glyph_ids_by_compound: dict[str, list[str]] = {}

    for cid, z, ang, parts, kind in compounds:
        T = energy_to_T(abs(z), ref=ref)
        st = state_for_T(T)
        lemma = cid.split("/", 1)[1]
        atoms.append(
            {
                "id": cid,
                "S": f"holo:{kind}:{lemma}",
                "E": e_signature(z, len(parts), kind),
                "T": T,
                "state": st,
                "bubble": f"holo/{kind}",
                "depth": 1,
                "payload": lemma if st == "HOT" else None,
                "mazur_tracer": float(ang / (2.0 * math.pi)),
                "topic": kind,
            }
        )
        bind(page, cid, f"has_{kind}")
        edges.append({"kind": "parent", "src": cid, "dst": page, "name": f"{kind}_of"})

        gids: list[str] = []
        for i, ch in enumerate(parts):
            a = A_of(ch)
            w = (
                wave_for_char(ch, amp=a, root_phase=root_phase)
                if kind == "root"
                else wave_for_char(ch, amp=a)
            )  # root_phase=None → φ tylko z hasha znaku
            gid = f"{cid}/ch/{i}/{ch}"
            gT = energy_to_T(a, ref=ref)
            if i == 0 and kind == "word":
                gT = max(gT, T_FOCUS_FIRST)  # fokus ≠ A
            gst = state_for_T(gT)
            atoms.append(
                {
                    "id": gid,
                    "S": f"holo:ch:{ch}",
                    "E": e_signature(w.z, 1, "char", amp=a),
                    "T": gT,
                    "state": gst,
                    "bubble": f"holo/{kind}/{lemma}",
                    "depth": 2,
                    "payload": ch if gst == "HOT" else None,
                    "mazur_tracer": float(w.phase / (2.0 * math.pi)),
                    "topic": lemma,
                    "A": a,
                    "freq": int(freq.get(ch, 0)),
                }
            )
            gids.append(gid)
            edges.append({"kind": "parent", "src": gid, "dst": cid, "name": "part_of"})
            bind(cid, gid, f"contains:{ch}")
        for a, b in zip(gids, gids[1:]):
            bind(a, b, "next")
        glyph_ids_by_compound[cid] = gids

    words = [c for c, *_r in compounds if c.startswith("lat/") or c.startswith("hira/")]
    for a, b in zip(words, words[1:]):
        bind(a, b, "next_word")

    root_gids = glyph_ids_by_compound.get(f"root/{root}", [])
    if root_binds:
        for i, a in enumerate(root_gids):
            for b in root_gids[i + 1 :]:
                bind(a, b, f"holo:root:{root}")

    # Archive: T stałe TOMB — nie pochodzi z A/freq (seal ma co ciąć)
    atoms.append(
        {
            "id": "holo/archive/X",
            "S": "holo:ch:X",
            "E": "archive false hologram T=T_TOMB fixed",
            "T": T_TOMB,
            "state": "TOMB",
            "bubble": "holo/archive",
            "depth": 2,
            "payload": None,
            "mazur_tracer": 0.01,
            "topic": "archive",
            "A": 0.0,
            "freq": 0,
        }
    )

    meta = {
        "method": "holo_phi",
        "A_def": "freq_in_lab_corpus + root_boost",
        "T_def": "energy_to_T(A or |Σ|, shared ref) + focus_first + archive T_TOMB",
        "phase_def": (
            "shared φ_root + offset"
            if root_shared_phase
            else "independent hash(ch) per family member"
        ),
        "root_shared_phase": root_shared_phase,
        "root_binds": root_binds,
        "n_lat_words": len(lat_words),
        "n_hira": len(hira_words),
        "root": root,
        "family": family,
        "family_glyph_ids": root_gids,
        "ref_energy": ref,
        "freq_max": freq_max,
        "A_FLOOR": A_FLOOR,
        "A_ROOT_BOOST": A_ROOT_BOOST,
    }
    return atoms, edges, meta
