"""
Dane kwantowo-mechaniczne → przestrzeń Φ pod Most.

Hierarchia (obserwator / tracer):
  powłoka n     → depth / bubble
  podpowłoka ℓ  → bin w tracerze + mazur
  orbital (m,s) → atom

Content (x): liczby kwantowe / zajętość / etykieta orbitalu (nie „płaski tekst chemii”).
Tracer (τ): n, ℓ, energia przybliżona, spin — ΔE = warunek Mostu.

Aufbau (uproszczony): 1s 2s 2p 3s 3p 4s 3d 4p …
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState

ROOT = Path(__file__).resolve().parent

# (n, ℓ) w kolejności Aufbau + przybliżona energia (Hartree-ish skala względna)
AUFBAU = [
    (1, 0),  # 1s
    (2, 0),  # 2s
    (2, 1),  # 2p
    (3, 0),  # 3s
    (3, 1),  # 3p
    (4, 0),  # 4s
    (3, 2),  # 3d
    (4, 1),  # 4p
    (5, 0),  # 5s
    (4, 2),  # 4d
]

SUBSHELL = {0: "s", 1: "p", 2: "d", 3: "f"}
# pojemność podpowłoki: 2*(2ℓ+1)
CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}

# przykładowe Z do seedów (elektrony)
ELEMENTS = {
    "H": 1,
    "He": 2,
    "C": 6,
    "N": 7,
    "O": 8,
    "Ne": 10,
    "Na": 11,
    "Si": 14,
    "Fe": 26,
    "Cu": 29,
}


def _approx_energy(n: int, ell: int) -> float:
    """Ujemna energia efektywna ~ -1/(n+ℓ)^2 (skala względna)."""
    return -1.0 / float((n + ell) ** 2)


def _depth_from_n(n: int) -> int:
    return int(max(0, min(2, n - 1)))  # 1→0, 2→1, ≥3→2


def fill_orbitals(Z: int) -> list[tuple[int, int, int, int]]:
    """
    Zwraca listę zajętych orbitali (n, ℓ, m, spin) — spin: 0=↑ 1=↓.
    Hund: najpierw równoległe spiny w podpowłoce, potem pary.
    """
    left = Z
    occupied: list[tuple[int, int, int, int]] = []
    for n, ell in AUFBAU:
        if left <= 0:
            break
        ms = list(range(-ell, ell + 1))
        # runda 1: po jednym ↑ na każde m
        for m in ms:
            if left <= 0:
                break
            occupied.append((n, ell, m, 0))
            left -= 1
        # runda 2: ↓ na m
        for m in ms:
            if left <= 0:
                break
            occupied.append((n, ell, m, 1))
            left -= 1
    return occupied


def element_to_graph(symbol: str, Z: int | None = None) -> PhiGraph:
    Z = ELEMENTS.get(symbol, Z)
    if Z is None:
        raise ValueError(f"nieznany pierwiastek {symbol!r}")
    occ = fill_orbitals(Z)
    atoms: list[PhiAtom] = []
    # nucleus / context
    atoms.append(
        PhiAtom(
            atom_id=f"{symbol}/nucleus",
            S=f"nucleus/Z{Z}",
            E=f"{symbol} Z={Z} QM context",
            T=1.0,
            state=PhiState.HOT,
            bubble_id=f"{symbol}/n0",
            depth=0,
            payload=f"nucleus {symbol}",
            mazur_tracer=0.05,
        )
    )
    for n, ell, m, spin in occ:
        sub = SUBSHELL[ell]
        e = _approx_energy(n, ell)
        # T: energia + offset m → ΔE wewnątrz podpowłoki (inaczej Most≈SM)
        T = min(1.0, max(0.12, abs(e) * 8.0 + 0.04 * (m + ell)))
        mazur = 0.7 * ((ell + 0.5) / 4.0) + 0.3 * ((Z % 17) / 17.0)
        spin_s = "up" if spin == 0 else "down"
        aid = f"{symbol}/{n}{sub}/m{m}/{spin_s}"
        st = PhiState.HOT if spin == 0 else PhiState.WARM
        depth = min(2, _depth_from_n(n) + (1 if ell >= 2 else 0))
        # pierwiastek w S/E — inaczej C/2p ≡ O/2p
        atoms.append(
            PhiAtom(
                atom_id=aid,
                S=f"{symbol}/e-/n{n}/l{ell}/m{m}/s{spin}",
                E=f"{symbol} {n}{sub} m={m} spin={spin_s} E≈{e:.4f}",
                T=T,
                state=st,
                bubble_id=f"{symbol}/n{n}/{sub}",
                depth=depth,
                payload=f"{symbol} orbital {n}{sub} m={m} {spin_s}",
                mazur_tracer=mazur,
            )
        )

    edges: list[PhiEdge] = []
    # parent: orbital → nucleus
    for a in atoms[1:]:
        edges.append(PhiEdge("parent", a.atom_id, atoms[0].atom_id, name="bound"))
    # bind: ta sama podpowłoka (n,ℓ) — z listy zajęć, bez parsowania S
    by_sub: dict[tuple[int, int], list[str]] = {}
    by_n: dict[int, dict[int, list[str]]] = {}
    for n, ell, m, spin in occ:
        sub = SUBSHELL[ell]
        spin_s = "up" if spin == 0 else "down"
        aid = f"{symbol}/{n}{sub}/m{m}/{spin_s}"
        by_sub.setdefault((n, ell), []).append(aid)
        by_n.setdefault(n, {}).setdefault(ell, []).append(aid)
    for (n, ell), ids in by_sub.items():
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                edges.append(
                    PhiEdge("bind", a, b, name=f"subshell-{n}{SUBSHELL[ell]}")
                )
    # bind: ta sama powłoka n między różnymi ℓ
    for n, subs in by_n.items():
        ells = list(subs.keys())
        for i, e1 in enumerate(ells):
            for e2 in ells[i + 1 :]:
                for a in subs[e1][:2]:
                    for b in subs[e2][:2]:
                        edges.append(PhiEdge("bind", a, b, name=f"shell-n{n}"))

    return PhiGraph(
        atoms=atoms,
        edges=edges,
        meta={"source": "qm_orbitals", "symbol": symbol, "Z": Z, "n_e": len(occ)},
    )


def multi_element_graph(symbols: list[str] | None = None) -> PhiGraph:
    symbols = symbols or ["H", "He", "C", "O", "Ne", "Fe"]
    atoms: list[PhiAtom] = []
    edges: list[PhiEdge] = []
    for sym in symbols:
        g = element_to_graph(sym)
        atoms.extend(g.atoms)
        edges.extend(g.edges)
        # hologram-like: nucleus ↔ valence hint
        nuc = next(a for a in g.atoms if a.atom_id.endswith("/nucleus"))
        valence = [a for a in g.atoms if a.depth == max(x.depth for x in g.atoms)]
        for v in valence[:4]:
            edges.append(PhiEdge("bind", nuc.atom_id, v.atom_id, name="valence"))
    return PhiGraph(
        atoms=atoms,
        edges=edges,
        meta={"source": "qm_orbitals_multi", "symbols": symbols},
    )


def export_phi_dump(
    out_dir: str | Path | None = None,
    symbols: list[str] | None = None,
) -> Path:
    out_dir = Path(out_dir) if out_dir else ROOT / "dumps" / "qm_orbitals"
    out_dir.mkdir(parents=True, exist_ok=True)
    g = multi_element_graph(symbols)
    with (out_dir / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in g.atoms:
            f.write(
                json.dumps(
                    {
                        "id": a.atom_id,
                        "S": a.S,
                        "E": a.E,
                        "T": a.T,
                        "state": a.state.name,
                        "bubble": a.bubble_id,
                        "depth": a.depth,
                        "payload": a.payload,
                        "mazur_tracer": a.mazur_tracer,
                        "topic": a.bubble_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (out_dir / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in g.edges:
            f.write(
                json.dumps(
                    {"kind": e.kind, "src": e.src, "dst": e.dst, "name": e.name},
                    ensure_ascii=False,
                )
                + "\n"
            )
    man = {
        "format": "phijson-v1",
        "description": "QM shells/subshells/orbitals → Φ (Aufbau)",
        "n_atoms": len(g.atoms),
        "n_edges": len(g.edges),
        "symbols": g.meta.get("symbols"),
        "source": "qm_orbitals_data",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "CASE_QM.md").write_text(
        "\n".join(
            [
                "# Case — elektrony na powłokach → Φ",
                "",
                "| QM | Φ / Most |",
                "|----|----------|",
                "| powłoka n | depth / bubble `n{n}` |",
                "| podpowłoka ℓ (s,p,d,f) | mazur_tracer + bind subshell |",
                "| orbital (m, spin) | atom `(S,E,T)` |",
                "| energia ≈ −1/(n+ℓ)² | T + treść E |",
                "",
                f"Atoms={len(g.atoms)}, edges={len(g.edges)}, symbols={g.meta.get('symbols')}",
                "",
                "Nauka: `python finetune_karmazyn_live.py` nie — użyj dumpa QM:",
                "`python finetune_qm_orbitals.py` albo",
                "`python smoke_checkpoint.py --dump dumps/qm_orbitals --force`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_dir


def main():
    import argparse

    ap = argparse.ArgumentParser(description="QM orbitals → PhiJSON")
    ap.add_argument(
        "--symbols",
        default="H,He,C,N,O,Ne,Na,Si,Fe,Cu",
        help="lista pierwiastków CSV",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "dumps" / "qm_orbitals")
    args = ap.parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out = export_phi_dump(args.out, syms)
    g = multi_element_graph(syms)
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    print(f"QM Φ → {out}  atoms={len(g.atoms)} bind≈{n_bind} symbols={syms}")


if __name__ == "__main__":
    main()
