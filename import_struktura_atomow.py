"""
Wczytaj struktura_atomow.csv / .json → PhiJSON (powłoki/podpowłoki/orbitale).

Format wiersza:
  atom_number, symbol, name, group, period,
  electron_configuration, electrons_per_shell

Przykład konfiguracji: \"[He] 2s2 2p2\" albo \"1s1\".

  python import_struktura_atomow.py
  python import_struktura_atomow.py --src struktura_atomow.json --out dumps/from_struktura
  python archiwum/skrypty/finetune_qm_orbitals.py --skip-export ...  # albo poniżej --finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState
from qm_orbitals_data import SUBSHELL, _approx_energy, _depth_from_n

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "archiwum" / "dane" / "struktura_atomow.csv"
DEFAULT_JSON = ROOT / "archiwum" / "dane" / "struktura_atomow.json"
DEFAULT_OUT = ROOT / "dumps" / "from_struktura"

# rdzenie gazów szlachetnych (rozszerzenie [He], [Ne], …)
_XE = "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6"
NOBLE_CORE = {
    "He": "1s2",
    "Ne": "1s2 2s2 2p6",
    "Ar": "1s2 2s2 2p6 3s2 3p6",
    "Kr": "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6",
    "Xe": _XE,
    # Rn = Xe + 4f14 5d10 6s2 6p6 (86 elektronów). Bez tego pełna tablica pada na [Rn].
    "Rn": _XE + " 4f14 5d10 6s2 6p6",
}

_TERM = re.compile(r"(\d+)([spdf])(\d+)", re.I)
_CORE = re.compile(r"\[([A-Za-z]+)\]")


def expand_configuration(cfg: str) -> list[tuple[int, int, int]]:
    """
    \"[He] 2s2 2p2\" → [(n,ℓ,count), ...]
    ℓ: s=0 p=1 d=2 f=3
    """
    cfg = (cfg or "").strip()
    if not cfg:
        return []
    m = _CORE.search(cfg)
    if m:
        noble = m.group(1)
        core = NOBLE_CORE.get(noble)
        if core is None:
            raise ValueError(f"nieznany rdzeń [{noble}] w {cfg!r}")
        cfg = _CORE.sub(core, cfg, count=1)
    out: list[tuple[int, int, int]] = []
    ell_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    for n_s, L, cnt_s in _TERM.findall(cfg):
        n, ell, cnt = int(n_s), ell_map[L.lower()], int(cnt_s)
        out.append((n, ell, cnt))
    if not out:
        raise ValueError(f"nieparsowalna konfiguracja: {cfg!r}")
    return out


def subshell_to_orbitals(
    symbol: str,
    Z: int,
    terms: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Hund: najpierw ↑ na każde m, potem ↓. Zwraca (n,ℓ,m,spin)."""
    occupied: list[tuple[int, int, int, int]] = []
    left_total = Z
    for n, ell, cnt in terms:
        ms = list(range(-ell, ell + 1))
        placed = 0
        # ↑
        for m in ms:
            if placed >= cnt or left_total <= 0:
                break
            occupied.append((n, ell, m, 0))
            placed += 1
            left_total -= 1
        # ↓
        for m in ms:
            if placed >= cnt or left_total <= 0:
                break
            occupied.append((n, ell, m, 1))
            placed += 1
            left_total -= 1
    return occupied


def row_to_graph(row: dict) -> PhiGraph:
    symbol = str(row["symbol"]).strip()
    Z = int(row["atom_number"])
    name = str(row.get("name") or symbol)
    group = int(row.get("group") or 0)
    period = int(row.get("period") or 0)
    cfg = str(row["electron_configuration"])
    terms = expand_configuration(cfg)
    occ = subshell_to_orbitals(symbol, Z, terms)

    atoms: list[PhiAtom] = [
        PhiAtom(
            atom_id=f"{symbol}/nucleus",
            S=f"nucleus/Z{Z}/g{group}/p{period}",
            E=f"{name} ({symbol}) Z={Z} cfg={cfg}",
            T=1.0,
            state=PhiState.HOT,
            bubble_id=f"{symbol}/core",
            depth=0,
            payload=f"{name}; shells={row.get('electrons_per_shell')}",
            mazur_tracer=0.05,
        )
    ]
    for n, ell, m, spin in occ:
        sub = SUBSHELL[ell]
        e = _approx_energy(n, ell)
        T = min(1.0, max(0.12, abs(e) * 8.0 + 0.04 * (m + ell)))
        mazur = 0.7 * ((ell + 0.5) / 4.0) + 0.3 * ((Z % 17) / 17.0)
        spin_s = "up" if spin == 0 else "down"
        aid = f"{symbol}/{n}{sub}/m{m}/{spin_s}"
        st = PhiState.HOT if spin == 0 else PhiState.WARM
        depth = min(2, _depth_from_n(n) + (1 if ell >= 2 else 0))
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
    for a in atoms[1:]:
        edges.append(PhiEdge("parent", a.atom_id, atoms[0].atom_id, name="bound"))
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
                edges.append(PhiEdge("bind", a, b, name=f"subshell-{n}{SUBSHELL[ell]}"))
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
        meta={
            "source": "struktura_atomow",
            "symbol": symbol,
            "Z": Z,
            "name": name,
            "cfg": cfg,
            "n_e": len(occ),
        },
    )


def load_rows(src: Path) -> list[dict]:
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.suffix.lower() == ".json":
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON musi być listą obiektów pierwiastków")
        return data
    if src.suffix.lower() == ".csv":
        with src.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"obsługiwane: .csv / .json, nie {src.suffix}")


def rows_to_graph(rows: list[dict]) -> PhiGraph:
    atoms: list[PhiAtom] = []
    edges: list[PhiEdge] = []
    symbols = []
    for row in rows:
        g = row_to_graph(row)
        atoms.extend(g.atoms)
        edges.extend(g.edges)
        symbols.append(g.meta["symbol"])
        nuc = next(a for a in g.atoms if a.atom_id.endswith("/nucleus"))
        valence = [a for a in g.atoms[1:] if a.depth == max(x.depth for x in g.atoms[1:])]
        for v in valence[:4]:
            edges.append(PhiEdge("bind", nuc.atom_id, v.atom_id, name="valence"))
    return PhiGraph(
        atoms=atoms,
        edges=edges,
        meta={"source": "struktura_atomow", "symbols": symbols},
    )


def export_dump(rows: list[dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = rows_to_graph(rows)
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
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    man = {
        "format": "phijson-v1",
        "description": "struktura_atomow.csv/json → Φ (konfiguracja elektronowa)",
        "n_atoms": len(g.atoms),
        "n_edges": len(g.edges),
        "n_bind": n_bind,
        "symbols": g.meta.get("symbols"),
        "source": "import_struktura_atomow",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "CASE_STRUKTURA.md").write_text(
        "\n".join(
            [
                "# Import struktura_atomow → Φ",
                "",
                f"Symbols: {g.meta.get('symbols')}",
                f"Atoms={len(g.atoms)} bind≈{n_bind}",
                "",
                "Nauka:",
                "```bat",
                "python import_struktura_atomow.py",
                "python finetune_struktura_atomow.py --no-warm --no-promote",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="struktura_atomow → PhiJSON")
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="CSV lub JSON (domyślnie: json jeśli jest, else csv)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    src = args.src
    if src is None:
        src = DEFAULT_JSON if DEFAULT_JSON.is_file() else DEFAULT_CSV
    rows = load_rows(src)
    out = export_dump(rows, args.out)
    g = rows_to_graph(rows)
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    print(
        f"import {src.name} → {out}  "
        f"atoms={len(g.atoms)} bind≈{n_bind} symbols={g.meta.get('symbols')}"
    )


if __name__ == "__main__":
    main()
