"""Pełna tablica (118) jako graf Φ — jądro + topologia (grupa/okres), bez wszystkich orbitali.

Dlaczego nie pełne orbitale Z=1..118: ~7k atomów → attention O(N²) nie wejdzie w 2–3 h.
Tu: 118 atomów (nucleus) + krawędzie same-group / same-period / sąsiedzi Z.
Osie w S/E/T/mazur: group, period, block, Z — nośnik wielowymiarowy.

  python make_periodic_full.py
  python export_mendelejew_full.py
  python run_mendelejew_full.py
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
from pathlib import Path

from make_periodic_full import build_rows
from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "dumps" / "from_mendelejew_full"


def _block(group: int, period: int, Z: int) -> str:
    if Z in range(57, 72) or Z in range(89, 104):
        return "f"
    if group in (1, 2) or (group == 18 and Z == 2):
        return "s"
    if 3 <= group <= 12:
        return "d"
    return "p"


def rows_to_table_graph(rows: list[dict]) -> PhiGraph:
    atoms: list[PhiAtom] = []
    by_sym: dict[str, PhiAtom] = {}
    by_Z: dict[int, PhiAtom] = {}
    meta_rows = []
    for r in rows:
        Z = int(r["atom_number"])
        sym = str(r["symbol"])
        group = int(r["group"])
        period = int(r["period"])
        block = _block(group, period, Z)
        name = str(r["name"])
        # T ~ okres (płytsze = lżejsze); mazur ~ grupa znormalizowana
        T = min(1.0, 0.15 + 0.12 * period)
        mazur = group / 18.0
        a = PhiAtom(
            atom_id=f"{sym}/nucleus",
            S=f"element/Z{Z}/g{group}/p{period}/block:{block}",
            E=f"{name} ({sym}) Z={Z} group={group} period={period} block={block}",
            T=T,
            state=PhiState.HOT,
            bubble_id=f"period/{period}",
            depth=min(2, max(0, period - 1) // 3),
            payload=f"{sym};Z={Z};g={group};p={period};block={block};cfg={r.get('electron_configuration')}",
            mazur_tracer=mazur,
        )
        atoms.append(a)
        by_sym[sym] = a
        by_Z[Z] = a
        meta_rows.append({"Z": Z, "symbol": sym, "group": group, "period": period, "block": block})

    edges: list[PhiEdge] = []
    # sąsiedzi Z (okresowy ciąg)
    for Z in range(1, 118):
        if Z in by_Z and (Z + 1) in by_Z:
            edges.append(
                PhiEdge("bind", by_Z[Z].atom_id, by_Z[Z + 1].atom_id, name="Z-adjacent")
            )
    # ta sama grupa (pion)
    by_group: dict[int, list[PhiAtom]] = {}
    for a, m in zip(atoms, meta_rows):
        by_group.setdefault(m["group"], []).append(a)
    for g, lst in by_group.items():
        lst_sorted = sorted(lst, key=lambda x: int(x.S.split("/Z")[1].split("/")[0]))
        for i in range(len(lst_sorted) - 1):
            edges.append(
                PhiEdge(
                    "bind",
                    lst_sorted[i].atom_id,
                    lst_sorted[i + 1].atom_id,
                    name=f"group-{g}",
                )
            )
    # ten sam okres (poziom) — tylko sąsiedzi w sortowaniu Z już pokryci;
    # dodaj bind co 2 w okresie dla gęstości (opcjonalnie skip)
    by_period: dict[int, list[tuple[int, PhiAtom]]] = {}
    for a, m in zip(atoms, meta_rows):
        by_period.setdefault(m["period"], []).append((m["Z"], a))
    for p, lst in by_period.items():
        lst = sorted(lst, key=lambda t: t[0])
        for i in range(len(lst) - 1):
            a, b = lst[i][1], lst[i + 1][1]
            # unikaj duplikatu Z-adjacent
            edges.append(
                PhiEdge("bind", a.atom_id, b.atom_id, name=f"period-{p}")
            )

    # dedupe edges
    seen = set()
    uniq = []
    for e in edges:
        key = (e.kind, min(e.src, e.dst), max(e.src, e.dst), e.name)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    return PhiGraph(
        atoms=atoms,
        edges=uniq,
        meta={
            "source": "mendelejew_full_table",
            "n_elements": 118,
            "mode": "nucleus+topology",
            "symbols": [m["symbol"] for m in meta_rows],
        },
    )


def export(out: Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    g = rows_to_table_graph(rows)
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
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
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
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
        "description": "Mendelejew full 118 — nucleus + group/period topology",
        "n_atoms": len(g.atoms),
        "n_edges": len(g.edges),
        "n_elements": 118,
        "mode": "nucleus+topology",
        "source": "export_mendelejew_full.py",
        "symbols": g.meta["symbols"],
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "CASE_MENDELEJEW_FULL.md").write_text(
        "\n".join(
            [
                "# Mendelejew full (118)",
                "",
                f"Atoms={len(g.atoms)} edges={len(g.edges)} mode=nucleus+topology",
                "",
                "Nauka (jutro, ~2–3 h):",
                "```bat",
                "python make_periodic_full.py",
                "python export_mendelejew_full.py",
                "python run_mendelejew_full.py",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"OK {out} atoms={len(g.atoms)} edges={len(g.edges)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    # upewnij się że CSV full istnieje
    from make_periodic_full import main as gen

    gen()
    export(args.out)


if __name__ == "__main__":
    main()
