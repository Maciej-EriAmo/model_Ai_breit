"""Hold-out pierwiastka na FULL118 (nucleus+topology) — MOST-only.

Ukryj Na lub K w treningu na pełnej tablicy; po treningu wstrzyknij
`{sym}/nucleus` i rankinguj względem atomów train.
Sukces: kuzyn z tej samej grupy (najbliższy okres) wysoko w rankingu symboli.

  python archiwum/skrypty/run_mendelejew_holdout_full.py
  python archiwum/skrypty/run_mendelejew_holdout_full.py --holdouts Na,K --epochs 64
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import _graph_clean_tensors, load_dump, make_dump_loaders
from export_mendelejew_full import rows_to_table_graph
from make_periodic_full import build_rows
from phi_schema import PhiGraph
from recall_data import RecallBatch
from recall_model import PhiBridgeEncoder, train_encoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"
SEEDS_DEFAULT = [0, 1]
EPOCHS_DEFAULT = 64


def _expected_cousin(holdout: str, train_symbols: list[str], rows_by_sym: dict) -> str | None:
    """Ta sama grupa; wśród train najbliższy period (Na→Li, K→Na; H na końcu remisu)."""
    if holdout not in rows_by_sym:
        return None
    g = int(rows_by_sym[holdout].get("group") or 0)
    p0 = int(rows_by_sym[holdout].get("period") or 0)
    cousins = [
        s
        for s, r in rows_by_sym.items()
        if s in train_symbols and s != holdout and int(r.get("group") or 0) == g
    ]
    if not cousins:
        return None
    # remis |Δperiod|: preferuj lżejszy (niższy okres) — Na→Li nie K; K→Na nie Rb
    cousins.sort(
        key=lambda s: (
            abs(int(rows_by_sym[s].get("period") or 0) - p0),
            0 if s != "H" else 1,
            int(rows_by_sym[s].get("period") or 0),
            int(rows_by_sym[s].get("atom_number") or 0),
        )
    )
    return cousins[0]


def _write_graph_dump(g: PhiGraph, out: Path, *, holdout: str) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
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
        "description": f"Mendelejew FULL holdout — without {holdout}",
        "n_atoms": len(g.atoms),
        "n_edges": len(g.edges),
        "mode": "nucleus+topology",
        "holdout": holdout,
        "symbols": [a.atom_id.split("/", 1)[0] for a in g.atoms],
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def build_holdout_dump(holdout: str, rows: list[dict]) -> tuple[Path, dict, list[str]]:
    train_rows = [r for r in rows if str(r["symbol"]) != holdout]
    held = next(r for r in rows if str(r["symbol"]) == holdout)
    g = rows_to_table_graph(train_rows)
    out = ROOT / "dumps" / f"from_mendelejew_full_holdout_{holdout}"
    _write_graph_dump(g, out, holdout=holdout)
    train_syms = [str(r["symbol"]) for r in train_rows]
    print(f"dump train → {out} n={len(train_syms)} (held={holdout})", flush=True)
    return out, held, train_syms


def fit_most(dump: Path, seed: int, epochs: int, most_kw: dict | None = None) -> tuple:
    n_atoms = sum(1 for _ in (dump / "atoms.jsonl").open(encoding="utf-8"))
    bs = min(16, max(4, n_atoms))
    _, train, val, _ = make_dump_loaders(
        dump,
        n_train=200,
        n_val=50,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
    )
    torch.manual_seed(seed)
    kw = most_kw or {}
    model = PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
        **kw,
    )
    metrics = train_encoder(model, train, val, epochs=epochs, lr=CANON.lr, ablate_kw={})
    return model, metrics


@torch.no_grad()
def rank_holdout(model, train_dump: Path, held_row: dict) -> dict:
    _, g_train = load_dump(train_dump)
    g_held = rows_to_table_graph([held_row])
    atoms = list(g_train.atoms) + list(g_held.atoms)
    edges = list(g_train.edges) + list(g_held.edges)
    g = PhiGraph(atoms=atoms, edges=edges, meta={"holdout": held_row["symbol"]})
    content, tracer, depths, states = _graph_clean_tensors(
        g, CANON.d_x, CANON.n_dim, blend_holo_proto=0.0
    )
    content = content.unsqueeze(0)
    tracer = tracer.unsqueeze(0)
    depths = depths.unsqueeze(0)
    states = states.unsqueeze(0)
    mask = torch.ones(1, len(atoms), dtype=torch.bool)
    batch = RecallBatch(
        content=content,
        tracer=tracer,
        depths=depths,
        states=states,
        mask=mask,
        query_idx=torch.zeros(1, dtype=torch.long),
        pos_idx=torch.zeros(1, dtype=torch.long),
        topics=torch.zeros(1, len(atoms), dtype=torch.long),
    )

    model.eval()
    tok, _ = model(batch.content, batch.tracer, batch.mask)
    tok = tok[0]

    n_train = len(g_train.atoms)
    hold_sym = str(held_row["symbol"])
    qid = f"{hold_sym}/nucleus"
    q_idx = next(i for i, a in enumerate(atoms) if a.atom_id == qid)
    q = tok[q_idx : q_idx + 1]
    cand = tok[:n_train]
    sim = F.cosine_similarity(q, cand, dim=-1)
    order = torch.argsort(sim, descending=True).tolist()

    ranked = []
    for rank, i in enumerate(order, start=1):
        aid = g_train.atoms[i].atom_id
        ranked.append(
            {
                "rank": rank,
                "atom_id": aid,
                "symbol": aid.split("/", 1)[0],
                "sim": float(sim[i]),
            }
        )

    by_sym: dict[str, list[float]] = defaultdict(list)
    for row in ranked:
        by_sym[row["symbol"]].append(row["sim"])
    sym_best = sorted(((s, max(vs)) for s, vs in by_sym.items()), key=lambda x: -x[1])
    return {
        "query": qid,
        "top_atoms": ranked[:15],
        "symbol_ranking": [{"symbol": s, "best_sim": simv} for s, simv in sym_best],
        "n_candidates": n_train,
    }


def _sym_rank(ranking: dict, symbol: str | None) -> int | None:
    if not symbol:
        return None
    for i, row in enumerate(ranking["symbol_ranking"], start=1):
        if row["symbol"] == symbol:
            return i
    return None


def run_one(
    holdout: str,
    rows: list[dict],
    rows_by_sym: dict,
    *,
    epochs: int,
    seeds: list[int],
    most_kw: dict | None = None,
) -> dict:
    dump, held_row, train_syms = build_holdout_dump(holdout, rows)
    cousin = _expected_cousin(holdout, train_syms, rows_by_sym)
    group = int(held_row.get("group") or 0)
    group_mates = [
        s
        for s, r in rows_by_sym.items()
        if s in train_syms and int(r.get("group") or 0) == group
    ]
    print(f"expected cousin={cousin} group={group} mates={group_mates}", flush=True)

    seed_rows = []
    for seed in seeds:
        t0 = time.time()
        mode = (most_kw or {}).get("delta_mode", "square")
        print(
            f"\n=== holdout={holdout} seed={seed} Most {mode} {epochs}ep ===",
            flush=True,
        )
        model, metrics = fit_most(dump, seed, epochs, most_kw=most_kw)
        ranking = rank_holdout(model, dump, held_row)
        c_rank = _sym_rank(ranking, cousin)
        group_ranks = {s: _sym_rank(ranking, s) for s in group_mates}
        best_group = min((r for r in group_ranks.values() if r is not None), default=None)
        row = {
            "seed": seed,
            "val_r1": metrics["r@1"],
            "val_r3": metrics["r@3"],
            "cousin": cousin,
            "cousin_symbol_rank": c_rank,
            "cousin_in_top5_atoms": any(
                a["symbol"] == cousin for a in ranking["top_atoms"][:5]
            )
            if cousin
            else False,
            "best_group_mate_rank": best_group,
            "group_mate_ranks": group_ranks,
            "top_symbols": ranking["symbol_ranking"][:8],
            "top_atoms": ranking["top_atoms"][:8],
            "query": ranking["query"],
            "elapsed_s": time.time() - t0,
        }
        seed_rows.append(row)
        print(
            f"  val r@1={row['val_r1']:.3f} top_sym={[x['symbol'] for x in row['top_symbols'][:5]]} "
            f"cousin={cousin} rank={c_rank} best_group={best_group} ({row['elapsed_s']:.0f}s)",
            flush=True,
        )

    n = len(seed_rows)
    mean_cousin = sum(r["cousin_symbol_rank"] or 999 for r in seed_rows) / n
    n_ok = sum(
        1
        for r in seed_rows
        if r["cousin_symbol_rank"] is not None and r["cousin_symbol_rank"] <= 3
    )
    n_group3 = sum(
        1
        for r in seed_rows
        if r["best_group_mate_rank"] is not None and r["best_group_mate_rank"] <= 3
    )
    if n_ok == n:
        verdict = (
            f"PASS: kuzyn {cousin} w top-3 symboli u Mosta na FULL118 hold-out `{holdout}` "
            f"({n_ok}/{n} seedów) — wzorzec grupy przeżywa ukrycie pierwiastka."
        )
    elif n_group3 == n:
        verdict = (
            f"PARTIAL: dokładny kuzyn {cousin} nie zawsze top-3, ale jakiś mate grupy {group} "
            f"jest w top-3 ({n_group3}/{n}) — sygnał grupy częściowy."
        )
    else:
        verdict = (
            f"FAIL: kuzyn {cousin} NIE w top-3 symboli (ok {n_ok}/{n}); best group-mate "
            f"top-3: {n_group3}/{n}. FULL118 Most uczy lokalną topologię, nie generalizuje "
            f"„grupa → alkaliowiec” na ukryty `{holdout}`."
        )

    return {
        "holdout": holdout,
        "n_train": len(train_syms),
        "expected_cousin": cousin,
        "group": group,
        "group_mates_in_train": group_mates,
        "epochs": epochs,
        "seeds": seeds,
        "rows": seed_rows,
        "aggregate": {
            "mean_cousin_rank": mean_cousin,
            "n_cousin_top3": n_ok,
            "n_group_mate_top3": n_group3,
            "n_seeds": n,
            "mean_val_r1": sum(r["val_r1"] for r in seed_rows) / n,
        },
        "verdict": verdict,
        "dump": str(dump),
        "most_kw": most_kw or {"delta_mode": "square"},
    }


def _write_md(report: dict, path: Path) -> None:
    mk = report.get("most_kw") or {}
    lines = [
        "# Mendelejew FULL118 hold-out — MOST (Na / K)",
        "",
        f"Epoki={report['epochs']} · seeds={report['seeds']} · mode=nucleus+topology",
        f"most_kw=`{mk}`",
        "",
        "Protokół: train bez ukrytego pierwiastka; query `{sym}/nucleus` wstrzyknięty po treningu; "
        "kandydaci = 117 jąder train. Kuzyn = ta sama grupa, najbliższy okres (H na końcu remisu).",
        "",
        f"**Gate:** cousin_symbol_rank ≤ 3 na wszystkich seedach (albo PARTIAL: dowolny group-mate ≤ 3).",
        "",
    ]
    for h in report["holdouts"]:
        lines += [
            f"## Hold-out `{h['holdout']}`",
            "",
            f"Train N={h['n_train']} · cousin=**{h['expected_cousin']}** · group={h['group']}",
            f"Group mates w train: {h['group_mates_in_train']}",
            "",
            f"**Werdykt:** {h['verdict']}",
            "",
            "| seed | val r@1 | cousin rank | best group rank | top symbole |",
            "|-----:|--------:|------------:|----------------:|-------------|",
        ]
        for r in h["rows"]:
            tops = ", ".join(x["symbol"] for x in r["top_symbols"][:5])
            lines.append(
                f"| {r['seed']} | {r['val_r1']:.3f} | "
                f"{r['cousin_symbol_rank'] if r['cousin_symbol_rank'] is not None else '—'} | "
                f"{r['best_group_mate_rank'] if r['best_group_mate_rank'] is not None else '—'} | "
                f"{tops} |"
            )
        lines.append("")
        for r in h["rows"]:
            lines.append(f"### seed {r['seed']} — top atomy / group ranks")
            lines.append(f"group_mate_ranks: `{r['group_mate_ranks']}`")
            lines.append("| rank | atom | sym | sim |")
            lines.append("|-----:|------|-----|----:|")
            for t in r["top_atoms"]:
                lines.append(
                    f"| {t['rank']} | `{t['atom_id']}` | {t['symbol']} | {t['sim']:.3f} |"
                )
            lines.append("")
    lines += [
        "## Podsumowanie",
        "",
        report["summary_verdict"],
        "",
        f"JSON: `{report.get('json_name', 'mendelejew_holdout_full_NaK_results.json')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdouts", default="Na,K")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    ap.add_argument(
        "--delta-mode",
        default="square",
        choices=("square", "directed", "onesided"),
        help="wariant zamka ΔE (onesided = najlepszy na FULL118 recall)",
    )
    ap.add_argument("--delta-alpha", type=float, default=1.0)
    ap.add_argument("--directed-bias", type=float, default=0.25)
    ap.add_argument("--gamma-init", type=float, default=0.0)
    ap.add_argument("--e-scale", type=float, default=1.0)
    ap.add_argument("--couple-temp", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="suffix plików wyjściowych, np. onesided")
    args = ap.parse_args(argv)

    holdouts = [x.strip() for x in args.holdouts.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    most_kw = {
        "delta_mode": args.delta_mode,
        "delta_alpha": args.delta_alpha,
        "directed_bias": args.directed_bias,
        "gamma_init": args.gamma_init,
        "e_scale": args.e_scale,
        "couple_temp": args.couple_temp,
    }
    tag = args.tag.strip() or (
        args.delta_mode if args.delta_mode != "square" else ""
    )
    rows = build_rows()
    rows_by_sym = {str(r["symbol"]): r for r in rows}

    t_all = time.time()
    results = []
    for h in holdouts:
        results.append(
            run_one(
                h, rows, rows_by_sym, epochs=args.epochs, seeds=seeds, most_kw=most_kw
            )
        )

    n_pass = sum(1 for h in results if h["verdict"].startswith("PASS"))
    n_partial = sum(1 for h in results if h["verdict"].startswith("PARTIAL"))
    n_fail = sum(1 for h in results if h["verdict"].startswith("FAIL"))
    summary = (
        f"FULL118 hold-out Most ({args.delta_mode}): "
        f"PASS={n_pass} PARTIAL={n_partial} FAIL={n_fail} "
        f"na {holdouts} (seeds={seeds}, epochs={args.epochs})."
    )
    json_name = (
        f"mendelejew_holdout_full_NaK_{tag}_results.json"
        if tag
        else "mendelejew_holdout_full_NaK_results.json"
    )
    md_name = (
        f"MENDELEJEW_HOLDOUT_FULL_NaK_{tag}.md"
        if tag
        else "MENDELEJEW_HOLDOUT_FULL_NaK.md"
    )
    report = {
        "protocol": f"Mendelejew FULL118 hold-out Na/K — Most delta_mode={args.delta_mode}",
        "epochs": args.epochs,
        "seeds": seeds,
        "most_kw": most_kw,
        "holdouts": results,
        "summary_verdict": summary,
        "elapsed_s": time.time() - t_all,
        "json_name": json_name,
        "baseline_square_ref": "MENDELEJEW_HOLDOUT_FULL_NaK.md (square FAIL 2/2)",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / json_name
    mp = OUT_DIR / md_name
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(report, mp)
    print("\n" + summary)
    print(f"JSON → {jp}\nMD → {mp}")
    return report


if __name__ == "__main__":
    main()
