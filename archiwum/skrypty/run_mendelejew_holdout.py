"""Hold-out pierwiastka na tablicy (P0 follow-up).

Ukryj Na w treningu; w ewaluacji wstrzyknij atomy Na z cechami Φ
(grupa/okres/orbitale) i rankinguj podobieństwo do atomów train.
Sukces: Li (ta sama grupa 1) wysoko dla query Na/nucleus.

  python run_mendelejew_holdout.py
  python run_mendelejew_holdout.py --holdout Fe --epochs 32
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
from import_struktura_atomow import export_dump, load_rows, row_to_graph
from phimost_learn import build_model
from recall_data import RecallBatch
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"
SRC = ROOT / "struktura_atomow.csv"


def _symbol_of(atom_id: str) -> str:
    return atom_id.split("/", 1)[0]


def _expected_cousin(holdout: str, train_symbols: list[str]) -> str | None:
    """Ten sam group; wśród nich najbliższy period (Na→Li, nie H)."""
    rows = {str(r["symbol"]): r for r in load_rows(SRC)}
    if holdout not in rows:
        return None
    g = int(rows[holdout].get("group") or 0)
    p0 = int(rows[holdout].get("period") or 0)
    cousins = [
        s
        for s, r in rows.items()
        if s in train_symbols and s != holdout and int(r.get("group") or 0) == g
    ]
    if not cousins:
        return None
    cousins.sort(
        key=lambda s: (
            abs(int(rows[s].get("period") or 0) - p0),
            0 if s != "H" else 1,  # H specjalny — na końcu remisu
            s,
        )
    )
    return cousins[0]

def build_holdout_dump(holdout: str) -> tuple[Path, dict, list[str]]:
    rows = load_rows(SRC)
    train_rows = [r for r in rows if str(r["symbol"]) != holdout]
    held = next(r for r in rows if str(r["symbol"]) == holdout)
    out = ROOT / "dumps" / f"from_struktura_holdout_{holdout}"
    export_dump(train_rows, out)
    train_syms = [str(r["symbol"]) for r in train_rows]
    print(f"dump train → {out} symbols={train_syms} (held={holdout})")
    return out, held, train_syms


def _make_model(kind: str):
    if kind == "most":
        return build_model(warm=False)
    if kind == "softmax":
        return PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        )
    if kind == "flat":
        return build_model(warm=False)
    raise ValueError(kind)


def fit_model(dump: Path, seed: int, epochs: int, kind: str):
    n_atoms = sum(1 for _ in (dump / "atoms.jsonl").open(encoding="utf-8"))
    bs = min(16, max(4, n_atoms))
    _, train, val, _ = make_dump_loaders(
        dump,
        n_train=160,
        n_val=40,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
    )
    ablate = {"flat_payload": True} if kind == "flat" else {}
    torch.manual_seed(seed)
    model = _make_model("most" if kind == "flat" else kind)
    metrics = train_encoder(
        model, train, val, epochs=epochs, lr=CANON.lr, ablate_kw=ablate
    )
    return model, metrics, ablate


@torch.no_grad()
def rank_holdout(
    model,
    train_dump: Path,
    held_row: dict,
    *,
    ablate_kw: dict,
    query_suffix: str = "/nucleus",
) -> dict:
    """Train-graph + held element atoms in one forward; rank candidates."""
    _, g_train = load_dump(train_dump)
    g_held = row_to_graph(held_row)
    # concat atoms; keep only held-internal edges + train edges
    atoms = list(g_train.atoms) + list(g_held.atoms)
    edges = list(g_train.edges) + list(g_held.edges)
    from phi_schema import PhiGraph

    g = PhiGraph(atoms=atoms, edges=edges, meta={"holdout": held_row["symbol"]})
    content, tracer, depths, states = _graph_clean_tensors(
        g, CANON.d_x, CANON.n_dim, blend_holo_proto=0.0
    )
    # batch dim
    content = content.unsqueeze(0)
    tracer = tracer.unsqueeze(0)
    depths = depths.unsqueeze(0)
    states = states.unsqueeze(0)
    mask = torch.ones(1, len(atoms), dtype=torch.bool)
    topics = torch.zeros(1, len(atoms), dtype=torch.long)
    batch = RecallBatch(
        content=content,
        tracer=tracer,
        depths=depths,
        states=states,
        mask=mask,
        query_idx=torch.zeros(1, dtype=torch.long),
        pos_idx=torch.zeros(1, dtype=torch.long),
        topics=topics,
    ).ablate(**ablate_kw)

    model.eval()
    tok, _ = model(batch.content, batch.tracer, batch.mask)
    tok = tok[0]  # [N, d]

    n_train = len(g_train.atoms)
    hold_sym = str(held_row["symbol"])
    qid = f"{hold_sym}{query_suffix}"
    q_idx = next(i for i, a in enumerate(atoms) if a.atom_id == qid)

    q = tok[q_idx : q_idx + 1]  # [1,d]
    cand = tok[:n_train]
    sim = F.cosine_similarity(q, cand, dim=-1)  # [n_train]
    order = torch.argsort(sim, descending=True).tolist()

    ranked = []
    for rank, i in enumerate(order, start=1):
        aid = g_train.atoms[i].atom_id
        ranked.append(
            {
                "rank": rank,
                "atom_id": aid,
                "symbol": _symbol_of(aid),
                "sim": float(sim[i]),
            }
        )

    # aggregate by symbol: best sim / mean of top atom
    by_sym: dict[str, list[float]] = defaultdict(list)
    for row in ranked:
        by_sym[row["symbol"]].append(row["sim"])
    sym_best = sorted(
        ((s, max(vs)) for s, vs in by_sym.items()),
        key=lambda x: -x[1],
    )
    return {
        "query": qid,
        "top_atoms": ranked[:10],
        "symbol_ranking": [{"symbol": s, "best_sim": sim} for s, sim in sym_best],
        "n_candidates": n_train,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", default="Na")
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t0 = time.time()
    dump, held_row, train_syms = build_holdout_dump(args.holdout)
    cousin = _expected_cousin(args.holdout, train_syms)
    print(f"expected cousin (same group): {cousin}")

    results_arms = {}
    for kind in ("most", "softmax", "flat"):
        print(f"\n=== train {kind} ({args.epochs} ep) ===", flush=True)
        model, metrics, ablate = fit_model(dump, args.seed, args.epochs, kind)
        print(
            f"  val r@1={metrics['r@1']:.3f} r@3={metrics['r@3']:.3f}",
            flush=True,
        )
        ranking = rank_holdout(model, dump, held_row, ablate_kw=ablate)
        # where is cousin?
        cousin_rank = None
        if cousin:
            for i, row in enumerate(ranking["symbol_ranking"], start=1):
                if row["symbol"] == cousin:
                    cousin_rank = i
                    break
        nucleus_hit = any(
            a["symbol"] == cousin for a in ranking["top_atoms"][:5]
        ) if cousin else False
        results_arms[kind] = {
            "val_r1": metrics["r@1"],
            "val_r3": metrics["r@3"],
            "cousin": cousin,
            "cousin_symbol_rank": cousin_rank,
            "cousin_in_top5_atoms": nucleus_hit,
            "top_symbols": ranking["symbol_ranking"][:5],
            "top_atoms": ranking["top_atoms"][:5],
            "query": ranking["query"],
        }
        print(
            f"  query={ranking['query']} top_sym={[x['symbol'] for x in ranking['symbol_ranking'][:5]]} "
            f"cousin={cousin} rank={cousin_rank} in_top5_atoms={nucleus_hit}",
            flush=True,
        )

    # werdykt
    most_ok = (
        results_arms["most"]["cousin_symbol_rank"] is not None
        and results_arms["most"]["cousin_symbol_rank"] <= 2
    )
    flat_ok = (
        results_arms["flat"]["cousin_symbol_rank"] is not None
        and results_arms["flat"]["cousin_symbol_rank"] <= 2
    )
    if most_ok and not flat_ok:
        verdict = (
            f"Most (i ew. Softmax) plasuje kuzyna {cousin} wysoko dla ukrytego "
            f"{args.holdout}; flat nie — wzorzec grupy przeżywa hold-out."
        )
    elif most_ok and flat_ok:
        verdict = (
            f"Kuzyn {cousin} wysoko także na flat — ostrożnie: może to artefakt "
            f"pozycji/liczby atomów, nie chemii."
        )
    elif not most_ok:
        verdict = (
            f"Kuzyn {cousin} NIE w top-2 symboli u Mostu — hold-out nie odzyskał "
            f"miejsca grupy dla {args.holdout} (za mały set / za słaby sygnał)."
        )
    else:
        verdict = "Wynik mieszany."

    report = {
        "protocol": "Mendelejew hold-out — ukryj pierwiastek, ranking Na/nucleus→train",
        "holdout": args.holdout,
        "train_symbols": train_syms,
        "expected_cousin": cousin,
        "epochs": args.epochs,
        "seed": args.seed,
        "arms": results_arms,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / f"mendelejew_holdout_{args.holdout}_results.json"
    mp = OUT_DIR / f"MENDELEJEW_HOLDOUT_{args.holdout}.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(report, mp)
    print("\n" + verdict)
    print(f"JSON → {jp}\nMD → {mp}")
    return report


def _write_md(report: dict, path: Path) -> None:
    lines = [
        f"# Mendelejew hold-out — ukryty `{report['holdout']}`",
        "",
        f"Train bez `{report['holdout']}`: {report['train_symbols']}",
        f"Oczekiwany kuzyn (ta sama grupa): **{report['expected_cousin']}**",
        f"Epoki={report['epochs']} · seed={report['seed']}",
        "",
        "Query: `{holdout}/nucleus` wstrzyknięty z cechami Φ po treningu; "
        "kandydaci = tylko atomy z traina. Ranking cosinus tokenów enkodera.",
        "",
        "## Wyniki",
        "",
        "| arm | val r@1 | top symbole | rank kuzyna | kuzyn w top5 atomów |",
        "|-----|--------:|-------------|------------:|:-------------------:|",
    ]
    for kind, a in report["arms"].items():
        tops = ", ".join(
            f"{x['symbol']}({x['best_sim']:.2f})" for x in a["top_symbols"]
        )
        lines.append(
            f"| {kind} | {a['val_r1']:.3f} | {tops} | "
            f"{a['cousin_symbol_rank'] if a['cousin_symbol_rank'] is not None else '—'} | "
            f"{'Y' if a['cousin_in_top5_atoms'] else 'N'} |"
        )
    lines += ["", f"**Werdykt:** {report['verdict']}", ""]
    for kind, a in report["arms"].items():
        lines.append(f"### {kind} — top atomy")
        lines.append("| rank | atom | sym | sim |")
        lines.append("|-----:|------|-----|----:|")
        for t in a["top_atoms"]:
            lines.append(
                f"| {t['rank']} | `{t['atom_id']}` | {t['symbol']} | {t['sim']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
