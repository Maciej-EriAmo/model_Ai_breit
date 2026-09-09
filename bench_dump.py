"""
Faza D — Softmax | Most | ZERO | R na offline dumpie Φ + case study.
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from phi_dump import load_dump, strip_all_payloads, warm_without_payload
from phi_schema import flat_text_content, PhiState
from dump_recall import (
    make_dump_loaders,
    bind_pairs_idx,
    graph_to_tensors,
    load_raw_records,
    make_recall_from_graph,
)
from checkpoint import save_encoder, DEFAULT_MOST
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder, recall_metrics
from scalar_r import eval_scalar_R, eval_cosine_only

ROOT = Path(__file__).resolve().parent
HISTORIA = ROOT / "historia"
DUMP = ROOT / "dumps" / "textfile_phi"


def fmt(m: dict) -> str:
    def f(key: str) -> str:
        v = m.get(key)
        if v is None or (isinstance(v, float) and v != v):  # NaN
            return "n/a"
        return f"{float(v):.3f}"

    note = m.get("note")
    extra = f" ({note})" if note else ""
    return f"r@1={f('r@1')} r@3={f('r@3')} r@5={f('r@5')} MRR={f('mrr')}{extra}"


def case_study(g) -> str:
    lines = [
        "# Case study runtime — flat vs Φ",
        "",
    ]
    main = next(a for a in g.atoms if a.atom_id == "__main__")
    text = main.payload or ""
    flat = flat_text_content(text, n_tokens=min(16, max(4, len(g.atoms))), d_x=8)
    rows = load_raw_records(DUMP)
    topics = [str(r.get("topic", r.get("id", "x"))) for r in rows]
    c, t, d, s = graph_to_tensors(g, topics, d_x=8, n_dim=6)

    lines += [
        f"- atomów w dumpie: {len(g.atoms)}",
        f"- bind pairs: {len(bind_pairs_idx(g))}",
        f"- flat chunks shape: {tuple(flat.shape)} (tylko bajty `__main__`)",
        f"- Φ content shape: {tuple(c.shape)} (wszystkie atomy S/E/payload)",
        f"- Φ tracer nonzero: {float(t.abs().sum()):.1f}",
        f"- stany: HOT={sum(1 for a in g.atoms if a.state==PhiState.HOT)} "
        f"WARM={sum(1 for a in g.atoms if a.state==PhiState.WARM)} "
        f"COLD={sum(1 for a in g.atoms if a.state==PhiState.COLD)}",
        "",
        "## Różnica ontologiczna",
        "",
        f"- flat ≠ Φ content: "
        f"{not torch.allclose(flat[: c.shape[0]].cpu() if flat.shape[0] == c.shape[0] else flat[0], c[0], atol=1e-4)}",
        "- WARM atomy bez payload nadal mają S/E/T w tensorach.",
        "",
        "## `__main__` peers (bind)",
        "",
    ]
    peers = sorted(
        {
            e.dst
            for e in g.edges
            if e.kind == "bind" and e.src == "__main__"
        }
    )
    for pid in peers:
        a = next(x for x in g.atoms if x.atom_id == pid)
        lines.append(
            f"- `{pid}` depth={a.depth} state={a.state.name} T={a.T:.2f} "
            f"payload={'yes' if a.payload else 'no'} E={a.E[:60]}"
        )
    lines.append("")
    return "\n".join(lines)


@torch.no_grad()
def eval_no_payload_most(model, g, topics, d_x=8, n_dim=6, batch_size=16) -> dict:
    """Most na dumpie bez payloadów — content z S/E only (strip HOT bytes)."""
    g2 = strip_all_payloads(g)
    # ten sam seed/batch co porównanie: strip musi zmienić tensory vs pełny graf
    batch = make_recall_from_graph(g2, topics, batch_size, d_x, n_dim, seed=42)
    full = make_recall_from_graph(g, topics, batch_size, d_x, n_dim, seed=42)
    if torch.allclose(batch.content, full.content, atol=1e-6):
        # brak HOT payload w grafie — metryka nieinformacyjna
        return {
            "r@1": float("nan"),
            "r@3": float("nan"),
            "r@5": float("nan"),
            "mrr": float("nan"),
            "note": "no HOT payload to strip",
        }
    tok, _ = model(batch.content, batch.tracer, batch.mask)
    return recall_metrics(tok, batch.query_idx, batch.pos_idx, mask=batch.mask)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    import make_dump_textfile

    make_dump_textfile.main()  # zawsze odśwież dump (docs/case na bieżąco)

    # docs na bieżąco — case
    man, g0 = load_dump(DUMP)
    g = warm_without_payload(g0)
    case_md = case_study(g)
    (DUMP / "CASE_RUNTIME.md").write_text(case_md + "\n", encoding="utf-8")
    print(case_md)

    d_x, n_dim = 8, 6
    g, train, val, val_clean = make_dump_loaders(
        DUMP, n_train=80, n_val=25, batch_size=16, d_x=d_x, n_dim=n_dim
    )

    print("=" * 72)
    print(f"FAZA D — dump `{DUMP.name}` atoms={len(g.atoms)} binds={len(bind_pairs_idx(g))}")
    print("=" * 72)

    results = {}

    torch.manual_seed(42)
    sm = PhiSoftmaxEncoder(input_dim=d_x, dim_tracer=n_dim, num_layers=2, d=32)
    results["Softmax"] = train_encoder(sm, train, val, epochs=14)
    print(f"Softmax     {fmt(results['Softmax'])}")

    torch.manual_seed(42)
    most = PhiBridgeEncoder(
        input_dim=d_x, dim_tracer=n_dim, num_layers=2, d=32, coupling="softplus"
    )
    results["Most"] = train_encoder(most, train, val, epochs=14)
    print(
        f"Most        {fmt(results['Most'])}  |dE|={results['Most']['mean_dE']:.4f}"
    )
    ckpt_path = save_encoder(
        most,
        DEFAULT_MOST,
        kind="most",
        meta={
            "task": "dump_recall",
            "dump": str(DUMP.name),
            "r@1": results["Most"]["r@1"],
            "epochs": 14,
        },
    )
    print(f"Checkpoint  {ckpt_path}")

    torch.manual_seed(42)
    most_z = PhiBridgeEncoder(
        input_dim=d_x, dim_tracer=n_dim, num_layers=2, d=32, coupling="softplus"
    )
    results["Most ZERO"] = train_encoder(
        most_z, train, val, epochs=14, ablate_kw={"zero_tracer": True}
    )
    print(f"Most ZERO   {fmt(results['Most ZERO'])}")

    torch.manual_seed(42)
    most_flat = PhiBridgeEncoder(
        input_dim=d_x, dim_tracer=n_dim, num_layers=2, d=32, coupling="softplus"
    )
    results["flat payload"] = train_encoder(
        most_flat, train, val, epochs=14, ablate_kw={"flat_payload": True}
    )
    print(f"flat        {fmt(results['flat payload'])}")

    rows = load_raw_records(DUMP)
    topics = [str(r.get("topic", r.get("id", "x"))) for r in rows]
    results["Most no-payload"] = eval_no_payload_most(most, g, topics, d_x, n_dim)
    print(f"Most S/E/T  {fmt(results['Most no-payload'])}  (strip all payloads)")

    results["R (depth)"] = eval_scalar_R(val, g=0.3)
    results["cosine"] = eval_cosine_only(val)
    print(f"R depth     {fmt(results['R (depth)'])}")
    print(f"cosine      {fmt(results['cosine'])}")

    # clean-ish batch eval
    most.eval()
    with torch.no_grad():
        tok, _ = most(val_clean.content, val_clean.tracer, val_clean.mask)
        results["Most clean-batch"] = recall_metrics(
            tok, val_clean.query_idx, val_clean.pos_idx, mask=val_clean.mask
        )
    print(f"Most clean  {fmt(results['Most clean-batch'])}")

    d_sm = results["Most"]["r@1"] - results["Softmax"]["r@1"]
    d_r = results["Most"]["r@1"] - results["R (depth)"]["r@1"]
    beat_sm = d_sm >= 0.02
    ok_r = d_r >= -0.02
    ok = beat_sm and ok_r

    print("=" * 72)
    print(f"Δ Most−SM r@1={d_sm:+.4f} ({'PASS' if beat_sm else 'FAIL'})")
    print(f"Δ Most−R  r@1={d_r:+.4f} ({'PASS' if ok_r else 'FAIL'})")
    print(f"WERDYKT FAZY D: {'PASS' if ok else 'FAIL'}")

    # WYNIKI
    order = [
        "Softmax",
        "Most",
        "Most ZERO",
        "flat payload",
        "Most no-payload",
        "Most clean-batch",
        "R (depth)",
        "cosine",
    ]
    lines = [
        "# Wyniki Faza D — offline dump Φ",
        "",
        f"**Dump:** `dumps/textfile_phi` ({man.get('description', '')})",
        f"Atoms={len(g.atoms)}, bind pairs={len(bind_pairs_idx(g))}.",
        "Trening na zaszumionych kopiach topologii dumpa; ewal na holdout.",
        "",
        "| wariant | r@1 | r@3 | r@5 | MRR |",
        "|---------|-----|-----|-----|-----|",
    ]
    for k in order:
        m = results[k]
        def cell(key: str) -> str:
            v = m.get(key)
            if v is None or (isinstance(v, float) and v != v):
                return "n/a"
            return f"{float(v):.3f}"

        lines.append(
            f"| {k} | {cell('r@1')} | {cell('r@3')} | {cell('r@5')} | {cell('mrr')} |"
        )
    lines += [
        "",
        f"**Δ Most−Softmax r@1** = {d_sm:+.4f} → {'PASS' if beat_sm else 'FAIL'}",
        f"**Δ Most−R r@1** = {d_r:+.4f} → {'PASS' if ok_r else 'FAIL'}",
        f"**Werdykt:** **{'PASS' if ok else 'FAIL'}**",
        "",
        "## Sanity",
        "- HOT/WARM: WARM bez payload w dumpie; `Most no-payload` = strip wszystkich bajtów.",
        "- ZERO / flat: kontrole obserwatora i Unix fiction.",
        "",
        "## Case study",
        "- `dumps/textfile_phi/CASE_TEXT.md`",
        "- `dumps/textfile_phi/CASE_RUNTIME.md`",
        "",
        "## Go / no-go Faza E",
        (
            "GO: live read-only Store opcjonalnie; generative head później."
            if ok
            else "NO-GO live: poprawić adapter/dump zanim Store."
        ),
    ]
    HISTORIA.mkdir(exist_ok=True)
    out = HISTORIA / "WYNIKI_FAZA_D.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zapisano: {out}")

    fig, ax = plt.subplots(figsize=(9, 4))
    names = order
    vals = [results[k]["r@1"] for k in names]
    ax.bar(range(len(names)), vals, color="#c44")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("recall@1")
    ax.set_title("PhiMost Faza D — dump textfile_phi")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = HISTORIA / "phi_most_faza_d.png"
    plt.savefig(out_png, dpi=120)
    print(f"Zapisano: {out_png}")

    # update FAZA_D doc
    doc = ROOT / "docs" / "FAZA_D.md"
    doc.write_text(
        "\n".join(
            [
                "# Faza D — Offline dump Karmazyn → PhiMost",
                "",
                f"**Status:** {'PASS' if ok else 'FAIL'} (bench)",
                f"**Dump:** `dumps/textfile_phi`",
                f"**Werdykt:** Most−SM r@1={d_sm:+.3f}, Most−R r@1={d_r:+.3f}",
                "",
                "## Artefakty",
                "",
                "- `make_dump_textfile.py` — budowa dumpa",
                "- `phi_dump.py` — loader PhiJSON",
                "- `dump_recall.py` — recall z bind",
                "- `bench_dump.py` — Softmax|Most|ZERO|R",
                "- `historia/WYNIKI_FAZA_D.md`, `historia/phi_most_faza_d.png`",
                "- case: `dumps/textfile_phi/CASE_TEXT.md`, `CASE_RUNTIME.md`",
                "",
                "## Acceptance",
                "",
                f"- [x] dump → PhiGraph",
                f"- [{'x' if ok else ' '}] Most bije Softmax na dumpie (≥+2 pkt r@1)",
                f"- [x] HOT/WARM / no-payload sanity",
                f"- [x] case study nie-płaski tekst",
                f"- [x] docs na bieżąco",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
