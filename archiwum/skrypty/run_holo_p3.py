"""
Holo-Φ P3 — ablacja rodziny 女 bez tautologii

Warunki (factorial), blend_holo=0.6 (z P2):
  ++  phase+ binds+   (pełna konstrukcja)
  -+  phase− binds+   (faza niezależna, **te same** binds) ← kluczowy kontrast
  +-  phase+ binds−   (faza wspólna, bez krawędzi rodziny)
  --  phase− binds−   (podłoga)

Anty-tautologia: porównanie ++ vs -+ trzyma graf binds stały — mierzy wkład
wstrzykniętej fazy w Most (τ), nie „wyjęcie odpowiedzi z krawędzi”.
Softmax (bez τ) powinien być niemal niewrażliwy na phase± przy stałych binds.

Metryki:
  1) gate ΔSM (Most−SM r@1)
  2) family_hit@3 — przy query=glyph rodziny, czy inny glyph rodziny ∈ top-3

  python run_holo_p3.py
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import json
from pathlib import Path

import torch
import torch.nn.functional as F

# Determinizm run-to-run: manual_seed(seed) sam nie wystarcza — wielowątkowy
# CPU matmul/attention ma niedeterministyczną kolejność redukcji fp (patrz
# archiwum/holo/HOLO_WNIOSKIab.md §5: ten sam seed dawał różne r@1
# między przebiegami identycznego kodu).
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import load_raw_records, make_dump_loaders, make_recall_from_graph, records_to_graph
from make_dump_holo import write_dump
from phimost_learn import build_model
from phi_dump import load_edges_jsonl
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "archiwum" / "holo"
BLEND = 0.6
SEEDS = [0, 1, 23, 42]
EPOCHS = 16
N_TRAIN = 80
N_VAL = 25

CONDITIONS = [
    ("pp", True, True),    # phase+ binds+
    ("mp", False, True),   # phase− binds+  ← klucz
    ("pm", True, False),   # phase+ binds−
    ("mm", False, False),  # phase− binds−
]


@torch.no_grad()
def family_hit_at_k(
    model,
    dump: Path,
    family_ids: list[str],
    *,
    k: int = 3,
    zero_tracer: bool = False,
) -> float:
    """Średni hit@k: query family glyph → inny family glyph w top-k."""
    rows = load_raw_records(dump)
    edges = load_edges_jsonl(dump / "edges.jsonl")
    g = records_to_graph(rows, edges)
    id_to_i = {a.atom_id: i for i, a in enumerate(g.atoms)}
    fam_idx = [id_to_i[i] for i in family_ids if i in id_to_i]
    if len(fam_idx) < 2:
        return float("nan")

    topics = [str(r.get("topic", r.get("id", "x"))) for r in rows]
    batch = make_recall_from_graph(
        g,
        topics,
        batch_size=1,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        seed=0,
        pad_to=len(g.atoms),
        pair_mode="all",
        noise_scale=0.0,
        t_jitter=0.0,
        blend_holo_proto=BLEND,
    )
    model.eval()
    tracer = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
    tok, _ = model(batch.content, tracer, batch.mask)
    h = F.normalize(tok[0], dim=-1)  # [N,d]
    hits = []
    for qi in fam_idx:
        q = h[qi : qi + 1]
        scores = (q @ h.T).squeeze(0)
        scores[qi] = -1e9
        top = scores.argsort(descending=True)[:k].tolist()
        hit = any(t in fam_idx for t in top)
        hits.append(1.0 if hit else 0.0)
    return float(sum(hits) / len(hits))


def train_condition(dump: Path, seed: int, *, bs: int) -> dict:
    torch.manual_seed(seed)
    _g, train, val, _vc = make_dump_loaders(
        dump,
        n_train=N_TRAIN,
        n_val=N_VAL,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
        blend_holo_proto=BLEND,
    )
    torch.manual_seed(seed)
    most = build_model(warm=False)
    torch.manual_seed(seed)
    most_m = train_encoder(most, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    sm = PhiSoftmaxEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
    )
    sm_m = train_encoder(sm, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    most_z = PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
    )
    z_m = train_encoder(
        most_z, train, val, epochs=EPOCHS, lr=CANON.lr, ablate_kw={"zero_tracer": True}
    )

    fam_ids = json.loads((dump / "family_ids.json").read_text(encoding="utf-8"))
    fam_most = family_hit_at_k(most, dump, fam_ids, k=3, zero_tracer=False)
    fam_sm = family_hit_at_k(sm, dump, fam_ids, k=3, zero_tracer=False)

    d_sm = most_m["r@1"] - sm_m["r@1"]
    return {
        "seed": seed,
        "most_r1": most_m["r@1"],
        "sm_r1": sm_m["r@1"],
        "zero_r1": z_m["r@1"],
        "delta_sm": d_sm,
        "gate_pass": d_sm >= CANON.gate_recall_r1,
        "family_hit3_most": fam_most,
        "family_hit3_sm": fam_sm,
        "family_delta": fam_most - fam_sm,
    }


def run() -> dict:
    n_atoms = len(load_raw_records(ROOT / "dumps" / "from_holo")) if (ROOT / "dumps" / "from_holo").exists() else 73
    # bs from typical dump size
    rows = []
    for tag, phase, binds in CONDITIONS:
        dump = ROOT / "dumps" / f"from_holo_p3_{tag}"
        print(f"========== {tag} phase={phase} binds={binds} ==========")
        write_dump(dump, root_shared_phase=phase, root_binds=binds)
        n_atoms = len(load_raw_records(dump))
        # Duże grafy (N≳400): mniejszy batch — uwaga O(N²) na próbkę
        bs = 4 if n_atoms >= 400 else min(16, max(4, n_atoms))
        for seed in SEEDS:
            r = train_condition(dump, seed, bs=bs)
            r["condition"] = tag
            r["root_shared_phase"] = phase
            r["root_binds"] = binds
            tag_g = "PASS" if r["gate_pass"] else "FAIL"
            print(
                f"  seed={seed} ΔSM={r['delta_sm']:+.4f} {tag_g}  "
                f"fam@3 Most={r['family_hit3_most']:.2f} SM={r['family_hit3_sm']:.2f} "
                f"Δfam={r['family_delta']:+.2f}"
            )
            rows.append(r)

    by_cond = []
    for tag, phase, binds in CONDITIONS:
        sub = [r for r in rows if r["condition"] == tag]
        by_cond.append(
            {
                "condition": tag,
                "phase": phase,
                "binds": binds,
                "n_pass": sum(1 for r in sub if r["gate_pass"]),
                "mean_delta_sm": sum(r["delta_sm"] for r in sub) / len(sub),
                "mean_fam_most": sum(r["family_hit3_most"] for r in sub) / len(sub),
                "mean_fam_sm": sum(r["family_hit3_sm"] for r in sub) / len(sub),
                "mean_fam_delta": sum(r["family_delta"] for r in sub) / len(sub),
            }
        )

    pp = next(c for c in by_cond if c["condition"] == "pp")
    mp = next(c for c in by_cond if c["condition"] == "mp")
    # kluczowy kontrast: spadek family_hit Most przy wyłączeniu fazy (binds stałe)
    phase_effect_most = pp["mean_fam_most"] - mp["mean_fam_most"]
    phase_effect_sm = pp["mean_fam_sm"] - mp["mean_fam_sm"]

    verdict = (
        f"++→−+ (binds stałe): Δfam_Most={phase_effect_most:+.3f} "
        f"Δfam_SM={phase_effect_sm:+.3f}. "
    )
    if phase_effect_most > 0.05 and abs(phase_effect_sm) < 0.05:
        verdict += "Most używa wstrzykniętej fazy ponad graf; Softmax niewrażliwy — OK metodologicznie, ale klaster=konstrukcja."
    elif abs(phase_effect_most) < 0.05 and abs(phase_effect_sm) < 0.05:
        verdict += "Faza wspólna mały efekt przy stałych binds — klaster idzie z krawędzi/content, nie z φ_root."
    else:
        verdict += "Efekt mieszany — patrz tabela; nie interpretować jako czystego dowodu uczenia semantyki."

    report = {
        "protocol": "Holo-Φ P3 root ablation",
        "blend_holo": BLEND,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "design": "phase± × binds±; key contrast ++ vs -+ (binds fixed)",
        "rows": rows,
        "by_condition": by_cond,
        "phase_effect_most_pp_minus_mp": phase_effect_most,
        "phase_effect_sm_pp_minus_mp": phase_effect_sm,
        "verdict": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holo_p3_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    print(verdict)
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Holo-Φ P3 — ablacja 女 (bez tautologii)",
        "",
        f"blend_holo=**{report['blend_holo']}** · seedy {report['seeds']} · epochs={report['epochs']}",
        "",
        "## Design",
        "",
        "| Warunek | phase shared | root binds | Sens |",
        "|---------|:------------:|:----------:|------|",
        "| `pp` (++ ) | tak | tak | pełna konstrukcja |",
        "| `mp` (−+) | **nie** | tak | **klucz:** faza off, binds te same |",
        "| `pm` (+−) | tak | nie | faza bez krawędzi rodziny |",
        "| `mm` (−−) | nie | nie | podłoga |",
        "",
        "Anty-tautologia: ++ vs −+ nie wyjmuje odpowiedzi z grafu — tylko z τ/fazy.",
        "",
        "## Agregat",
        "",
        "| cond | phase | binds | PASS | mean ΔSM | fam@3 Most | fam@3 SM | Δfam |",
        "|------|:-----:|:-----:|-----:|---------:|-----------:|---------:|-----:|",
    ]
    for c in report["by_condition"]:
        lines.append(
            f"| `{c['condition']}` | {c['phase']} | {c['binds']} | "
            f"{c['n_pass']}/4 | {c['mean_delta_sm']:+.4f} | "
            f"{c['mean_fam_most']:.3f} | {c['mean_fam_sm']:.3f} | "
            f"{c['mean_fam_delta']:+.3f} |"
        )
    lines += [
        "",
        f"**Efekt fazy (pp−mp, binds stałe):** Most fam@3 {report['phase_effect_most_pp_minus_mp']:+.3f} · "
        f"SM {report['phase_effect_sm_pp_minus_mp']:+.3f}",
        "",
        f"**Werdykt:** {report['verdict']}",
        "",
        "## Pełne wiersze",
        "",
        "| cond | seed | ΔSM | gate | fam@3 Most | fam@3 SM |",
        "|------|-----:|----:|:----:|-----------:|---------:|",
    ]
    for r in report["rows"]:
        tag = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| `{r['condition']}` | {r['seed']} | {r['delta_sm']:+.4f} | {tag} | "
            f"{r['family_hit3_most']:.2f} | {r['family_hit3_sm']:.2f} |"
        )
    lines += ["", "JSON: `holo_p3_results.json`", ""]
    (OUT_DIR / "HOLO_P3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[P3] wrote {OUT_DIR / 'HOLO_P3.md'}")


if __name__ == "__main__":
    run()
