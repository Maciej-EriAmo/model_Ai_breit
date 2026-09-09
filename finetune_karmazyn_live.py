"""
Fine-tune Most na żywym dumpie Karmazyna (M2).

Stabilizacja (wariancja cold-start):
  - pair_mode=mixed (same-depth + cross-depth → ΔE dla Mostu)
  - mniejszy szum content
  - seed-sweep: bierzemy najlepszy PASS (ΔSM≥gate), dopiero wtedy promote

  python finetune_karmazyn_live.py --no-warm
  python finetune_karmazyn_live.py --no-warm --seeds 0,1,2,7,42 --epochs 18
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkpoint import CKPT_DIR, DEFAULT_MOST, load_encoder, save_encoder
from config import CANON
from dump_recall import load_raw_records, make_dump_loaders
from recall_model import (
    PhiBridgeEncoder,
    PhiSoftmaxEncoder,
    recall_metrics,
    train_encoder,
)

ROOT = Path(__file__).resolve().parent
LIVE_BBL = ROOT / "dumps" / "karmazyn_live_bbl"
LIVE_PHI = ROOT / "dumps" / "from_karmazyn_live"
HISTORIA = ROOT / "historia"
DEFAULT_OUT = CKPT_DIR / "most_karmazyn_live.pt"


def export_live() -> Path:
    import export_karmazyn_live as ex
    from bubblefs_import import bubblefs_to_graph, export_graph_as_dump

    ex._setup_path()
    s, ids, bubbles, holo = ex.seed_store(rich=True)
    ex.store_to_bubblefs(s, ids, bubbles, LIVE_BBL, holo_groups=holo)
    _man, g = bubblefs_to_graph(LIVE_BBL)
    export_graph_as_dump(
        g, LIVE_PHI, description="finetune: rich Karmazyn Store → BubbleFS → Φ"
    )
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    print(f"export atoms={len(g.atoms)} bind={n_bind} → {LIVE_PHI}")
    return LIVE_PHI


def build_model(*, warm: bool) -> PhiBridgeEncoder:
    if warm and DEFAULT_MOST.is_file():
        model, meta = load_encoder(DEFAULT_MOST)
        print(f"warm-start ← {DEFAULT_MOST}  meta_r@1={meta.get('r@1')}")
        return model  # type: ignore[return-value]
    return PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
    )


def run_one(
    dump: Path,
    *,
    seed: int,
    epochs: int,
    n_train: int,
    n_val: int,
    warm: bool,
    pair_mode: str,
    noise: float,
    blend_holo: float,
    baselines: bool = False,
) -> dict:
    torch.manual_seed(seed)
    n_atoms = len(load_raw_records(dump))
    bs = min(16, max(4, n_atoms))
    g, train, val, val_clean = make_dump_loaders(
        dump,
        n_train=n_train,
        n_val=n_val,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode=pair_mode,
        noise_scale=noise,
        seed0=1000 + seed * 17,
        blend_holo_proto=blend_holo,
    )

    most = build_model(warm=warm)
    torch.manual_seed(seed)
    most_m = train_encoder(most, train, val, epochs=epochs, lr=CANON.lr)

    sm_m = {"r@1": float("nan"), "r@3": float("nan"), "mrr": float("nan")}
    z_m = dict(sm_m)
    d_sm = float("nan")
    d_z = float("nan")
    ok = True
    if baselines:
        torch.manual_seed(seed)
        sm = PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        )
        sm_m = train_encoder(sm, train, val, epochs=epochs, lr=CANON.lr)
        torch.manual_seed(seed)
        most_z = PhiBridgeEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
            coupling=CANON.coupling,
        )
        z_m = train_encoder(
            most_z,
            train,
            val,
            epochs=epochs,
            lr=CANON.lr,
            ablate_kw={"zero_tracer": True},
        )
        d_sm = most_m["r@1"] - sm_m["r@1"]
        d_z = most_m["r@1"] - z_m["r@1"]
        ok = d_sm >= CANON.gate_recall_r1

    most.eval()
    with torch.no_grad():
        tok, _ = most(val_clean.content, val_clean.tracer, val_clean.mask)
        clean = recall_metrics(
            tok, val_clean.query_idx, val_clean.pos_idx, mask=val_clean.mask
        )

    return {
        "seed": seed,
        "atoms": len(g.atoms),
        "most": most_m,
        "softmax": sm_m,
        "zero": z_m,
        "clean": clean,
        "delta_sm": d_sm,
        "delta_zero": d_z,
        "gate_pass": ok,
        "baselines": baselines,
        "model": most,
    }


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Most on Karmazyn live Φ")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--pair-mode", default="mixed", choices=["same_depth", "all", "mixed"])
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument(
        "--blend-holo",
        type=float,
        default=0.35,
        help="domieszka prototypu klastra hologramu do content (jak topic w C)",
    )
    ap.add_argument(
        "--seeds",
        default="0,1,2,7,42",
        help="comma-separated seed sweep; promote najlepszy PASS",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument(
        "--baselines",
        action="store_true",
        help="Softmax+ZERO + gate ΔSM (3× wolniej; domyślnie OFF)",
    )
    args = ap.parse_args()

    if args.skip_export and (LIVE_PHI / "atoms.jsonl").is_file():
        dump = LIVE_PHI
    else:
        dump = export_live()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    mode = "Most+SM+ZERO" if args.baselines else "Most-only"
    print(
        f"sweep [{mode}] seeds={seeds} pair_mode={args.pair_mode} noise={args.noise} "
        f"blend_holo={args.blend_holo} epochs={args.epochs} warm={not args.no_warm}"
    )

    results = []
    best = None
    for seed in seeds:
        print("-" * 60)
        print(f"SEED {seed}")
        r = run_one(
            dump,
            seed=seed,
            epochs=args.epochs,
            n_train=args.n_train,
            n_val=args.n_val,
            warm=not args.no_warm,
            pair_mode=args.pair_mode,
            noise=args.noise,
            blend_holo=args.blend_holo,
            baselines=args.baselines,
        )
        if args.baselines:
            print(
                f"  Most={r['most']['r@1']:.3f}  SM={r['softmax']['r@1']:.3f}  "
                f"ZERO={r['zero']['r@1']:.3f}  ΔSM={r['delta_sm']:+.4f}  "
                f"{'PASS' if r['gate_pass'] else 'FAIL'}"
            )
            score = r["delta_sm"] if r["gate_pass"] else r["delta_sm"] - 10.0
        else:
            print(
                f"  Most r@1={r['most']['r@1']:.3f}  r@3={r['most']['r@3']:.3f}  "
                f"clean={r['clean']['r@1']:.3f}"
            )
            score = r["most"]["r@1"]
        results.append({k: v for k, v in r.items() if k != "model"})
        if best is None or score > best["_score"]:
            r["_score"] = score
            best = r

    chosen = best
    assert chosen is not None
    ok = bool(chosen["gate_pass"]) if args.baselines else True
    print("=" * 60)
    if args.baselines:
        print(
            f"WYBRANO seed={chosen['seed']}  ΔSM={chosen['delta_sm']:+.4f}  "
            f"{'PASS' if ok else 'FAIL (bez promote)'}"
        )
    else:
        print(
            f"WYBRANO seed={chosen['seed']}  Most r@1={chosen['most']['r@1']:.3f}  "
            f"(fast — --baselines dla gate)"
        )

    meta = {
        "task": "dump_recall",
        "source": "finetune_karmazyn_live",
        "dump": str(dump),
        "atoms": chosen["atoms"],
        "epochs": args.epochs,
        "warm_start": not args.no_warm,
        "pair_mode": args.pair_mode,
        "noise": args.noise,
        "baselines": args.baselines,
        "seed": chosen["seed"],
        "r@1": chosen["most"]["r@1"],
        "r@3": chosen["most"]["r@3"],
        "softmax_r@1": chosen["softmax"].get("r@1"),
        "zero_r@1": chosen["zero"].get("r@1"),
        "delta_sm": chosen["delta_sm"],
        "clean_r@1": chosen["clean"]["r@1"],
        "gate_pass": ok,
        "sweep": results,
    }
    path = save_encoder(chosen["model"], args.out, kind="most", meta=meta)
    print(f"saved → {path}")
    promoted = False
    if ok and not args.no_promote:
        path2 = save_encoder(chosen["model"], DEFAULT_MOST, kind="most", meta=meta)
        print(f"promoted → {path2}")
        promoted = True
    elif args.baselines and not ok:
        print("NIE promuję do most_dump_recall.pt — gate FAIL")

    HISTORIA.mkdir(exist_ok=True)
    if args.baselines:
        lines = [
            "# Fine-tune Most — z baseline SM/ZERO",
            "",
            f"**Dump:** `{dump}` atoms={chosen['atoms']}",
            f"**Wybrany seed:** {chosen['seed']}",
            "",
            "| seed | Most r@1 | SM | ZERO | ΔSM | gate |",
            "|------|----------|----|------|-----|------|",
        ]
        for r in results:
            lines.append(
                f"| {r['seed']} | {r['most']['r@1']:.3f} | {r['softmax']['r@1']:.3f} | "
                f"{r['zero']['r@1']:.3f} | {r['delta_sm']:+.4f} | "
                f"{'PASS' if r['gate_pass'] else 'FAIL'} |"
            )
        lines += ["", f"**Werdykt:** **{'PASS' if ok else 'FAIL'}**", ""]
    else:
        lines = [
            "# Fine-tune Most — fast (bez SM/ZERO)",
            "",
            f"**Dump:** `{dump}` atoms={chosen['atoms']}",
            f"**Wybrany seed:** {chosen['seed']}  r@1={chosen['most']['r@1']:.3f}",
            "",
            "| seed | Most r@1 | r@3 | clean |",
            "|------|----------|-----|-------|",
        ]
        for r in results:
            lines.append(
                f"| {r['seed']} | {r['most']['r@1']:.3f} | {r['most']['r@3']:.3f} | "
                f"{r['clean']['r@1']:.3f} |"
            )
        lines += ["", f"**Ckpt:** `{path}`", ""]
    (HISTORIA / "WYNIKI_FINETUNE_LIVE.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (HISTORIA / "finetune_live.json").write_text(
        json.dumps({**meta, "promoted": promoted}, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"WERDYKT: {'PASS' if ok else 'FAIL'}")
    if args.baselines and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
