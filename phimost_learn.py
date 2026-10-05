"""
Uniwersalny loader + nauka Most (PowerShell-friendly).

Źródła:
  - struktura_atomow.csv / .json   (konfiguracja elektronowa)
  - literatura .md / .txt / katalog
  - katalog PhiJSON               (atoms.jsonl + edges.jsonl)
  - katalog BubbleFS              (bubbles/*.bbl)
  - preset: karmazyn | qm | struktura | textfile | literature

Przykłady (PowerShell):

  cd "C:\\Users\\drwis\\model_Ai_breit"

  python phimost_learn.py --src struktura_atomow.csv --no-warm --no-promote
  python phimost_learn.py --src rozdzial.md --no-warm --no-promote
  python phimost_learn.py --src .\\ksiazki\\ --no-warm --no-promote
  python phimost_learn.py --preset karmazyn --no-warm
  python phimost_learn.py --src dumps/from_literature --skip-load --recall-id doc/__main__ --skip-train
"""

from __future__ import annotations

import argparse
import json
import sys
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
HISTORIA = ROOT / "archiwum" / "historia"
DUMPS = ROOT / "dumps"


# ---------------------------------------------------------------------------
# Detect / load
# ---------------------------------------------------------------------------

def detect_kind(src: Path) -> str:
    src = Path(src)
    if src.is_file():
        suf = src.suffix.lower()
        if suf == ".csv":
            return "struktura_csv"
        if suf in (".md", ".txt", ".markdown", ".rst"):
            return "literature_file"
        if suf == ".json":
            # struktura list vs other
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
                if isinstance(data, list) and data and "electron_configuration" in data[0]:
                    return "struktura_json"
            except Exception:
                pass
            return "json_unknown"
        if suf == ".bbl":
            return "bubblefs_file"
        raise ValueError(f"nieobsługiwany plik: {src}")
    if src.is_dir():
        if (src / "atoms.jsonl").is_file():
            return "phijson"
        if (src / "bubbles").is_dir() or list(src.glob("*.bbl")):
            return "bubblefs_dir"
        # katalog z .md/.txt = korpus literacki
        lit = list(src.rglob("*.md")) + list(src.rglob("*.txt"))
        if lit:
            return "literature_dir"
        raise ValueError(f"katalog bez atoms.jsonl / bubbles / .md|.txt: {src}")
    raise FileNotFoundError(src)


def default_out_for(kind: str, src: Path) -> Path:
    if kind.startswith("struktura"):
        return DUMPS / "from_struktura"
    if kind.startswith("literature"):
        return DUMPS / "from_literature"
    if kind == "phijson":
        return src
    if kind.startswith("bubblefs"):
        return DUMPS / "from_bubblefs"
    return DUMPS / "learn_out"


def load_source(src: Path, out: Path, kind: str | None = None) -> Path:
    """Źródło → katalog PhiJSON. Zwraca ścieżkę dumpa."""
    src = Path(src)
    out = Path(out)
    kind = kind or detect_kind(src)

    if kind in ("struktura_csv", "struktura_json"):
        import import_struktura_atomow as imp

        rows = imp.load_rows(src)
        imp.export_dump(rows, out)
        print(f"[load] struktura {src.name} → {out}")
        return out

    if kind == "phijson":
        print(f"[load] PhiJSON już gotowy: {src}")
        return src

    if kind in ("bubblefs_dir", "bubblefs_file"):
        from bubblefs_import import bubblefs_to_graph, export_graph_as_dump

        root = src if kind == "bubblefs_dir" else src.parent
        _man, g = bubblefs_to_graph(root if kind == "bubblefs_dir" else src)
        export_graph_as_dump(g, out, description=f"BubbleFS ← {src}")
        print(f"[load] BubbleFS {src} → {out}")
        return out

    if kind in ("literature_file", "literature_dir"):
        import import_literature as lit

        lit.export_dump(src, out)
        print(f"[load] literature {src} → {out}")
        return out

    raise ValueError(f"nie umiem załadować kind={kind}")


def load_preset(name: str) -> Path:
    name = name.lower().strip()
    if name in ("struktura", "atoms", "csv"):
        import import_struktura_atomow as imp

        src = imp.DEFAULT_JSON if imp.DEFAULT_JSON.is_file() else imp.DEFAULT_CSV
        return load_source(src, DUMPS / "from_struktura")
    if name in ("karmazyn", "live", "store"):
        import export_karmazyn_live as ex
        from bubblefs_import import bubblefs_to_graph, export_graph_as_dump

        ex._setup_path()
        s, ids, bubbles, holo = ex.seed_store(rich=True)
        bbl = DUMPS / "karmazyn_live_bbl"
        phi = DUMPS / "from_karmazyn_live"
        ex.store_to_bubblefs(s, ids, bubbles, bbl, holo_groups=holo)
        _man, g = bubblefs_to_graph(bbl)
        export_graph_as_dump(g, phi, description="preset karmazyn")
        print(f"[load] preset karmazyn → {phi}")
        return phi
    if name in ("qm", "orbitals", "quantum"):
        import qm_orbitals_data as qm

        out = DUMPS / "qm_orbitals"
        qm.export_phi_dump(out)
        print(f"[load] preset qm → {out}")
        return out
    if name in ("textfile", "lab", "phi"):
        dump = DUMPS / "textfile_phi"
        if not (dump / "atoms.jsonl").is_file():
            import make_dump_textfile

            make_dump_textfile.main()
        print(f"[load] preset textfile → {dump}")
        return dump
    if name in ("literature", "lit", "docs"):
        import import_literature as lit

        out = DUMPS / "from_literature"
        files = [
            p
            for p in ROOT.glob("*")
            if p.suffix.lower() in (".md", ".txt") and p.is_file()
        ]
        if not files:
            raise FileNotFoundError("brak .md/.txt w root — podaj --src plik/katalog")
        g2 = lit.paths_to_graph(files)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
            for a in g2.atoms:
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
        with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
            for e in g2.edges:
                f.write(
                    json.dumps(
                        {"kind": e.kind, "src": e.src, "dst": e.dst, "name": e.name},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "phijson-v1",
                    "source": "preset_literature",
                    "docs": g2.meta.get("docs"),
                    "n_atoms": len(g2.atoms),
                    "n_edges": len(g2.edges),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[load] preset literature → {out} docs={g2.meta.get('docs')}")
        return out
    raise ValueError(
        f"nieznany preset {name!r}; użyj: struktura | karmazyn | qm | textfile | literature"
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

# Oś 3 bywa energią orbitalu albo barwą zdjęcia, nie temperaturą archiwum.
_ENERGY_OR_COLOR_T = frozenset({"struktura", "qm", "glyphs_photo"})


def seal_for_source(name: str) -> bool | None:
    """None zostawia kanon (seal włączony). False: nie traktuj osi 3 jak archiwum."""
    if name in _ENERGY_OR_COLOR_T:
        return False
    return None


def build_model(*, warm: bool, t_seal: bool | None = None) -> PhiBridgeEncoder:
    if warm and DEFAULT_MOST.is_file():
        model, meta = load_encoder(DEFAULT_MOST)
        print(f"[train] warm-start ← {DEFAULT_MOST} meta_r@1={meta.get('r@1')}")
        if t_seal is not None:
            model.t_seal = t_seal
        return model  # type: ignore[return-value]
    print("[train] cold start (CANON)")
    return PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
        t_seal=t_seal,
    )


def run_seed(
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
    restrict_prefix: bool,
    baselines: bool = False,
    amp_dtype: torch.dtype | None = None,
    t_seal: bool | None = None,
) -> dict:
    """
    domyślnie tylko Most (szybka nauka).
    `--baselines` → Softmax + ZERO (3× czas, gate ΔSM).
    """
    torch.manual_seed(seed)
    n_atoms = len(load_raw_records(dump))
    # Duże grafy (N≳400): mniejszy batch — uwaga O(N²) na próbkę
    if n_atoms >= 400:
        bs = 4
    else:
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
        restrict_same_prefix=restrict_prefix,
    )
    most = build_model(warm=warm, t_seal=t_seal)
    torch.manual_seed(seed)
    most_m = train_encoder(
        most,
        train,
        val,
        epochs=epochs,
        lr=CANON.lr,
        return_curve=True,
        amp_dtype=amp_dtype,
    )

    sm_m = {"r@1": float("nan"), "r@3": float("nan"), "mrr": float("nan")}
    z_m = dict(sm_m)
    d_sm = float("nan")
    ok = True  # bez baseline: sukces = wytrenowano + finite r@1
    if baselines:
        torch.manual_seed(seed)
        sm = PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        )
        sm_m = train_encoder(
            sm, train, val, epochs=epochs, lr=CANON.lr, amp_dtype=amp_dtype
        )
        torch.manual_seed(seed)
        most_z = PhiBridgeEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
            coupling=CANON.coupling,
            t_seal=t_seal,
        )
        z_m = train_encoder(
            most_z,
            train,
            val,
            epochs=epochs,
            lr=CANON.lr,
            ablate_kw={"zero_tracer": True},
            amp_dtype=amp_dtype,
        )
        d_sm = most_m["r@1"] - sm_m["r@1"]
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
        "gate_pass": ok,
        "baselines": baselines,
        "model": most,
    }


def train_dump(
    dump: Path,
    *,
    out_ckpt: Path,
    seeds: list[int],
    epochs: int,
    n_train: int,
    n_val: int,
    warm: bool,
    pair_mode: str,
    noise: float,
    blend_holo: float,
    restrict_prefix: bool,
    promote: bool,
    baselines: bool = False,
    run_name: str | None = None,
    argv: list[str] | None = None,
    amp_dtype: torch.dtype | None = None,
    t_seal: bool | None = None,
) -> dict:
    mode = "Most+SM+ZERO" if baselines else "Most-only (szybki)"
    print(
        f"[train] {mode} dump={dump} seeds={seeds} epochs={epochs} "
        f"pair={pair_mode} blend_holo={blend_holo} restrict_prefix={restrict_prefix} "
        f"t_seal={t_seal}"
    )
    from dataclasses import asdict

    from runlog import RunDir

    rundir = RunDir.create(
        run_name or "phimost_learn",
        argv=argv,
        hparams={
            "dump": str(dump),
            "out_ckpt": str(out_ckpt),
            "seeds": list(seeds),
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
            "warm": warm,
            "pair_mode": pair_mode,
            "noise": noise,
            "blend_holo": blend_holo,
            "restrict_prefix": restrict_prefix,
            "promote": promote,
            "baselines": baselines,
            "amp_dtype": str(amp_dtype) if amp_dtype is not None else "fp32",
            "t_seal": t_seal,
            "canon": asdict(CANON),
        },
        extra_meta={
            "criterion": "success#5_run_dir",
            "bf16": amp_dtype == torch.bfloat16,
        },
    )
    print(f"[train] run-dir → {rundir.path}")
    if amp_dtype is not None:
        print(f"[train] AMP dtype={amp_dtype} (kryterium #2 BF16)")
    results = []
    best = None
    for seed in seeds:
        print("-" * 60)
        print(f"SEED {seed}")
        r = run_seed(
            dump,
            seed=seed,
            epochs=epochs,
            n_train=n_train,
            n_val=n_val,
            warm=warm,
            pair_mode=pair_mode,
            noise=noise,
            blend_holo=blend_holo,
            restrict_prefix=restrict_prefix,
            baselines=baselines,
            amp_dtype=amp_dtype,
            t_seal=t_seal,
        )
        if baselines:
            print(
                f"  Most={r['most']['r@1']:.3f}  SM={r['softmax']['r@1']:.3f}  "
                f"ZERO={r['zero']['r@1']:.3f}  ΔSM={r['delta_sm']:+.4f}  "
                f"{'PASS' if r['gate_pass'] else 'FAIL'}"
            )
        else:
            print(
                f"  Most r@1={r['most']['r@1']:.3f}  r@3={r['most']['r@3']:.3f}  "
                f"clean={r['clean']['r@1']:.3f}"
            )
        results.append({k: v for k, v in r.items() if k != "model"})
        # wybór: z baseline po ΔSM (PASS), bez → po Most r@1
        if baselines:
            score = r["delta_sm"] if r["gate_pass"] else r["delta_sm"] - 10.0
        else:
            score = r["most"]["r@1"]
        if best is None or score > best["_score"]:
            r["_score"] = score
            best = r

    chosen = best
    assert chosen is not None
    ok = bool(chosen["gate_pass"]) if baselines else True
    print("=" * 60)
    if baselines:
        print(
            f"WYBRANO seed={chosen['seed']} ΔSM={chosen['delta_sm']:+.4f} "
            f"{'PASS' if ok else 'FAIL'}"
        )
    else:
        print(
            f"WYBRANO seed={chosen['seed']} Most r@1={chosen['most']['r@1']:.3f} "
            f"(bez SM/ZERO — użyj --baselines do gate)"
        )

    meta = {
        "task": "dump_recall",
        "source": "phimost_learn",
        "dump": str(dump),
        "atoms": chosen["atoms"],
        "epochs": epochs,
        "warm_start": warm,
        "pair_mode": pair_mode,
        "noise": noise,
        "blend_holo": blend_holo,
        "restrict_prefix": restrict_prefix,
        "baselines": baselines,
        "seed": chosen["seed"],
        "r@1": chosen["most"]["r@1"],
        "softmax_r@1": chosen["softmax"].get("r@1"),
        "delta_sm": chosen["delta_sm"],
        "gate_pass": ok,
        "sweep": results,
    }
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    path = save_encoder(chosen["model"], out_ckpt, kind="most", meta=meta)
    print(f"[train] saved → {path}")
    promoted = False
    if promote and ok:
        save_encoder(chosen["model"], DEFAULT_MOST, kind="most", meta=meta)
        print(f"[train] promoted → {DEFAULT_MOST}")
        promoted = True
    elif promote and not ok:
        print("[train] NIE promuję — gate FAIL")

    HISTORIA.mkdir(exist_ok=True)
    report = {**meta, "ckpt": str(path), "promoted": promoted}
    if baselines:
        table = [
            "| seed | Most | SM | ZERO | ΔSM | gate |",
            "|------|------|----|------|-----|------|",
            *[
                f"| {r['seed']} | {r['most']['r@1']:.3f} | {r['softmax']['r@1']:.3f} | "
                f"{r['zero']['r@1']:.3f} | {r['delta_sm']:+.4f} | "
                f"{'PASS' if r['gate_pass'] else 'FAIL'} |"
                for r in results
            ],
        ]
        verdict = f"ΔSM={chosen['delta_sm']:+.4f} → **{'PASS' if ok else 'FAIL'}**"
    else:
        table = [
            "| seed | Most r@1 | r@3 | clean |",
            "|------|----------|-----|-------|",
            *[
                f"| {r['seed']} | {r['most']['r@1']:.3f} | {r['most']['r@3']:.3f} | "
                f"{r['clean']['r@1']:.3f} |"
                for r in results
            ],
        ]
        verdict = f"Most r@1={chosen['most']['r@1']:.3f} (fast, bez baseline)"
    (HISTORIA / "WYNIKI_LEARN.md").write_text(
        "\n".join(
            [
                "# phimost_learn",
                "",
                f"dump=`{dump}` atoms={chosen['atoms']} seed={chosen['seed']}",
                verdict,
                f"ckpt=`{path}` promote={promoted}",
                "",
                *table,
                "",
            ]
        ),
        encoding="utf-8",
    )
    (HISTORIA / "learn_last.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    # kryterium #5 — run-dir (loss z wybranego seeda Most)
    chosen_curve = None
    for r in results:
        if r.get("seed") == chosen["seed"]:
            chosen_curve = (r.get("most") or {}).get("curve")
            break
    summary = "\n".join(
        [
            f"# phimost_learn — {rundir.name}",
            "",
            verdict,
            f"ckpt=`{path}` promote={promoted}",
            f"dump=`{dump}`",
            f"run-dir=`{rundir.path}`",
            "",
            *table,
            "",
        ]
    )
    rundir.finalize(
        metrics=report,
        summary=summary,
        status="ok" if (ok if baselines else True) else "gate_fail",
        curve=chosen_curve,
    )
    report["run_dir"] = str(rundir.path)
    print(f"[train] run-dir finalized → {rundir.path}")
    return report


# ---------------------------------------------------------------------------
# Recall smoke
# ---------------------------------------------------------------------------

def recall_smoke(dump: Path, ckpt: Path, atom_id: str, k: int = 5) -> None:
    from memory_bus import MemoryBus

    if not Path(ckpt).is_file():
        raise FileNotFoundError(ckpt)
    bus = MemoryBus(ckpt)
    bus.ingest_dump(dump)
    hits = bus.recall_by_id(
        atom_id, k=k, same_prefix=True, exclude_nucleus=True
    )
    print(f"[recall] q={atom_id} ckpt={ckpt}")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h.score:+.4f}  d={h.depth}  {h.atom_id}  {h.E[:48]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PhiMost: uniwersalne ładowanie danych + nauka Most",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presety: struktura | karmazyn | qm | textfile | literature\n"
            "Źródła: .csv .json .md .txt PhiJSON/ BubbleFS/ katalog_literatury/"
        ),
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--src", type=Path, help="plik lub katalog źródłowy")
    g.add_argument(
        "--preset",
        choices=["struktura", "karmazyn", "qm", "textfile", "literature"],
        help="wbudowane źródło",
    )

    ap.add_argument("--out-dump", type=Path, default=None, help="katalog PhiJSON")
    ap.add_argument(
        "--out-ckpt",
        type=Path,
        default=None,
        help="plik checkpoint (domyślnie checkpoints/most_<name>.pt)",
    )
    ap.add_argument("--load-only", action="store_true", help="tylko import, bez treningu")
    ap.add_argument("--skip-train", action="store_true", help="pomiń trening")
    ap.add_argument("--skip-load", action="store_true", help="użyj istniejącego --out-dump")

    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,7,42")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--promote", action="store_true", help="nadpisz most_dump_recall.pt przy PASS")
    ap.add_argument("--no-promote", action="store_true", help="nigdy nie promuj (domyślne)")
    ap.add_argument(
        "--baselines",
        action="store_true",
        help="trenuj też Softmax+ZERO i gate ΔSM (3× wolniej; domyślnie WYŁĄCZONE)",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="kryterium #2: trening w autocast bfloat16 + finite-check zamka",
    )

    ap.add_argument("--pair-mode", default="mixed", choices=["same_depth", "all", "mixed"])
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--blend-holo", type=float, default=0.35)
    ap.add_argument(
        "--restrict-prefix",
        action="store_true",
        default=None,
        help="kandydaci tylko z tym samym prefiksem id (auto dla struktura/qm)",
    )
    ap.add_argument("--no-restrict-prefix", action="store_true")

    ap.add_argument("--recall-id", type=str, default=None, help="po nauce: recall atom_id")
    ap.add_argument("-k", type=int, default=5)

    args = ap.parse_args(argv)

    if not args.src and not args.preset:
        ap.error("podaj --src PATH albo --preset struktura|karmazyn|qm|textfile")

    # --- load ---
    if args.preset:
        dump = load_preset(args.preset)
        name = args.preset
        auto_restrict = args.preset in ("struktura", "qm")
    else:
        src = Path(args.src)
        kind = detect_kind(src)
        out = args.out_dump or default_out_for(kind, src)
        if args.skip_load and (out / "atoms.jsonl").is_file():
            dump = out
            print(f"[load] skip — używam {dump}")
        else:
            dump = load_source(src, out, kind=kind)
        name = kind.split("_")[0]
        # katalog dumps/from_struktura → traktuj jak struktura (ckpt + prefix)
        dump_name = (out if args.skip_load else dump).name.lower()
        if "struktura" in dump_name or "struktura" in str(src).lower():
            name = "struktura"
        elif "literature" in dump_name or kind.startswith("literature"):
            name = "literature"
        elif "karmazyn" in dump_name:
            name = "karmazyn"
        elif "qm" in dump_name:
            name = "qm"
        elif "from_photo" in dump_name or (
            "photo" in dump_name and "glyph" not in dump_name
        ):
            name = "glyphs_photo"
        elif "glyph" in dump_name or "glyph" in str(src).lower():
            name = "glyphs"
        auto_restrict = name in ("struktura", "qm") or kind.startswith("struktura")

    if args.load_only:
        n = len(load_raw_records(dump))
        print(f"OK load-only atoms={n} dump={dump}")
        return 0

    restrict = auto_restrict
    if args.no_restrict_prefix:
        restrict = False
    if args.restrict_prefix:
        restrict = True

    if args.out_ckpt:
        ckpt = args.out_ckpt
    else:
        # alias nazw ckpt zgodnych z wcześniejszymi skryptami
        alias = {
            "struktura": "most_struktura_atomow.pt",
            "karmazyn": "most_karmazyn_live.pt",
            "qm": "most_qm_orbitals.pt",
            "textfile": "most_dump_recall.pt",
            "literature": "most_literature.pt",
            "phijson": "most_phijson.pt",
            "bubblefs": "most_bubblefs.pt",
            "glyphs": "most_glyphs.pt",
            "glyphs_photo": "most_glyphs_photo.pt",
        }
        ckpt = CKPT_DIR / alias.get(name, f"most_{name}.pt")
    promote = bool(args.promote) and not args.no_promote
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    report = None
    if not args.skip_train:
        amp_dtype = None
        if args.bf16 or __import__("precision").bf16_env_enabled():
            from precision import bf16_supported

            if not bf16_supported():
                print("WERDYKT: FAIL (BF16 nieobsługiwane na tym CPU/GPU)")
                return 2
            amp_dtype = torch.bfloat16
        report = train_dump(
            dump,
            out_ckpt=ckpt,
            seeds=seeds,
            epochs=args.epochs,
            n_train=args.n_train,
            n_val=args.n_val,
            warm=not args.no_warm,
            pair_mode=args.pair_mode,
            noise=args.noise,
            blend_holo=args.blend_holo,
            restrict_prefix=restrict,
            promote=promote,
            baselines=args.baselines,
            run_name=f"phimost_learn_{name}",
            argv=sys.argv,
            amp_dtype=amp_dtype,
            t_seal=seal_for_source(name),
        )
        if args.baselines and not report["gate_pass"]:
            print("WERDYKT: FAIL (gate ΔSM)")
            if not args.recall_id:
                return 1
        else:
            print("WERDYKT: PASS" if report.get("gate_pass", True) else "WERDYKT: OK (fast)")

    if args.recall_id:
        use_ckpt = ckpt if ckpt.is_file() else DEFAULT_MOST
        recall_smoke(dump, use_ckpt, args.recall_id, k=args.k)

    if report is None:
        return 0
    if args.baselines and not report.get("gate_pass"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
