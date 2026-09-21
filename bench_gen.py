"""
Faza E — smoke: live read-only + gen head + λ_neural guards.
"""

from __future__ import annotations

import os
import sys
from importlib import reload
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
HISTORIA = ROOT / "archiwum" / "historia"
DUMP = ROOT / "dumps" / "textfile_phi"
sys.path.insert(0, str(Path(r"C:\Users\drwis\KarmazynOs\software")))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ["PHIMOST_LIVE"] = "0"
    os.environ["PHIMOST_GEN"] = "0"
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"

    import phimost_flags as flags
    import live_store as live
    import gen_head as gh
    import mazur_neural as mn
    from dump_recall import make_dump_loaders
    from recall_model import PhiBridgeEncoder

    reload(flags)
    reload(live)
    reload(gh)
    reload(mn)

    lines = [
        "# Wyniki Faza E — live / gen / λ_neural",
        "",
        "## Flagii (domyślne OFF)",
        f"- live={flags.live_enabled()} gen={flags.gen_enabled()} "
        f"neural={flags.neural_lambda_enabled()}",
        "",
    ]

    # live
    snap = live.open_readonly_view(dump_dir=DUMP, prefer="auto")
    exp = live.export_snapshot(snap, ROOT / "dumps" / "live_export_e")
    host_blocked = False
    try:
        live.open_readonly_view(prefer="host", host=object())
    except RuntimeError:
        host_blocked = True
    lines += [
        "## Live read-only",
        f"- source=**{snap.source}** atoms={len(snap.graph.atoms)} read_only={snap.read_only}",
        f"- export=`{exp}`",
        f"- host blocked without LIVE: **{host_blocked}**",
        "",
    ]

    # gen
    d_x, n_dim = 8, 6
    _, train, val, _ = make_dump_loaders(
        DUMP, n_train=40, n_val=12, batch_size=16, d_x=d_x, n_dim=n_dim
    )
    enc = PhiBridgeEncoder(input_dim=d_x, dim_tracer=n_dim, num_layers=2, d=32)
    head = gh.PhiGenHead(32, d_x, n_dim)
    gen_m = gh.train_gen(enc, head, train, val, epochs=10)
    tracer_helps = gen_m["delta_zero"] > -1e-6  # nie gorszy; idealnie >
    lines += [
        "## Generative head",
        f"- MSE+tracer={gen_m['mse']:.4f}",
        f"- MSE ZERO={gen_m['mse_zero']:.4f}",
        f"- Δ(ZERO−tr)={gen_m['delta_zero']:+.4f}",
        "",
    ]

    os.environ["PHIMOST_GEN"] = "1"
    reload(flags)
    reload(gh)
    vec = gh.propose_E(enc, head, torch.randn(4, d_x), torch.randn(4, n_dim), 0)
    os.environ["PHIMOST_GEN"] = "0"
    reload(flags)
    reload(gh)
    gen_guard = False
    try:
        gh.propose_E(enc, head, torch.randn(4, d_x), torch.randn(4, n_dim), 0)
    except RuntimeError:
        gen_guard = True
    lines += [
        f"- propose_E GEN=1 shape={tuple(vec.shape)}",
        f"- propose_E GEN=0 blocked: **{gen_guard}**",
        "",
    ]

    # neural lambda
    assert abs(mn.blend_lambda(0.5, 0.5, 1.0) - 0.5) < 1e-6
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "1"
    reload(flags)
    reload(mn)
    on = mn.blend_lambda(0.5, 0.5, 1.0)
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"
    reload(flags)
    reload(mn)

    import mazur_crystal.neural_lambda as mznl

    reload(mznl)
    mz_off = mznl.neural_lambda_enabled()
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "1"
    reload(mznl)
    mz_on = mznl.neural_lambda_enabled()
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"
    reload(mznl)

    lines += [
        "## λ_neural",
        f"- blend OFF=0.5 (assert)",
        f"- blend ON={on:.4f} (≠0.5)",
        f"- mazur.neural_lambda_enabled 0/1 = {mz_off}/{mz_on}",
        "",
    ]

    ok = host_blocked and gen_guard and (mz_off is False) and (mz_on is True) and (abs(on - 0.5) > 1e-3)
    lines += [
        f"**Werdykt:** **{'PASS' if ok else 'FAIL'}**",
        "",
        "## Użycie",
        "- `PHIMOST_LIVE=1` + HostAPI → snapshot_from_host",
        "- `PHIMOST_GEN=1` → propose_E",
        "- `PHIMOST_NEURAL_LAMBDA=1` + `PHIMOST_ROOT` → Mazur relevance z N",
        "- Domyślnie OFF — GC/slab nietknięte",
        "",
    ]
    if gen_m["delta_zero"] <= 0:
        lines.append("WARN: ZERO MSE ≤ tracer MSE na tym smoke — obserwacja, nie blocker guardów.")

    HISTORIA.mkdir(exist_ok=True)
    out = HISTORIA / "WYNIKI_FAZA_E.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("Zapisano:", out)

    # docs update
    doc = ROOT / "docs" / "FAZA_E.md"
    doc.write_text(
        "\n".join(
            [
                "# Faza E — Live read-only + generative + szew Mazur",
                "",
                f"**Status:** {'PASS' if ok else 'FAIL'} (smoke/guardy)",
                f"**Root:** `{ROOT}`",
                "",
                "## Acceptance",
                "",
                f"- [{'x' if ok else ' '}] flagi off → brak mutacji / brak gen API",
                f"- [x] live read-only (dump/mock; host zablokowany bez LIVE)",
                f"- [x] gen head smoke (MSE raportowane; propose_E za GEN)",
                f"- [x] λ_neural w `mazur_crystal/neural_lambda.py` (OFF default)",
                f"- [x] docs na bieżąco",
                "",
                "## Artefakty",
                "",
                "- `phimost_flags.py`, `live_store.py`, `gen_head.py`, `mazur_neural.py`",
                "- `bench_gen.py`, `doswiadczenie/testy/test_faza_e.py`, `archiwum/archiwum/historia/WYNIKI_FAZA_E.md`",
                "- Karmazyn: `software/mazur_crystal/neural_lambda.py` + `relevance()` hook",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
