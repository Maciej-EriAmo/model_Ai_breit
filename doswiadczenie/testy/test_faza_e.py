"""Faza E — unit testy flag / live / gen guard / λ_neural."""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

import os
from importlib import reload

import torch


def test_flags_default_off():
    os.environ.pop("PHIMOST_LIVE", None)
    os.environ.pop("PHIMOST_GEN", None)
    os.environ.pop("PHIMOST_NEURAL_LAMBDA", None)
    import phimost_flags as f

    reload(f)
    assert f.live_enabled() is False
    assert f.gen_enabled() is False
    assert f.neural_lambda_enabled() is False


def test_live_readonly_dump_and_mock():
    from live_store import open_readonly_view, snapshot_mock, export_snapshot
    from pathlib import Path

    mock = snapshot_mock()
    assert mock.read_only and mock.source == "mock"
    snap = open_readonly_view(prefer="dump")
    assert snap.read_only
    out = export_snapshot(snap, Path("dumps") / "_test_export_e")
    assert (out / "atoms.jsonl").exists()


def test_live_host_blocked_without_flag():
    os.environ["PHIMOST_LIVE"] = "0"
    import phimost_flags as f
    import live_store as ls

    reload(f)
    reload(ls)
    try:
        ls.open_readonly_view(prefer="host", host=object())
        assert False, "should raise"
    except RuntimeError:
        pass


def test_gen_propose_guard():
    from recall_model import PhiBridgeEncoder
    from gen_head import PhiGenHead
    import gen_head as gh
    import phimost_flags as f

    enc = PhiBridgeEncoder(input_dim=8, dim_tracer=6, num_layers=1, d=32)
    head = PhiGenHead(32, 8, 6)
    c = torch.randn(3, 8)
    tr = torch.randn(3, 6)

    os.environ["PHIMOST_GEN"] = "0"
    reload(f)
    reload(gh)
    try:
        gh.propose_E(enc, head, c, tr, 0)
        assert False
    except RuntimeError:
        pass

    os.environ["PHIMOST_GEN"] = "1"
    reload(f)
    reload(gh)
    v = gh.propose_E(enc, head, c, tr, 0)
    assert v.shape == (8,)
    os.environ["PHIMOST_GEN"] = "0"


def test_blend_lambda_flag():
    import mazur_neural as mn
    import phimost_flags as f

    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"
    reload(f)
    reload(mn)
    assert abs(mn.blend_lambda(0.5, 0.5, 1.0) - 0.5) < 1e-6

    os.environ["PHIMOST_NEURAL_LAMBDA"] = "1"
    os.environ["PHIMOST_LAMBDA_NEURAL"] = "0.2"
    reload(f)
    reload(mn)
    # (0.2*0.5 + 0.8*0.5 + 0.2*1.0) / 1.2 = (0.1+0.4+0.2)/1.2 = 0.7/1.2
    assert abs(mn.blend_lambda(0.5, 0.5, 1.0) - (0.7 / 1.2)) < 1e-6
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"


def test_mazur_neural_lambda_module_default_off():
    import sys
    from pathlib import Path
    from importlib import reload

    sys.path.insert(0, str(Path(r"C:\Users\drwis\KarmazynOs\software")))
    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"
    import mazur_crystal.neural_lambda as nl

    reload(nl)
    assert nl.neural_lambda_enabled() is False


if __name__ == "__main__":
    test_flags_default_off()
    test_live_readonly_dump_and_mock()
    test_live_host_blocked_without_flag()
    test_gen_propose_guard()
    test_blend_lambda_flag()
    test_mazur_neural_lambda_module_default_off()
    print("OK — Faza E unit tests")
