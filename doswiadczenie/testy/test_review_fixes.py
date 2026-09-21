"""Regresja 4 bugów z external /review."""

from __future__ import annotations

import _labroot  # root labu na sys.path

import os

import torch

from nbody_data import solar_system_batch
from dump_recall import graph_to_tensors, load_raw_records, records_to_graph
from phi_dump import load_edges_jsonl, load_dump, strip_all_payloads
from phi_schema import PhiState
from pathlib import Path


def test_solar_moon_not_giant_path():
    """Moon near Earth; giants on Kepler a>>1 — nie odwrotnie."""
    batch = solar_system_batch(1, include_moon=True, seed=0)
    # bodies order: Sun…Neptune, Moon last
    # Jupiter index = 5 (after Sun+4 rocky)
    # Moon index = 9
    # r from content kinematics mixed — use energy: Moon should be near Earth energy-ish
    # Better: regenerate one sample and inspect positions via private path —
    # check hierarchy one-hots: Moon hier=1, Jupiter hier=2
    assert batch.tracer[0, 5, 2] == 1.0  # Jupiter giant
    assert batch.tracer[0, 9, 1] == 1.0  # Moon rocky
    # Jupiter |a| energy dim / content: giants farther — tracer dim4 log a higher for Jupiter than Moon
    assert batch.tracer[0, 5, 4] > batch.tracer[0, 9, 4]


def test_strip_payload_changes_dump_content():
    root = _labroot.ROOT / "dumps" / "textfile_phi"
    if not (root / "atoms.jsonl").exists():
        import make_dump_textfile

        make_dump_textfile.main()
    _, g = load_dump(root)
    assert any(a.payload for a in g.atoms if a.state == PhiState.HOT)
    c0, _, _, _ = graph_to_tensors(g, d_x=8, n_dim=6)
    g2 = strip_all_payloads(g)
    c1, _, _, _ = graph_to_tensors(g2, d_x=8, n_dim=6)
    assert not torch.allclose(c0, c1, atol=1e-5)


def test_neural_uses_encoder_not_hash_only():
    import phimost_flags as flags
    import mazur_neural as mn
    from importlib import reload
    from recall_model import PhiBridgeEncoder
    from phi_schema import PhiAtom, PhiState

    os.environ["PHIMOST_NEURAL_LAMBDA"] = "1"
    reload(flags)
    reload(mn)

    enc = PhiBridgeEncoder(input_dim=8, dim_tracer=6, num_layers=1, d=32)
    mn.set_encoder(enc)
    a = PhiAtom("a", "Sa", "unique-aaa", 1.0, PhiState.HOT, "b", 0, "payload-a", 1.0)
    p = PhiAtom("p", "Sp", "unique-ppp", 1.0, PhiState.HOT, "b", 0, "payload-p", 1.0)
    n1 = mn.neural_component_for_pair(a, p)
    assert n1 is not None and 0.0 <= n1 <= 1.0

    # without encoder → None
    mn.set_encoder(None)
    assert mn.neural_component_for_pair(a, p) is None

    os.environ["PHIMOST_NEURAL_LAMBDA"] = "0"
    reload(flags)
    reload(mn)


def test_live_host_fail_closed():
    os.environ["PHIMOST_LIVE"] = "1"
    import phimost_flags as flags
    import live_store as live
    from importlib import reload

    reload(flags)
    reload(live)

    class Boom:
        def list_atoms(self, *_a, **_k):
            raise RuntimeError("boom")

    try:
        live.snapshot_from_host(Boom())
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "failed" in str(e).lower() or "boom" in str(e).lower()

    class Empty:
        def list_atoms(self, *_a, **_k):
            return []

    try:
        live.snapshot_from_host(Empty())
        assert False, "expected empty error"
    except RuntimeError as e:
        assert "zero" in str(e).lower()

    os.environ["PHIMOST_LIVE"] = "0"
    reload(flags)
    reload(live)


if __name__ == "__main__":
    test_solar_moon_not_giant_path()
    test_strip_payload_changes_dump_content()
    test_neural_uses_encoder_not_hash_only()
    test_live_host_fail_closed()
    print("OK — 4 review bugs fixed")
