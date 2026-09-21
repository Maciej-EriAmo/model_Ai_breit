"""Faza A — ontologia: ten sam tekst jako Unix flat ≠ Φ-graph."""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

import torch

from synthetic_phi import generate_phi_tree, same_text_two_views


def test_phi_batch_shapes():
    batch = generate_phi_tree(4, seq_len=16, d_x=8, n_dim=6)
    assert batch.content.shape == (4, 16, 8)
    assert batch.tracer.shape == (4, 16, 6)
    assert batch.labels.shape == (4,)
    assert batch.mask.sum() == 4 * 16


def test_tracer_has_depth_geometry():
    batch = generate_phi_tree(2, n_dim=6)
    # atomy 0 i 12: różne depth bins → różne pierwsze 3 współrzędne tracera
    t0 = batch.tracer[0, 0, :3]
    t12 = batch.tracer[0, 12, :3]
    assert not torch.allclose(t0, t12), "depth bins muszą różnić tracer"
    assert batch.tracer[0, 0, 0] == 1.0
    assert batch.tracer[0, 12, 2] == 1.0


def test_text_file_is_not_flat():
    text = "Karmazyn nie jest systemem plikow. Atom A=(S,E,T). Plik tekstowy ma warstwy."
    flat, phi = same_text_two_views(text, n_atoms=8, d_x=8, n_dim=6)

    assert flat.shape == (8, 8)
    assert phi.content.shape == (1, 8, 8)
    assert phi.tracer.shape == (1, 8, 6)

    # content z Φ ≠ same flat chunki (S/E/state mieszają embed)
    assert not torch.allclose(flat, phi.content[0], atol=1e-5)

    # flat nie ma tracera Φ; Φ ma niezerową geometrię depth/T/state
    assert phi.tracer.abs().sum() > 0
    # różne depth → różne one-hot
    assert not torch.allclose(phi.tracer[0, 0, :3], phi.tracer[0, 7, :3])

    # WARM/COLD bez payload: content nadal z S/E (nie wymaga bajtów)
    from phi_schema import PhiState

    # states w batchu: nie wszystkie HOT
    assert (phi.states[0] != int(PhiState.HOT)).any()


def test_zero_tracer_control_shape():
    batch = generate_phi_tree(2, n_dim=6, use_tracer=False)
    assert torch.count_nonzero(batch.tracer) == 0


if __name__ == "__main__":
    test_phi_batch_shapes()
    test_tracer_has_depth_geometry()
    test_text_file_is_not_flat()
    test_zero_tracer_control_shape()
    print("OK — Faza A: plik tekstowy nie jest płaski")
