"""Regresja poprawek z przeglądu kodu (seal, recall, rdzenie, pętle, puste gałki)."""

from __future__ import annotations

import _labroot  # root labu na sys.path

import json
import os
import tempfile
from pathlib import Path

import torch

from nbody_data import solar_system_batch
from dump_recall import graph_to_tensors, load_raw_records, records_to_graph
from phi_dump import load_edges_jsonl, load_dump, strip_all_payloads
from phi_schema import PhiState


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


def test_text_query_matches_E_half():
    """Tekst trafia w połowę E contentu, nie w hash całej osi."""
    import torch.nn.functional as F
    from phi_schema import PhiAtom, atom_content_vec, text_as_content_vec

    d = 8
    hit = PhiAtom("h", "S-hit", "energia-orbitalu", 0.9, PhiState.WARM, "b", 0)
    miss = PhiAtom("m", "S-miss", "zupelnie-inny-opis", 0.9, PhiState.WARM, "b", 0)
    q = F.normalize(text_as_content_vec("energia-orbitalu", d), dim=-1)
    c = F.normalize(
        torch.stack([atom_content_vec(miss, d), atom_content_vec(hit, d)]),
        dim=-1,
    )
    assert int((c @ q).argmax().item()) == 1
    assert torch.dot(q, c[1]) > torch.dot(q, c[0])


def test_rn_core_and_full_table():
    from import_struktura_atomow import expand_configuration

    assert sum(c for _, _, c in expand_configuration("[Rn]")) == 86
    path = _labroot.ROOT / "archiwum" / "dane" / "struktura_atomow_full.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 118
    rn = 0
    for row in rows:
        cfg = str(row["electron_configuration"])
        if "[Rn]" in cfg:
            rn += 1
        n_e = sum(c for _, _, c in expand_configuration(cfg))
        assert n_e == int(row["atom_number"]), (row["symbol"], cfg, n_e)
    assert rn > 0


def test_inode_rank_clips_at_two():
    from bubblefs_import import _inode_ranks

    ranks = _inode_ranks([{"inode": v} for v in (10, 3, 7, 1, 9)])
    assert [ranks[i] for i in sorted(ranks)] == [0, 1, 2, 2, 2]


def test_shared_secret_decrypts_hot_payload():
    import base64

    from bubblefs_import import _export_key, _xor_crypt, bbl_to_atom

    secret = b"test-secret"
    bid = "b1"
    plain = "softplus tracer"
    cipher = _xor_crypt(plain.encode(), _export_key(secret, bid))
    bbl = {
        "id": bid,
        "label": "lab",
        "content_b64": base64.b64encode(cipher).decode(),
        "content_encrypted": True,
    }
    opened = bbl_to_atom(bbl, shared_secret=secret)
    assert opened.state == PhiState.HOT
    assert opened.payload and "softplus tracer" in opened.payload
    closed = bbl_to_atom(bbl, shared_secret=None)
    assert closed.payload is None


def test_three_body_dim4_is_radius_not_slot():
    from nbody_data import three_body_batch

    batch = three_body_batch(4, seed=0)
    col = batch.tracer[:, :, 4]
    slot = torch.tensor([0.0, 0.5, 1.0])
    assert float(col.std()) > 0.0
    assert not torch.allclose(col, slot.view(1, 3).expand_as(col))


def test_parent_edges_point_at_atoms():
    from phi_schema import PhiAtom, layer_parent_edges

    depths = [0] * 4 + [1] * 8 + [2] * 4
    atoms = [
        PhiAtom(f"a0_{i}", "s", "e", 1.0, PhiState.HOT, f"bub-d{d}", d)
        for i, d in enumerate(depths)
    ]
    edges = layer_parent_edges(atoms)
    ids = {a.atom_id for a in atoms}
    assert edges
    for edge in edges:
        assert edge.kind == "parent"
        assert edge.src in ids and edge.dst in ids
    assert any(e.src == "a0_4" and e.dst == "a0_0" for e in edges)
    assert any(e.src == "a0_12" and e.dst == "a0_4" for e in edges)


def test_t_seal_persists_false_and_legacy_struktura():
    from checkpoint import DEFAULT_MOST, load_encoder, save_encoder
    from gen_head import PhiGenHead, load_gen_bundle, save_gen_bundle
    from recall_model import PhiBridgeEncoder

    enc = PhiBridgeEncoder(
        input_dim=8, d=32, h=2, dim_tracer=6, num_layers=1, t_seal=False
    )
    opened = PhiBridgeEncoder(
        input_dim=8, d=32, h=2, dim_tracer=6, num_layers=1, t_seal=None
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        false_path = root / "sealed_off.pt"
        none_path = root / "sealed_none.pt"
        save_encoder(enc, false_path)
        save_encoder(opened, none_path)
        loaded, _ = load_encoder(false_path)
        loaded_none, _ = load_encoder(none_path)
        assert loaded.t_seal is False
        assert loaded_none.t_seal is None

        existed = DEFAULT_MOST.is_file()
        gen_path = root / "gen.pt"
        try:
            save_gen_bundle(enc, PhiGenHead(32, 8, 6), gen_path)
            gen_enc, _, _ = load_gen_bundle(gen_path)
            assert gen_enc.t_seal is False
        finally:
            if not existed and DEFAULT_MOST.is_file():
                DEFAULT_MOST.unlink()

    ckpt = _labroot.ROOT / "checkpoints" / "most_struktura_atomow.pt"
    if not ckpt.is_file():
        return
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location="cpu")
    meta = payload.get("meta") or {}
    blob = " ".join(str(meta.get(k, "")) for k in ("dump", "source", "task")).lower()
    if any(tok in blob for tok in ("struktura", "qm_orbital", "glyphs_photo", "from_photo")):
        legacy, _ = load_encoder(ckpt)
        assert legacy.t_seal is False


def test_with_most_knobs_keeps_side_and_dropout():
    from lm_tiny import MostKnobs, TinyLM, TinyLMConfig, with_most_knobs

    cfg = with_most_knobs(TinyLMConfig(side_sources=3, dropout=0.1), MostKnobs())
    assert cfg.side_sources == 3
    assert cfg.dropout == 0.1
    assert cfg.most == MostKnobs()

    try:
        TinyLM(
            TinyLMConfig(
                d=32,
                h=2,
                num_layers=1,
                max_seq=8,
                most=MostKnobs(delta_mode="directed", delta_alpha=0.2),
            ),
            attn="most",
        )
        raise AssertionError("directed delta_mode powinien paść")
    except ValueError as exc:
        assert "square" in str(exc)

    model = TinyLM(TinyLMConfig(d=32, h=2, num_layers=1, max_seq=8), attn="most")
    assert not hasattr(model.blocks[0], "delta_alpha")
    assert not hasattr(model.blocks[0], "directed_bias")
    assert model.blocks[0].drop_p == 0.0
    assert not any("dropout" in key for key in model.state_dict())


def test_dropout_changes_train_forward_only_when_positive():
    from lm_tiny import TinyLM, TinyLMConfig

    cfg = TinyLMConfig(dropout=0.5, d=32, h=2, num_layers=1, vocab_size=32, max_seq=16)
    model = TinyLM(cfg, attn="softmax")
    model.train()
    idx = torch.randint(0, 32, (2, 8))
    torch.manual_seed(1)
    y1 = model(idx)
    torch.manual_seed(2)
    y2 = model(idx)
    assert not torch.allclose(y1, y2)

    plain = TinyLM(
        TinyLMConfig(dropout=0.0, d=32, h=2, num_layers=1, vocab_size=32, max_seq=16),
        attn="softmax",
    )
    plain.eval()
    a = plain(idx)
    plain.train()
    b = plain(idx)
    assert torch.allclose(a, b)


def test_missing_dump_does_not_build_textfile():
    from phimost_gen import DUMP, train_on_dump

    missing = Path(tempfile.mkdtemp()) / "nie_ma_takiego_dumpu"
    before = (DUMP / "atoms.jsonl").stat().st_mtime if (DUMP / "atoms.jsonl").is_file() else None
    try:
        train_on_dump(missing, epochs=1)
        raise AssertionError("brakujący dump powinien rzucić FileNotFoundError")
    except FileNotFoundError:
        pass
    assert not missing.exists()
    after = (DUMP / "atoms.jsonl").stat().st_mtime if (DUMP / "atoms.jsonl").is_file() else None
    assert before == after


def test_literature_chunk_is_hot_with_payload():
    from import_literature import file_to_graph

    body = "Akapit o moście Breita i tracerze orbitalnym, dość długi żeby przejść próg. "
    text = "# Tytul\n\n" + body * 2 + "\n"
    folder = Path(tempfile.mkdtemp())
    path = folder / "note.md"
    path.write_text(text, encoding="utf-8")
    graph = file_to_graph(path)
    chunks = [atom for atom in graph.atoms if "/chunk/" in atom.S]
    assert chunks
    assert all(atom.state == PhiState.HOT and atom.payload for atom in chunks)


def test_score_metrics_rank_the_positive():
    from recall_model import metrics_from_scores

    scores = torch.tensor([[0.1, 0.9, 0.2], [0.4, 0.1, 0.0]])
    pos = torch.tensor([1, 0])
    metrics = metrics_from_scores(scores, pos, ks=(1, 3))
    assert metrics["r@1"] == 1.0
    assert metrics["mrr"] == 1.0


if __name__ == "__main__":
    test_solar_moon_not_giant_path()
    test_strip_payload_changes_dump_content()
    test_neural_uses_encoder_not_hash_only()
    test_live_host_fail_closed()
    test_text_query_matches_E_half()
    test_rn_core_and_full_table()
    test_inode_rank_clips_at_two()
    test_shared_secret_decrypts_hot_payload()
    test_three_body_dim4_is_radius_not_slot()
    test_parent_edges_point_at_atoms()
    test_t_seal_persists_false_and_legacy_struktura()
    test_with_most_knobs_keeps_side_and_dropout()
    test_dropout_changes_train_forward_only_when_positive()
    test_missing_dump_does_not_build_textfile()
    test_literature_chunk_is_hot_with_payload()
    test_score_metrics_rank_the_positive()
    print("OK — review fixes")
