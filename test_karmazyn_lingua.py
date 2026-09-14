"""
Testy karmazyn_lingua: overlay hologram vs symbolic, LinguaLayer nad
prawdziwym Store (bind pozycyjny n+1, unbind, idempotencja), most PhiJSON
do reszty repo (to_phi_atoms + phi_dump).

Przepisane pod API v4 (LinguaLayer nad realnym karmazyn_substrate.Store —
nie ma już osobnych klas MorphAtom/LinguisticBubble ani opcjonalnego
connect_real_substrate: ta warstwa zawsze zakłada, że karmazyn_substrate/
karmazyn_hrr są zainstalowane, patrz nagłówek karmazyn_lingua.py).

Styl jak test_holo_AT.py / test_transcode_contract.py: gołe funkcje test_*
z assert, bez frameworka.
"""

from __future__ import annotations

import math

import numpy as np

from karmazyn_lingua import (
    FEATURE_DIM,
    AtomEncoding,
    FeatureOverlay,
    LinguaLayer,
    feature_vector,
    przyklad_pisac_rodzina,
    to_phi_atoms,
)
from phi_dump import load_atoms_jsonl, load_edges_jsonl  # noqa: F401 (kontrakt sanity poniżej)


def _fresh_lingua() -> LinguaLayer:
    import karmazyn_substrate as sub

    store = sub.Store()
    return LinguaLayer(store, dim=sub.VEC_DIM)


# --- 1. feature_vector(): hologram odróżnia to, czego symbolic nie może ----

def test_symbolic_shared_slot_indistinguishable() -> None:
    """Trzy prefiksy o tej samej amplitude na wspólnym slocie (slot=1,
    czyli wymiary [2,3]=Re,Im) muszą kolapsować do jednej wartości pod
    symbolic (Im zawsze 0) — to słaby baseline, nie przewaga hologramu."""
    feats = {
        "na-": FeatureOverlay("a_na", 0.8, math.pi / 4, 1),
        "prze-": FeatureOverlay("a_prze", 0.8, math.pi / 3, 1),
        "wy-": FeatureOverlay("a_wy", 0.8, math.pi / 6, 1),
    }
    slot1 = {round(feature_vector([f], AtomEncoding.SYMBOLIC)[2], 6) for f in feats.values()}
    assert len(slot1) == 1, f"symbolic powinien zlewać slot1 (Re), dostał {slot1}"


def test_hologram_distinguishes_same_amplitude_by_phase() -> None:
    feats = {
        "na-": FeatureOverlay("a_na", 0.8, math.pi / 4, 1),
        "prze-": FeatureOverlay("a_prze", 0.8, math.pi / 3, 1),
        "wy-": FeatureOverlay("a_wy", 0.8, math.pi / 6, 1),
    }
    slot1 = {
        (round(v[2], 6), round(v[3], 6))
        for v in (feature_vector([f], AtomEncoding.HOLOGRAM) for f in feats.values())
    }
    assert len(slot1) == 3, f"hologram (Re,Im) powinien rozróżnić trzy fazy, dostał {slot1}"
    # amplitude=0.8 dla wszystkich trzech -> |Re+iIm| <= 0.8
    assert all(abs(complex(re, im)) <= 0.8 + 1e-9 for re, im in slot1)


def test_onehot_dedicated_dims_no_collision() -> None:
    """Twardy baseline: slot adresuje wymiar BEZPOŚREDNIO (nie 2*slot jak
    hologram/symbolic) — dwie różne wartości na różnych slotach lądują na
    rozłącznych, jednowymiarowych pozycjach, bez śladu jedna u drugiej."""
    a = feature_vector([FeatureOverlay("a", 0.8, 0.0, 0)], AtomEncoding.SYMBOLIC_ONEHOT)
    b = feature_vector([FeatureOverlay("b", 0.8, 0.0, 1)], AtomEncoding.SYMBOLIC_ONEHOT)
    assert a[0] == 0.8 and a[1] == 0.0
    assert b[1] == 0.8 and b[0] == 0.0
    assert np.count_nonzero(a) == 1 and np.count_nonzero(b) == 1


def test_onehot_uses_full_feature_dim_range() -> None:
    """Bez fazy do zmieszczenia obok amplitude, onehot ma do dyspozycji
    pełne FEATURE_DIM=64 sloty, nie tylko N_SLOTS=32 jak hologram/symbolic."""
    v = feature_vector([FeatureOverlay("last", 1.0, 0.0, FEATURE_DIM - 1)], AtomEncoding.SYMBOLIC_ONEHOT)
    assert v[FEATURE_DIM - 1] == 1.0
    try:
        feature_vector([FeatureOverlay("oob", 1.0, 0.0, FEATURE_DIM)], AtomEncoding.SYMBOLIC_ONEHOT)
    except ValueError:
        pass
    else:
        raise AssertionError("slot >= FEATURE_DIM powinien rzucić ValueError dla onehot")


def test_onehot_hard_baseline_perfect_retrieval() -> None:
    """Gdy każdy prefiks dostaje własny dedykowany slot (nie współdzielony
    jak w _run_r1), retrieval jest bezstratny niezależnie od fazy — górna
    granica r@1=1.0, punkt odniesienia dla słabego symbolic (r@1=0.33)."""
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.SYMBOLIC_ONEHOT)
    prefiksy_slot = {"na-": 1, "prze-": 2, "wy-": 3}
    prefiksy = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, 0.0, slot)], AtomEncoding.SYMBOLIC_ONEHOT)
        for n, slot in prefiksy_slot.items()
    }
    candidates = list(prefiksy.values())

    for nazwa, prefiks in prefiksy.items():
        slowo = lingua.compose(f"{nazwa}pisać_onehot", [prefiks, rdzen], kind="word")
        odzyskany = lingua.unbind_position(slowo, position=1)
        top1 = lingua.nearest_morpheme(odzyskany, candidates, top_k=1)[0][1]
        assert top1.E == nazwa, f"onehot powinien być bezstratny, dostał top1={top1.E!r} dla {nazwa!r}"


def test_slot_out_of_range_raises() -> None:
    bad = [FeatureOverlay("x", amplitude=1.0, phase=0.0, slot=999)]
    try:
        feature_vector(bad, AtomEncoding.HOLOGRAM)
    except ValueError:
        pass
    else:
        raise AssertionError("slot poza N_SLOTS powinien rzucić ValueError")


# --- 2. LinguaLayer nad realnym Store: bind pozycyjny, nie suma ------------

def test_compose_uses_positional_bind_not_plain_sum() -> None:
    """Kolejność argumentów compose() zmienia wynik (bind(P^i, .) zależy od
    pozycji) — czysta suma elementwise dałaby ten sam wektor niezależnie od
    kolejności. To odróżnia prawdziwy HRR bind od fallbacku bez HRR."""
    lingua = _fresh_lingua()
    a = lingua.make_morpheme("a", [FeatureOverlay("fa", 1.0, 0.3, 0)], AtomEncoding.HOLOGRAM)
    b = lingua.make_morpheme("b", [FeatureOverlay("fb", 1.0, 1.1, 1)], AtomEncoding.HOLOGRAM)

    ab = lingua.compose("ab", [a, b], kind="word")
    ba = lingua.compose("ba", [b, a], kind="word")

    fv_ab = np.asarray(ab.vector[:FEATURE_DIM], dtype=float)
    fv_ba = np.asarray(ba.vector[:FEATURE_DIM], dtype=float)
    naive_sum = np.asarray(a.vector[:FEATURE_DIM], dtype=float) + np.asarray(b.vector[:FEATURE_DIM], dtype=float)

    assert not np.allclose(fv_ab, fv_ba), "kolejność powinna wpływać na wynik (bind pozycyjny, nie suma)"
    assert not np.allclose(fv_ab, naive_sum), "compose() nie powinien być zwykłą sumą, gdy HRR jest dostępny"


def test_unbind_position_zero_raises() -> None:
    """P^0 jest elementem identycznościowym splotu — unbind(...,P^0) nic
    nie separuje, więc position musi być >=1 (patrz nagłówek modułu)."""
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    try:
        lingua.unbind_position(slowa[0], position=0)
    except ValueError:
        pass
    else:
        raise AssertionError("position=0 powinno rzucić ValueError")


def _r1_for_phases(fazy: dict[str, float], *, label: str) -> float:
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiksy = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
        for n, ph in fazy.items()
    }
    candidates = list(prefiksy.values())

    trafienia = 0
    for nazwa, prefiks in prefiksy.items():
        slowo = lingua.compose(f"{nazwa}{label}", [prefiks, rdzen], kind="word")
        odzyskany = lingua.unbind_position(slowo, position=1)
        top1 = lingua.nearest_morpheme(odzyskany, candidates, top_k=1)[0][1]
        trafienia += top1.E == nazwa
    return trafienia / len(fazy)


def test_unbind_position_recovers_correct_morpheme_far_phases() -> None:
    """r@1 = 1.0 dla faz rozstawionych 0/120/240 stopni — dokładnie wynik
    opisany (i zweryfikowany empirycznie) w nagłówku karmazyn_lingua.py."""
    fazy = {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3}
    assert _r1_for_phases(fazy, label="pisac_far") == 1.0


def test_unbind_position_subtracts_known_interference_close_phases() -> None:
    """Regresja na poprawkę z lingua_probe_mlp.py: dla faz stłoczonych
    30°-60° goły unbind+nearest_morpheme dawał r@1=0.33 (poziom przypadku),
    bo mieszał w wynik interferencję od WSPÓLNEGO rdzenia. unbind_position
    teraz odejmuje ten znany, dokładny wkład zanim odwiąże pozycję 1 —
    r@1 musi wrócić do 1.0, tak jak dla faz rozstawionych (sygnał był
    zawsze odzyskiwalny, tylko zła metoda go traciła)."""
    fazy = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    assert _r1_for_phases(fazy, label="pisac_close") == 1.0


def test_get_or_create_idempotent() -> None:
    """Ten sam morfem/atom użyty ponownie jest tworzony RAZ i wykorzystany
    ponownie (get_or_create), nie duplikowany w Store."""
    lingua = _fresh_lingua()
    a1 = lingua.make_morpheme("dup", [FeatureOverlay("f", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    a2 = lingua.make_morpheme("dup", [FeatureOverlay("f", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    assert a1.id == a2.id
    assert np.array_equal(a1.vector, a2.vector)


def test_composite_atoms_are_persisted_in_store() -> None:
    """compose() rejestruje złożenie jako nowy atom w tym samym realnym
    Store, nie tylko w pamięci Pythona — sprawdzalne przez store.get_atom."""
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    for slowo in slowa:
        assert lingua.store.has_atom(slowo.id)
        assert lingua.store.get_atom(slowo.id).id == slowo.id


# --- 3. Most bridge: to_phi_atoms -> kontrakt PhiJSON tego repo ------------

def test_to_phi_atoms_shape_and_recursion_depth() -> None:
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(lingua, slowa)

    ids = [a["id"] for a in atoms]
    assert len(ids) == len(set(ids)), "id atomów muszą być unikalne"
    assert {"corpus", "lingua_page"} <= set(ids)

    # page depth=0, bubble(słowo) depth=1, morfem depth=2
    words = [a for a in atoms if a["S"].startswith("lingua:bubble:")]
    assert len(words) == 3
    assert all(w["depth"] == 1 for w in words)
    morphs = [a for a in atoms if a["S"].startswith("lingua:morpheme:")]
    assert len(morphs) == 6  # 3 słowa x (prefiks + rdzeń), rdzeń dzielony ale liczony per-parent
    assert all(m["depth"] == 2 for m in morphs)

    # każdy atom poza korzeniami (corpus, page) ma dokładnie jedną krawędź parent
    parent_edges = [e for e in edges if e["kind"] == "parent"]
    assert len(parent_edges) == len(atoms) - 2

    assert meta["n_top_bubbles"] == 3
    assert meta["ref_energy"] > 0


def test_to_phi_atoms_all_required_keys_present() -> None:
    """phi_dump.load_atoms_jsonl / load_edges_jsonl czytają te dokładne
    klucze — sprawdzamy kontrakt wprost, żeby przyszła zmiana pól w
    to_phi_atoms nie zepsuła loadera cicho."""
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.SYMBOLIC)
    atoms, edges, _meta = to_phi_atoms(lingua, slowa)
    required_atom_keys = {"id", "S", "E", "T", "state", "bubble", "depth", "payload", "mazur_tracer"}
    for a in atoms:
        assert required_atom_keys <= set(a.keys()), a
        assert a["state"] in ("HOT", "WARM", "COLD", "TOMB")
        assert 0.0 <= a["T"] <= 1.0
    required_edge_keys = {"kind", "src", "dst", "name"}
    for e in edges:
        assert required_edge_keys <= set(e.keys()), e


def test_to_phi_atoms_roundtrips_through_phi_dump(tmp_path) -> None:
    import json

    from bubblefs_import import export_graph_as_dump  # noqa: F401 (nieużywane bezpośrednio; sanity import)
    from phi_dump import graph_to_content_tracer, load_dump

    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(lingua, slowa)

    out = tmp_path / "from_lingua_test"
    out.mkdir()
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({"source": "test", **meta}), encoding="utf-8")

    man, g = load_dump(out)
    assert len(g.atoms) == len(atoms)
    assert len(g.edges) == len(edges)
    batch = graph_to_content_tracer(g)
    assert batch.content.shape[0] == 1
    assert batch.content.shape[1] == len(atoms)


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        if t.__code__.co_argcount == 0:
            t()
        else:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
        print(f"OK {t.__name__}")
    print(f"\n{len(tests)} testy OK", file=sys.stderr)
