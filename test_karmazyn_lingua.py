"""
Testy karmazyn_lingua: MorphAtom hologram vs symbolic, LinguisticBubble
(HRR bind rekurencyjny n+1), most PhiJSON do reszty repo (to_phi_atoms +
phi_dump), i (gdy dostępny) prawdziwy substrat KarmazynOS.

Styl jak test_holo_AT.py / test_transcode_contract.py: gołe funkcje test_*
z assert, bez frameworka.
"""

from __future__ import annotations

import math

from karmazyn_lingua import (
    AtomEncoding,
    FeatureOverlay,
    LinguisticBubble,
    MorphAtom,
    connect_real_substrate,
    przyklad_pisac_rodzina,
    przyklad_pisac_rodzina_bubbles,
    to_phi_atoms,
)
from phi_dump import load_atoms_jsonl, load_edges_jsonl  # noqa: F401 (kontrakt sanity poniżej)


# --- 1. MorphAtom: hologram odróżnia to, czego symbolic nie może ------------

def test_symbolic_shared_slot_indistinguishable() -> None:
    """Docstring modułu to zapowiada wprost: trzy prefiksy o tej samej
    amplitude na wspólnym slocie muszą kolapsować do jednej wartości pod
    symbolic — to słaby baseline, nie przewaga hologramu."""
    wyniki = przyklad_pisac_rodzina(AtomEncoding.SYMBOLIC)
    slot1 = {round(v[1], 6) for v in wyniki.values()}
    assert len(slot1) == 1, f"symbolic powinien zlewać slot1, dostał {slot1}"


def test_hologram_distinguishes_same_amplitude_by_phase() -> None:
    wyniki = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    slot1 = [round(v[1], 6) for v in wyniki.values()]
    assert len(set(slot1)) == len(slot1), f"hologram powinien rozróżnić fazy, dostał {slot1}"
    # amplitude=0.8 dla wszystkich trzech -> |slot1| <= 0.8 (część rzeczywista cosinusa)
    assert all(abs(x) <= 0.8 + 1e-9 for x in slot1)


def test_onehot_not_implemented_yet() -> None:
    atom = MorphAtom(identity="x", encoding=AtomEncoding.SYMBOLIC_ONEHOT)
    try:
        atom.feature_vector()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("SYMBOLIC_ONEHOT powinien być jeszcze NotImplementedError")


def test_slot_out_of_range_raises() -> None:
    bad = MorphAtom(
        identity="zle",
        base_features=[FeatureOverlay("x", amplitude=1.0, phase=0.0, slot=999)],
    )
    try:
        bad.feature_vector()
    except ValueError:
        pass
    else:
        raise AssertionError("slot poza FEATURE_DIM powinien rzucić ValueError")


# --- 2. LinguisticBubble: bez substratu (fallback) vs recursion n+1 ---------

def test_bubble_without_substrate_uses_sum_fallback() -> None:
    a = MorphAtom(identity="a", base_features=[FeatureOverlay("f", 1.0, 0.0, 0)])
    b = MorphAtom(identity="b", base_features=[FeatureOverlay("g", 1.0, 0.0, 1)])
    bub = LinguisticBubble(label="ab", members=[a, b])
    v = bub.as_atom_vector(dim=64)
    assert v[0] == 1.0 and v[1] == 1.0


class _FakeSubstrate:
    """Minimalny double zgodny z SubstrateProtocol — sprawdza że
    LinguisticBubble woła hrr_bind (nie sumę) gdy substrate jest podpięty."""

    def __init__(self) -> None:
        self.bind_calls = 0
        self.atoms: dict[str, dict] = {}
        self.bubbles: dict[str, list[str]] = {}

    def create_atom(self, id, S, E, T, **kwargs):
        self.atoms[id] = {"S": S, "E": E, "T": T, **kwargs}
        return id

    def get_atom(self, id):
        return self.atoms.get(id)

    def create_bubble(self, label, atom_ids=None):
        self.bubbles[label] = list(atom_ids or [])
        return label

    def hrr_bind(self, a, b):
        self.bind_calls += 1
        # placeholder non-commutative marker: concat-truncate, żeby test
        # odróżnił to od sumy elementwise (fallback).
        return [x + 1000.0 * y for x, y in zip(a, b)]

    def hrr_query(self, composite, probe):
        return 1.0


def test_bubble_with_substrate_calls_hrr_bind_not_sum() -> None:
    fake = _FakeSubstrate()
    a = MorphAtom(identity="a", base_features=[FeatureOverlay("f", 1.0, 0.0, 0)])
    b = MorphAtom(identity="b", base_features=[FeatureOverlay("g", 1.0, 0.0, 1)])
    bub = LinguisticBubble(label="ab", members=[a, b], substrate=fake)
    v = bub.as_atom_vector(dim=64)
    assert fake.bind_calls == 1
    # fallback sumy dałby v[1] == 1.0; hrr_bind fake daje 1000x-marker
    assert v[1] != 1.0


def test_recursion_n_plus_1_word_into_phrase() -> None:
    """Bubble wyższego poziomu (fraza) przyjmuje inny Bubble (słowo) jako
    członka — dokładnie rekurencja obiecana w docstringu modułu."""
    fake = _FakeSubstrate()
    rdzen = MorphAtom(identity="pisa", base_features=[FeatureOverlay("r", 1.0, 0.0, 0)])
    prefiks = MorphAtom(identity="na-", base_features=[FeatureOverlay("p", 0.8, 0.1, 1)])
    slowo = LinguisticBubble(label="napisać", members=[prefiks, rdzen], substrate=fake)
    inny_rdzen = MorphAtom(identity="list", base_features=[FeatureOverlay("r2", 1.0, 0.0, 2)])
    fraza = LinguisticBubble(label="napisać list", members=[slowo, inny_rdzen], substrate=fake)

    v = fraza.as_atom_vector(dim=64)
    assert len(v) == 64
    # dwa bindy: wewnątrz słowa + fraza<->rdzeń listu
    assert fake.bind_calls == 2


# --- 3. RealSubstrateAdapter / connect_real_substrate (łagodna degradacja) -

def test_connect_real_substrate_graceful_or_working() -> None:
    """Nie zakładamy, że KarmazynOs jest zainstalowany na maszynie CI —
    None jest poprawnym wynikiem. Jeśli substrat JEST dostępny (jak na tej
    maszynie deweloperskiej), sprawdzamy, że realny bind faktycznie
    zwraca wektor D o tej samej długości (nie NotImplementedError)."""
    adapter = connect_real_substrate()
    if adapter is None:
        return
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0, 0.0]
    out = adapter.hrr_bind(a, b)
    assert len(out) == len(a)
    q = adapter.hrr_query(out, out)
    assert q > 0.99, f"self-similarity powinna być ~1.0, dostał {q}"


def test_persist_roundtrip_when_substrate_available() -> None:
    adapter = connect_real_substrate()
    if adapter is None:
        return
    bubbles = przyklad_pisac_rodzina_bubbles(AtomEncoding.HOLOGRAM, substrate=adapter)
    label = bubbles[0].persist(dim=64)
    assert label == bubbles[0].label
    stored = adapter.get_atom(f"{bubbles[0].label}/{bubbles[0].members[0].identity}/0")
    assert stored is not None


# --- 4. Most bridge: to_phi_atoms -> kontrakt PhiJSON tego repo ------------

def test_to_phi_atoms_shape_and_recursion_depth() -> None:
    bubbles = przyklad_pisac_rodzina_bubbles(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(bubbles)

    ids = [a["id"] for a in atoms]
    assert len(ids) == len(set(ids)), "id atomów muszą być unikalne"
    assert {"corpus", "lingua_page"} <= set(ids)

    # page depth=0, bubble(słowo) depth=1, morfem depth=2
    by_id = {a["id"]: a for a in atoms}
    words = [a for a in atoms if a["S"].startswith("lingua:bubble:")]
    assert len(words) == 3
    assert all(w["depth"] == 1 for w in words)
    morphs = [a for a in atoms if a["S"].startswith("lingua:hologram:")]
    assert len(morphs) == 6  # 3 słowa x (prefiks + rdzeń)
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
    bubbles = przyklad_pisac_rodzina_bubbles(AtomEncoding.SYMBOLIC)
    atoms, edges, _meta = to_phi_atoms(bubbles)
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
    from phi_dump import load_dump, graph_to_content_tracer

    bubbles = przyklad_pisac_rodzina_bubbles(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(bubbles)

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
