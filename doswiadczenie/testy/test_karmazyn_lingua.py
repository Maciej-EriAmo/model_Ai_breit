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

import _labroot  # noqa: F401  — root labu na sys.path

import math

import numpy as np

from karmazyn_lingua import (
    FEATURE_DIM,
    AtomEncoding,
    FeatureOverlay,
    LinguaLayer,
    PhaseRegistry,
    Projector,
    feature_vector,
    glyph_features_from_unicode,
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


# --- 4. unbind_position(blind=True): ablacja korekty strukturalnej --------

def test_unbind_position_blind_loses_signal_for_close_phases() -> None:
    """blind=True pomija odejmowanie znanego wkładu sąsiadów (compose()
    zna ich dokładnie, patrz unbind_position) — dla faz stłoczonych
    30°-60° ta ablacja MUSI cofnąć r@1 do poziomu przypadku, dokładnie
    tak jak przed poprawką opisaną w LINGUA_PROBE_MLP.md (kontrast z
    test_unbind_position_subtracts_known_interference_close_phases, gdzie
    blind=False/domyślne daje r@1=1.0 na TYCH SAMYCH fazach)."""
    fazy = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    r1 = _r1_for_phases_blind(fazy, label="pisac_close_blind", blind=True)
    assert abs(r1 - 1 / 3) < 1e-9, f"blind na fazach bliskich powinien wrocic do przypadku, dostal {r1}"


def test_unbind_position_blind_keeps_signal_for_far_phases() -> None:
    """Dla faz rozstawionych 0°/120°/240° interferencja od wspólnego rdzenia
    jest i tak zdominowana przez separację faz — pominięcie korekty
    (blind=True) nie powinno nic popsuć, w przeciwieństwie do faz bliskich."""
    fazy = {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3}
    assert _r1_for_phases_blind(fazy, label="pisac_far_blind", blind=True) == 1.0


def _r1_for_phases_blind(fazy: dict[str, float], *, label: str, blind: bool) -> float:
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
        odzyskany = lingua.unbind_position(slowo, position=1, blind=blind)
        top1 = lingua.nearest_morpheme(odzyskany, candidates, top_k=1)[0][1]
        trafienia += top1.E == nazwa
    return trafienia / len(fazy)


# --- 5. Projector: skanowanie pozycji x trybu (oracle vs blind) ------------

def _pisac_close_phase_setup():
    """Trzy prefiksy o fazach stłoczonych 30°-60° (najtrudniejszy przypadek
    z sekcji 4) + rdzeń, jeszcze NIEZŁOŻONE — do budowania kompozytów per
    test, tak jak w _run_projector_scan/_run_observer_search."""
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiksy_faza = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    prefiks_atoms = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
        for n, ph in prefiksy_faza.items()
    }
    return lingua, rdzen, prefiks_atoms


def test_projector_scan_recovers_position_and_morpheme_without_prior_knowledge() -> None:
    """Projector.scan() NIE dostaje position=1 z góry (w przeciwieństwie do
    _r1_for_phases, który zawsze woła unbind_position(position=1)) — musi
    sam, przez skan pozycji 1 i 2 na puli kandydatów złożonej z prefiksów
    I rdzenia razem, trafić zarówno WŁAŚCIWĄ pozycję (1, bo prefiks jest
    pierwszy w compose([prefiks, rdzen])) jak i WŁAŚCIWY morfem."""
    lingua, rdzen, prefiks_atoms = _pisac_close_phase_setup()
    projector = Projector(lingua)
    candidates = list(prefiks_atoms.values()) + [rdzen]

    for nazwa, prefiks in prefiks_atoms.items():
        slowo = lingua.compose(f"{nazwa}pisać_scan", [prefiks, rdzen], kind="word")
        pos, blind, _sim, atom = projector.best(slowo, candidates, positions=(1, 2))
        assert pos == 1, f"projektor powinien sam trafic pozycje 1 dla {nazwa!r}, dostal {pos}"
        assert atom.E == nazwa, f"projektor powinien sam trafic morfem {nazwa!r}, dostal {atom.E!r}"
        assert blind is False, "scan() z domyslnym blind_modes=(False,) nie powinien zwracac blind=True"


def test_projector_best_matches_scan_top_result() -> None:
    """best() to udokumentowany skrót: musi być dokładnie scan()[0], nie
    osobną, niezależnie liczoną ścieżką."""
    lingua, rdzen, prefiks_atoms = _pisac_close_phase_setup()
    projector = Projector(lingua)
    candidates = list(prefiks_atoms.values()) + [rdzen]
    prefiks = prefiks_atoms["na-"]
    slowo = lingua.compose("na-pisać_best", [prefiks, rdzen], kind="word")

    scan_wyniki = projector.scan(slowo, candidates, positions=(1, 2))
    assert projector.best(slowo, candidates, positions=(1, 2)) == scan_wyniki[0]


def test_projector_best_returns_none_without_candidates() -> None:
    """Gdy nie ma czego dopasować (pusta lista kandydatów), best() musi
    zwrócić None zamiast rzucić wyjątkiem lub zwrócić fałszywy wynik."""
    lingua, rdzen, prefiks_atoms = _pisac_close_phase_setup()
    projector = Projector(lingua)
    slowo = lingua.compose("na-pisać_empty", [prefiks_atoms["na-"], rdzen], kind="word")
    assert projector.best(slowo, [], positions=(1, 2)) is None


def test_projector_scan_mixed_modes_observer_prefers_oracle_over_blind() -> None:
    """Z obiema opcjami na stole (blind_modes=(False, True)), obserwator
    (sortowanie scan() po similarity) sam wybiera oracle jako top wynik —
    bez żadnej podpowiedzi, bo oracle po prostu daje silniejszy sygnał na
    fazach stłoczonych (patrz test_unbind_position_blind_loses_signal_for_
    close_phases: blind tu pada do przypadku). Sprawdzamy też, że przepaść
    similarity między najlepszym widokiem oracle i najlepszym blind jest
    dodatnia i wyraźna — to sam w sobie sygnał wiarygodności, jaki dostaje
    obserwator bez znajomości ground truth."""
    lingua, rdzen, prefiks_atoms = _pisac_close_phase_setup()
    projector = Projector(lingua)
    candidates = list(prefiks_atoms.values())

    for nazwa, prefiks in prefiks_atoms.items():
        slowo = lingua.compose(f"{nazwa}pisać_mixed", [prefiks, rdzen], kind="word")
        wyniki = projector.scan(slowo, candidates, positions=(1, 2), blind_modes=(False, True))
        pos, blind, _sim, atom = wyniki[0]
        assert blind is False, f"obserwator powinien preferowac oracle dla {nazwa!r}, wybral blind={blind}"
        assert pos == 1 and atom.E == nazwa

        best_oracle = max((r for r in wyniki if not r[1]), key=lambda r: r[2])
        best_blind = max((r for r in wyniki if r[1]), key=lambda r: r[2])
        assert best_oracle[2] > best_blind[2], (
            f"przepasc similarity oracle-blind powinna byc dodatnia dla {nazwa!r}, "
            f"oracle={best_oracle[2]:.3f} blind={best_blind[2]:.3f}"
        )


# --- 6. glyph_features_from_unicode: ekstraktor cech dla nowego znaku -----

def test_glyph_extractor_atomic_char_has_only_base_feature() -> None:
    """Znak bez kanonicznej dekompozycji Unicode (np. 'k') nie ma znaków
    łączących do wykrycia — ekstraktor NIE ZGADUJE struktury, której
    Unicode nie opisuje, więc zwraca tylko cechę bazy, zero diakrytyku."""
    base_reg, mark_reg = PhaseRegistry(), PhaseRegistry()
    feats = glyph_features_from_unicode("k", base_reg, mark_reg)
    assert len(feats) == 1
    assert feats[0].slot == 0
    assert len(mark_reg) == 0


def test_glyph_extractor_decomposes_latin_diacritic() -> None:
    """'é' (nigdy ręcznie nie wpisane do żadnej tabeli w tym repo) musi
    rozłożyć się przez NFD na bazę 'e' (slot 0) + znak łączący U+0301
    COMBINING ACUTE ACCENT (slot 1) — dokładnie ta sama struktura co
    ręcznie kodowana hiragana z raportu, ale wykryta automatycznie."""
    base_reg, mark_reg = PhaseRegistry(), PhaseRegistry()
    feats = glyph_features_from_unicode("é", base_reg, mark_reg)
    assert [f.slot for f in feats] == [0, 1]
    assert feats[0].name == "base:e"
    assert "ACUTE" in feats[1].name


def test_glyph_extractor_decomposes_hiragana_dakuten() -> None:
    """Ten sam mechanizm (NFD) generalizuje na hiraganę z sekcji 10
    raportu: 'ば' = 'は' + COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK —
    dowód, że ekstraktor nie jest specyficzny dla alfabetu łacińskiego
    (cel z sekcji 1: jeden mechanizm, wiele pism)."""
    base_reg, mark_reg = PhaseRegistry(), PhaseRegistry()
    feats = glyph_features_from_unicode("ば", base_reg, mark_reg)
    assert feats[0].name == "base:は"
    assert "VOICED" in feats[1].name


def test_glyph_extractor_is_deterministic_across_calls() -> None:
    """Ten sam znak wywołany dwa razy na TYM SAMYM rejestrze musi dostać
    IDENTYCZNE fazy (rejestr pamięta, nie losuje na nowo) — analog
    test_get_or_create_idempotent, ale dla ekstraktora, nie Store."""
    base_reg, mark_reg = PhaseRegistry(), PhaseRegistry()
    f1 = glyph_features_from_unicode("é", base_reg, mark_reg)
    f2 = glyph_features_from_unicode("é", base_reg, mark_reg)
    assert [f.phase for f in f1] == [f.phase for f in f2]
    assert len(base_reg) == 1 and len(mark_reg) == 1


def test_phase_registry_capacity_exhausted_raises() -> None:
    """Pojemność jest rezerwowana Z GÓRY (lekcja 9B: rozstaw faz musi być
    stabilny, nie doliczany na bieżąco) — przekroczenie jej musi jawnie
    odmówić (ValueError), nie po cichu nadpisać/skolidować istniejącą
    fazę (patrz wzorzec _check_slots / _check_slots_onehot)."""
    reg = PhaseRegistry(capacity=2)
    reg.phase_for("a")
    reg.phase_for("b")
    try:
        reg.phase_for("c")
    except ValueError:
        pass
    else:
        raise AssertionError("trzeci klucz przy capacity=2 powinien rzucić ValueError")


def test_glyph_extractor_retrieval_recovers_base_and_mark_axes() -> None:
    """Test integracyjny na PRAWDZIWYCH, nigdy ręcznie niewpisanych do tabeli
    znakach: 4 bazy (a,e,o,u) x 3 akcenty (´,`,^) = 12 prawdziwych znaków
    łacińskich z narodowych alfabetów. Ekstraktor sam odkrywa strukturę
    (NFD), rejestr sam przydziela rozstawione fazy — jedyna ręczna wiedza
    to LISTA znaków do przetestowania, nie ich struktura wewnętrzna.

    Metodologia z sekcji 10.2 raportu (dowód 'template ma zera wszędzie
    poza swoim slotem'): dwie ROZŁĄCZNE pule czystych kandydatów (same
    bazy, same akcenty), każdy złożony znak odpytywany OSOBNO względem
    obu osi — jeśli overlay faktycznie się nie miesza między slotami,
    r@1=1.0 na obu osiach dla wszystkich 12 znaków."""
    baza_akcent = {
        "á": ("a", "´"), "à": ("a", "`"), "â": ("a", "^"),
        "é": ("e", "´"), "è": ("e", "`"), "ê": ("e", "^"),
        "ó": ("o", "´"), "ò": ("o", "`"), "ô": ("o", "^"),
        "ú": ("u", "´"), "ù": ("u", "`"), "û": ("u", "^"),
    }
    base_reg, mark_reg = PhaseRegistry(capacity=8), PhaseRegistry(capacity=8)
    lingua = _fresh_lingua()

    # pule czystych kandydatów per oś, zbudowane z TYCH SAMYCH rejestrów
    base_candidates = {
        b: lingua.make_morpheme(f"purebase:{b}",
                                 [FeatureOverlay(f"base:{b}", 1.0, base_reg.phase_for(b), slot=0)],
                                 AtomEncoding.HOLOGRAM)
        for b in {b for b, _ in baza_akcent.values()}
    }
    znak_do_kodu_akcentu = {"´": "́", "`": "̀", "^": "̂"}
    mark_candidates = {
        a: lingua.make_morpheme(f"puremark:{a}",
                                 [FeatureOverlay(f"mark:{a}", 0.8,
                                                  mark_reg.phase_for(znak_do_kodu_akcentu[a]), slot=1)],
                                 AtomEncoding.HOLOGRAM)
        for a in {a for _, a in baza_akcent.values()}
    }

    trafienia_base = trafienia_mark = 0
    for znak, (oczek_base, oczek_mark) in baza_akcent.items():
        feats = glyph_features_from_unicode(znak, base_reg, mark_reg)
        atom = lingua.make_morpheme(f"glyph:{znak}", feats, AtomEncoding.HOLOGRAM)
        query = np.asarray(atom.vector[:FEATURE_DIM], dtype=float)

        top1_base = lingua.nearest_morpheme(query, list(base_candidates.values()), top_k=1)[0][1]
        top1_mark = lingua.nearest_morpheme(query, list(mark_candidates.values()), top_k=1)[0][1]
        trafienia_base += (top1_base.E == f"purebase:{oczek_base}")
        trafienia_mark += (top1_mark.E == f"puremark:{oczek_mark}")

    n = len(baza_akcent)
    assert trafienia_base == n, f"baza: r@1={trafienia_base/n:.3f} (oczekiwano 1.0)"
    assert trafienia_mark == n, f"akcent: r@1={trafienia_mark/n:.3f} (oczekiwano 1.0)"


# --- 7. observe_with_confidence: ruchomy obserwator (jitter -> stabilność) -

def test_observe_with_confidence_stable_for_reliable_retrieval() -> None:
    """Fazy rozstawione 0°/120°/240° (margines decyzji duży, patrz sekcja 5
    raportu) — dopasowanie musi przeżyć jitter bez zmiany zdania:
    confidence=1.0 dla wszystkich trzech słów, obserwator zawsze trafia
    właściwy prefiks nawet pod poruszeniem punktu widzenia."""
    fazy = {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3}
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiksy = {n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
                for n, ph in fazy.items()}
    candidates = list(prefiksy.values())
    rng = np.random.default_rng(42)

    for nazwa, prefiks in prefiksy.items():
        slowo = lingua.compose(f"{nazwa}far_conf", [prefiks, rdzen], kind="word")
        fv = lingua.unbind_position(slowo, position=1)
        wynik = lingua.observe_with_confidence(fv, candidates, n_jitter=30, jitter_fraction=0.5, rng=rng)
        assert wynik["winner"].E == nazwa
        assert wynik["confidence"] == 1.0, f"{nazwa}: confidence={wynik['confidence']}"


def test_observe_with_confidence_unstable_for_ambiguous_retrieval() -> None:
    """Fazy stłoczone 30°-60° + blind=True (ta sama degeneracja co
    test_unbind_position_blind_loses_signal_for_close_phases, r@1=0.333) —
    margines decyzji jest mały, więc jitter MUSI odsłonić niepewność:
    confidence wyraźnie poniżej 1.0, i różna dla różnych słów (w
    przeciwieństwie do sztywnego 1.0/1.0/1.0 przypadku stabilnego) —
    obserwator wykrywa niepewność BEZ znajomości poprawnej odpowiedzi."""
    fazy = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiksy = {n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
                for n, ph in fazy.items()}
    candidates = list(prefiksy.values())
    rng = np.random.default_rng(42)

    confidences = []
    for nazwa, prefiks in prefiksy.items():
        slowo = lingua.compose(f"{nazwa}close_conf", [prefiks, rdzen], kind="word")
        fv = lingua.unbind_position(slowo, position=1, blind=True)
        wynik = lingua.observe_with_confidence(fv, candidates, n_jitter=30, jitter_fraction=0.5, rng=rng)
        confidences.append(wynik["confidence"])

    assert all(c < 1.0 for c in confidences), f"blind na fazach bliskich powinien być niepewny, dostał {confidences}"
    assert len(set(confidences)) > 1, "trzy różne słowa nie powinny dać identycznej (sztywnej) pewności"
    assert min(confidences) < 0.6, f"co najmniej jedno słowo powinno być WYRAŹNIE niepewne, dostał {confidences}"


def test_observe_with_confidence_empty_candidates_raises() -> None:
    lingua = _fresh_lingua()
    a = lingua.make_morpheme("x", [FeatureOverlay("f", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    try:
        lingua.observe_with_confidence(np.asarray(a.vector[:FEATURE_DIM], dtype=float), [])
    except ValueError:
        pass
    else:
        raise AssertionError("pusta lista kandydatów powinna rzucić ValueError")


def test_observe_with_confidence_votes_sum_to_n_jitter() -> None:
    """Sanity kontraktu: suma głosów w wyniku musi być dokładnie n_jitter —
    każda próba jitterowana głosuje raz, ani więcej, ani mniej."""
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    fv = np.asarray(slowa[0].vector[:FEATURE_DIM], dtype=float)
    wynik = lingua.observe_with_confidence(fv, slowa, n_jitter=17, rng=np.random.default_rng(1))
    assert sum(wynik["votes"].values()) == 17


# --- 8. yōon: sekwencyjny compose() na PRAWDZIWYM piśmie (nie syntetyce) --

_YOON_ROWS = {
    "k": "き", "s": "し", "t": "ち", "n": "に", "h": "ひ", "m": "み", "r": "り",
    "g": "ぎ", "z": "じ", "b": "び", "p": "ぴ",
}  # 11 wierszy spółgłoskowych biorących udział w jōon (prawdziwa hiragana)
_YOON_SMALLS = {"ya": "ゃ", "yu": "ゅ", "yo": "ょ"}  # małe kana ya/yu/yo


def test_yoon_sequential_compose_recovers_row_and_small_on_real_script() -> None:
    """Pierwszy test SEKWENCYJNEGO compose() (bind pozycyjny dwóch REALNYCH
    morfemów, nie flat overlay jak hiragana z sekcji 10 raportu) na
    prawdziwym piśmie: 33 prawdziwe kombinacje jōon (11 wierszy x
    ya/yu/yo) — standardowy, weryfikowalny zestaw japońskiego sylabariusza,
    żadna kombinacja nie jest zmyślona. Każda kombinacja = compose(wielka
    kana, mała kana), pozycja 1=wiersz, pozycja 2=ya/yu/yo.

    Rozstrzyga pytanie otwarte z sekcji 9A/11.3 raportu: sweep pojemności
    na danych SYNTETYCZNYCH przeskoczył wprost z N=10 (r@1=1.0) do N=12
    (r@1=0.667), nigdy nie sprawdzając N=11 — a to WŁAŚNIE liczba wierszy
    biorących udział w prawdziwym jōon. Zweryfikowane empirycznie przed
    napisaniem tego testu: N=11 na wspólnym slocie w sekwencyjnym compose()
    działa IDEALNIE (r@1=1.0) — granica pojemności leży gdzieś między 11 a
    12, nie między 10 a 11."""
    lingua = _fresh_lingua()
    big_reg = PhaseRegistry(capacity=len(_YOON_ROWS))
    small_reg = PhaseRegistry(capacity=len(_YOON_SMALLS))

    big_atoms = {
        row: lingua.make_morpheme(f"big:{kana}",
                                   [FeatureOverlay(f"row_{row}", 1.0, big_reg.phase_for(kana), slot=0)],
                                   AtomEncoding.HOLOGRAM)
        for row, kana in _YOON_ROWS.items()
    }
    small_atoms = {
        s: lingua.make_morpheme(f"small:{kana}",
                                 [FeatureOverlay(f"s_{s}", 0.8, small_reg.phase_for(kana), slot=1)],
                                 AtomEncoding.HOLOGRAM)
        for s, kana in _YOON_SMALLS.items()
    }
    big_candidates = list(big_atoms.values())
    small_candidates = list(small_atoms.values())

    trafienia_row = trafienia_small = 0
    n = 0
    for row, kana in _YOON_ROWS.items():
        for s, small_kana in _YOON_SMALLS.items():
            n += 1
            combo = lingua.compose(f"yoon:{kana}{small_kana}", [big_atoms[row], small_atoms[s]], kind="word")
            top1_row = lingua.nearest_morpheme(
                lingua.unbind_position(combo, position=1), big_candidates, top_k=1)[0][1]
            top1_small = lingua.nearest_morpheme(
                lingua.unbind_position(combo, position=2), small_candidates, top_k=1)[0][1]
            trafienia_row += (top1_row.E == f"big:{kana}")
            trafienia_small += (top1_small.E == f"small:{small_kana}")

    assert n == 33, f"standardowy jōon to 11x3=33 kombinacji, policzono {n}"
    assert trafienia_row == n, f"wiersz: r@1={trafienia_row/n:.3f} (oczekiwano 1.0)"
    assert trafienia_small == n, f"mały kana: r@1={trafienia_small/n:.3f} (oczekiwano 1.0)"


# --- 9. trwałość atomów w czasie: reach-GC / tick() ------------------------

def _ticks_to_cross_tomb(sub_mod) -> int:
    """Ile ticków potrzeba, żeby T_INIT ostygł poniżej T_TOMB przy
    multiplikatywnym rozpadzie T *= DECAY co tick (patrz Store._tick_body) —
    liczone z PRAWDZIWYCH stałych substratu, nie zgadywane na sztywno, żeby
    test nie posypał się przy zmianie stałych termicznych silnika."""
    n = math.ceil(math.log(sub_mod.T_TOMB / sub_mod.T_INIT) / math.log(sub_mod.DECAY))
    return n + 1  # margines: GC działa W TYM SAMYM ticku, w którym atom pada is_dead()


def test_unrooted_composite_is_reaped_by_reach_gc_when_cold() -> None:
    """compose() domyślnie NIE chroni złożenia (bubble_root=False), a
    Store() startuje bez żadnych korzeni (store.roots=[]) — więc KAŻDY
    atom jest 'nieosiągalny' od pierwszego ticku. Gdy ostygnie poniżej
    T_TOMB, reach-GC go usuwa (has_atom -> False) — dokładnie obawa z
    sekcji 11.3 raportu ('możliwe, że atom ostygnie i zostanie zebrany,
    zanim ktoś spróbuje go odzyskać'), teraz zweryfikowana na prawdziwym
    Store.tick(), nie tylko przypuszczona."""
    import karmazyn_substrate as sub
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiks = lingua.make_morpheme("na-", [FeatureOverlay("a_na", 0.8, math.pi / 4, 1)], AtomEncoding.HOLOGRAM)
    slowo = lingua.compose("na-pisac_gc", [prefiks, rdzen], kind="word")
    wid = slowo.id

    for _ in range(_ticks_to_cross_tomb(sub)):
        lingua.store.tick()

    assert not lingua.store.has_atom(wid), "niezakorzeniony atom powinien zostać zebrany po ostygnięciu"


def test_rooted_composite_survives_gc_and_stays_retrievable() -> None:
    """compose(..., bubble_root=True) chroni złożenie — bąbel trafia do
    store.roots, więc atomy są w REACH: zamiast reapowania dostają
    retencję TOMB. has_atom() zostaje True, i unbind_position/
    nearest_morpheme dalej działają poprawnie NAWET po ostygnięciu —
    prawdziwa trwałość, nie tylko 'jeszcze nie zniknął'."""
    import karmazyn_substrate as sub
    fazy = {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3}
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiksy = {n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
                for n, ph in fazy.items()}
    candidates = list(prefiksy.values())
    slowa = {n: lingua.compose(f"{n}pisac_rooted", [p, rdzen], kind="word", bubble_root=True)
             for n, p in prefiksy.items()}

    for _ in range(_ticks_to_cross_tomb(sub)):
        lingua.store.tick()

    for n, slowo in slowa.items():
        assert lingua.store.has_atom(slowo.id), f"{n}: zakorzeniony atom nie powinien zostać zebrany"
        assert lingua.store.get_atom(slowo.id).state == "TOMB"
        top1 = lingua.nearest_morpheme(lingua.unbind_position(slowo, position=1), candidates, top_k=1)[0][1]
        assert top1.E == n, f"{n}: retrieval powinien działać też po ostygnięciu do TOMB, dostał {top1.E!r}"


def test_unrooted_family_raises_when_oracle_siblings_were_reaped() -> None:
    """Regresja na poprawkę 2 w unbind_position (naprawiona po tym, jak ten
    test — w swojej wcześniejszej wersji — ZNALAZŁ przez uruchomienie, nie
    czytanie kodu, że korekta 'oracle' cicho degradowała się do poziomu
    blind, gdy reach-GC zebrał sąsiadów: wołający prosił o blind=False,
    dostawał wynik jakości blind=True bez żadnego ostrzeżenia). Teraz
    unbind_position(blind=False) na złożeniu, którego sąsiad zniknął ze
    Store, musi jawnie odmówić (RuntimeError) zamiast cicho zwrócić
    mylnie pewny siebie, zdegradowany wynik — a blind=True (który nigdy
    nie sięga po sąsiadów) musi dalej działać bez przeszkód."""
    import karmazyn_substrate as sub
    fazy = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    lingua = _fresh_lingua()
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.HOLOGRAM)
    prefiksy = {n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
                for n, ph in fazy.items()}
    slowa = {n: lingua.compose(f"{n}pisac_gc_raise", [p, rdzen], kind="word")  # bubble_root=False domyślne
             for n, p in prefiksy.items()}

    trafienia_przed = sum(
        lingua.nearest_morpheme(lingua.unbind_position(s, position=1),
                                 list(prefiksy.values()), top_k=1)[0][1].E == n
        for n, s in slowa.items()
    )
    assert trafienia_przed == 3, "przed GC oracle powinien działać idealnie (r@1=1.0), jak w sekcji 5 raportu"

    for _ in range(_ticks_to_cross_tomb(sub)):
        lingua.store.tick()
    assert not lingua.store.has_atom(rdzen.id), "test zakłada, że sąsiedzi FAKTYCZNIE zostali zebrani"

    for n, slowo in slowa.items():
        try:
            lingua.unbind_position(slowo, position=1, blind=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"{n}: unbind_position(blind=False) powinien rzucić RuntimeError, "
                f"gdy sąsiad zniknął ze Store, nie cicho zdegradować wynik"
            )
        # blind=True nigdy nie sięga po sąsiadów — musi działać bez przeszkód
        fv_blind = lingua.unbind_position(slowo, position=1, blind=True)
        assert fv_blind.shape == (FEATURE_DIM,)


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
