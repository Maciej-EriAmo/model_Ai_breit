"""P0: A z częstości; T z energii na wspólnym ref; archive T stałe."""

from __future__ import annotations

from holo_phi import A_FLOOR, A_ROOT_BOOST, T_TOMB, build_holo_graph, amp_from_freq


def test_amp_monotonic_in_freq() -> None:
    assert amp_from_freq(1, 10) < amp_from_freq(5, 10) < amp_from_freq(10, 10)
    assert amp_from_freq(0, 10) == A_FLOOR
    assert amp_from_freq(1, 10, root_member=True) >= A_ROOT_BOOST


def test_build_A_from_freq_not_hash() -> None:
    atoms, _edges, meta = build_holo_graph()
    assert meta["A_def"].startswith("freq")
    # E zawiera A=; pola A/freq na znakach
    with_a = [a for a in atoms if a.get("depth") == 2 and "archive" not in a["id"] and "A" in a]
    assert with_a, "oczekiwano pola A na glyphach"
    # częstsze litery (np. E w ang. słowach lab) mają A >= rzadszych
    by_ch: dict[str, float] = {}
    for a in with_a:
        ch = a["S"].split(":")[-1]
        by_ch[ch] = max(by_ch.get(ch, 0.0), float(a["A"]))
    assert by_ch.get("E", 0) >= by_ch.get("W", 0)  # E częstsze w MOST/BRIDGE/... niż W


def test_shared_ref_and_archive_tomb() -> None:
    atoms, _e, meta = build_holo_graph()
    assert meta["ref_energy"] > 0
    arch = next(a for a in atoms if a["id"] == "holo/archive/X")
    assert arch["T"] == T_TOMB
    assert arch.get("A", 0) == 0.0


def test_focus_first_letter() -> None:
    atoms, _e, _m = build_holo_graph()
    # M w MOST — pierwsza litera
    m = next(a for a in atoms if a["id"].endswith("/ch/0/M") and "MOST" in a["id"])
    assert m["T"] >= 0.90


if __name__ == "__main__":
    test_amp_monotonic_in_freq()
    test_build_A_from_freq_not_hash()
    test_shared_ref_and_archive_tomb()
    test_focus_first_letter()
    print("OK test_holo_AT")
