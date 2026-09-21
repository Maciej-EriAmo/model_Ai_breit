"""Polski alfabet (32 litery) — korpus jak Kana P1b gojūon: grupy → compose.

Bez q/v/x (obce). Metoda = ta sama co przyklad_hiragana_gojuon:
FeatureOverlay(grupa, pozycja) + compose członków grupy w „słowo”.
"""

from __future__ import annotations

from collections import defaultdict

from karmazyn_lingua import AtomEncoding, FeatureOverlay, LinguaLayer, PhaseRegistry

# Klasyczny alfabet polski (32) — bez q, v, x
POLISH_LETTERS = list("aąbcćdeęfghijklłmnńoóprsśtuwyzźż")
assert len(POLISH_LETTERS) == 32

# Grupy ~5 liter (jak rzędy gojūon) — struktura dla krawędzi next/bind
GROUP_SIZE = 5
POLISH_GROUPS: list[tuple[str, list[str]]] = []
for i in range(0, len(POLISH_LETTERS), GROUP_SIZE):
    chunk = POLISH_LETTERS[i : i + GROUP_SIZE]
    POLISH_GROUPS.append((f"g{i // GROUP_SIZE}", chunk))

GROUP_SLOT, POS_SLOT = 0, 1


def przyklad_polski_alfabet(
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM,
) -> tuple["LinguaLayer", list]:
    """32 litery PL, same compose-po-grupach co P1b hiragana."""
    import karmazyn_substrate as sub

    if encoding is AtomEncoding.SYMBOLIC_ONEHOT:
        raise NotImplementedError(
            "SYMBOLIC_ONEHOT nieobsługiwany — porównanie HOLOGRAM vs SYMBOLIC."
        )

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)
    n_groups = len(POLISH_GROUPS)
    max_pos = max(len(ch) for _, ch in POLISH_GROUPS)
    group_reg = PhaseRegistry(capacity=n_groups)
    pos_reg = PhaseRegistry(capacity=max_pos)

    by_group: dict[str, list] = defaultdict(list)
    for gname, letters in POLISH_GROUPS:
        for j, ch in enumerate(letters):
            feats = [
                FeatureOverlay(f"group:{gname}", 1.0, group_reg.phase_for(gname), GROUP_SLOT),
                FeatureOverlay(f"pos:{j}", 0.9, pos_reg.phase_for(str(j)), POS_SLOT),
            ]
            m = lingua.make_morpheme(f"pl:{ch}", feats, encoding)
            by_group[gname].append(m)

    slowa = [
        lingua.compose(f"group:{gname}", morphs, kind="word")
        for gname, morphs in by_group.items()
    ]
    return lingua, slowa
