"""
karmazyn_lingua.py (v4 — dwie poprawki znalezione przez testy na realnym stosie)
==================================================================================

Warstwa lingwistyczna NA WIERZCHU zamrożonego substratu KarmazynOS
(analogicznie do karmazyn_econ.py) — Store/Atom/HRR pozostają NIETKNIĘTE.

DWIE POPRAWKI względem v3 — obie znalezione przez faktyczne odpalenie kodu
na prawdziwym Store+Atom+HRR (nie przez inspekcję), więc opisuję je tu
z dowodem, nie tylko z wnioskiem:

  1. POZYCJA 0 JEST ZDEGENEROWANA. P^0 (z HRROperations.P_power) to
     wektor delta — element IDENTYCZNOŚCIOWY splotu: bind(P^0, x) = x.
     Sprawdzone empirycznie: unbind(bundle(bind(P^0,pref), bind(P^1,root)),
     P^0) zwraca CAŁĄ sumę BEZ ZMIAN (np.allclose = True), a nie "pref
     z szumem od root" jak przy każdej innej pozycji. Innymi słowy:
     unbind() na pozycji 0 nic nie separuje — po prostu oddaje bundle
     z powrotem. compose() w tym pliku dlatego NIGDY nie używa P^0:
     pozycje liczą się od 1 (enumerate(..., start=1)).
     To dotyczy też HRROperations.encode_sequence/decode_position z
     karmazyn_hrr.py — tam pozycja 0 ma tę samą właściwość, jeśli ktoś
     kiedyś będzie chciał unbindować pozycję 0 z sekwencji.

  2. OVERLAY 1D (TYLKO CZĘŚĆ RZECZYWISTA) JEST ŚLEPY NA FAZĘ POD COSINE
     SIMILARITY. v1-v3 kodowały cechę jako pojedynczą liczbę rzeczywistą
     A*cos(phi) w jednym slocie. Problem: cosine similarity jest niezmiennicza
     względem skali — dwa wektory różniące się TYLKO długością wzdłuż tej
     samej osi (a nie kierunkiem) mają similarity=1.0 między sobą. Jeśli
     wszystkie warianty cechy (np. na-/prze-/wy-) mają fazę w tej samej
     ćwiartce (0°-90°), ich rzuty na oś Re są tylko różnej DŁUGOŚCI, więc
     retrieval nie widzi między nimi żadnej różnicy — sprawdzone empirycznie:
     r@1 = 0.33 (poziom przypadku) dla WSZYSTKICH trybów, nawet hologram,
     dopóki slot miał tylko 1 wymiar. Poprawka: każdy slot cechy = 2 wymiary
     rzeczywiste (Re, Im) zamiast 1 — faza staje się prawdziwym KIERUNKIEM
     w płaszczyźnie 2D, nie tylko długością na jednej osi. FEATURE_DIM=64
     realnych wymiarów => N_SLOTS=32 kanałów cech (2 wymiary/slot).

WYNIK PO OBU POPRAWKACH (patrz _demo_retrieval, uruchomione naprawdę):
  - SYMBOLIC (Im zawsze 0, brak fazy): r@1 = 0.33 zawsze, niezależnie od
    separacji faz — bo dla symbolic te warianty SĄ identycznymi wektorami,
    to nie kwestia szumu, tylko reprezentacja nie ma gdzie zakodować różnicy.
  - HOLOGRAM, fazy stłoczone w 30°-60° (near/prze-/wy z pierwszej wersji
    demo): r@1 = 0.33 — reprezentacja MA zdolność rozróżniania (wartości
    similarity już się różnią), ale separacja faz jest za mała względem
    szumu z bind/bundle przy D=64.
  - HOLOGRAM, fazy rozstawione 0°/120°/240°: r@1 = 1.0 — pełna, poprawna
    separacja.
  Wniosek: "hologram MOŻE nieść informację o fazie" i "hologram FAKTYCZNIE
  poprawia retrieval w tym zadaniu" to dwa różne twierdzenia — drugie
  zależy od separacji wartości i wymiarowości, nie jest automatyczne.
  To jest dokładnie ten sam typ pytania, które P3/P4a zadały Holo-Phi.

Podział pracy (bez zmian koncepcyjnie):
  ATOM   = morfem/glif LUB złożone słowo/fraza (nierozróżnialne na poziomie
           substratu — obie to Atom, stąd rekurencja n+1). Wektor D=2048 =
           [overlay cech, FEATURE_DIM] + [content-wektor z nazwy, z Store].
  BUBBLE = grupowanie/ochrona GC (Store.create_bubble) — bez wektorów.
  compose() = n+1: bind(P^i, cechy_i) dla i=1,2,... (NIGDY i=0), bundle
           (suma), rejestracja jako nowy atom.

Most/PhiMost (ten repozytorium): `to_phi_atoms` (sekcja na dole) eksportuje
drzewo złożonych atomów LinguaLayer (np. rodzina słów z compose()) do
kanonu PhiJSON tego repo (te same klucze co holo_phi.build_holo_graph /
transcode.phi_build), więc wpina się w istniejący pipeline
`make_dump_*.py -> dumps/... -> phimost_learn.py` bez zmiany silnika Most.
CLI: `make_dump_lingua.py`. Chodzi po realnym Store rekurencyjnie przez
atom.value["members"] (ustawiane przez compose()), więc nie potrzebuje
osobnych klas drzewa jak stare MorphAtom/LinguisticBubble — sam Store
JEST drzewem.

SYMBOLIC_ONEHOT (twardy baseline): zaimplementowany. W przeciwieństwie do
SYMBOLIC (gdzie kolizja slotów jest CELOWA — pokazuje słabość baseline'u)
i HOLOGRAM (gdzie slot dzieli się na 2 wymiary Re/Im), ONEHOT adresuje
wymiar BEZPOŚREDNIO przez `slot` (0..FEATURE_DIM-1, pełne 64, nie N_SLOTS)
i wymaga, żeby wołający nadał każdej odrębnej wartości WŁASNY dedykowany
slot — wtedy kolizja jest strukturalnie niemożliwa (nie tylko unikana przez
konwencję). Sprawdzone empirycznie (_run_onehot_baseline): r@1 = 1.0 zawsze,
niezależnie od fazy (ONEHOT jej nie używa) — górna granica retrievalu na
tym pipeline, punkt odniesienia dla SYMBOLIC (r@1=0.33, słaby baseline) i
HOLOGRAM (r@1 zależny od separacji faz, patrz wyżej).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

try:
    import karmazyn_hrr as _hrr
    _HAS_HRR = True
except Exception:
    _hrr = None
    _HAS_HRR = False


FEATURE_DIM = 64                 # calkowita liczba wymiarow rzeczywistych na overlay cech
N_SLOTS = FEATURE_DIM // 2       # kazdy slot cechy = 2 wymiary rzeczywiste (Re, Im)
DEFAULT_VEC_DIM = 2048           # musi się zgadzać z VEC_DIM w karmazyn_substrate.py


class AtomEncoding(Enum):
    HOLOGRAM = "hologram"         # nakładanie A*e^(i*phi) na cechy — Re I Im
    SYMBOLIC = "symbolic"         # tylko Re=amplitude, Im=0 zawsze — słaby baseline
    SYMBOLIC_ONEHOT = "onehot"    # każdy odrębny morfem = własny dedykowany
                                   # wymiar (0..FEATURE_DIM-1) — twardy baseline,
                                   # wołający musi nadać różnym wartościom różne sloty


@dataclass
class FeatureOverlay:
    """Jedna nałożona cecha (diakrytyk, aspekt, przypadek...).
    amplitude/phase — parametry hologramu (A*e^(i*phi)).
    slot — kanał w obrębie N_SLOTS (0..N_SLOTS-1), mapowany na 2 wymiary
    rzeczywiste [2*slot, 2*slot+1] = (Re, Im). Różne WARTOŚCI tej samej
    kategorii cechy mogą celowo dzielić slot, żeby sprawdzić, czy faza
    (kierunek w 2D, nie długość) je rozróżnia."""
    name: str
    amplitude: float
    phase: float
    slot: int


def _check_slots(features: list[FeatureOverlay]) -> None:
    for f in features:
        if not (0 <= f.slot < N_SLOTS):
            raise ValueError(
                f"slot {f.slot} poza zakresem N_SLOTS={N_SLOTS} "
                f"(FEATURE_DIM={FEATURE_DIM}, 2 wymiary rzeczywiste na slot)"
            )


def _overlay_hologram(features: list[FeatureOverlay]) -> np.ndarray:
    """Re i Im jako OSOBNE wymiary — faza staje się kierunkiem w 2D,
    nie tylko długością na jednej osi (patrz poprawka 2 w nagłówku)."""
    _check_slots(features)
    out = np.zeros(FEATURE_DIM, dtype=float)
    for f in features:
        z = cmath.rect(f.amplitude, f.phase)      # A*e^(i*phi)
        i = 2 * f.slot
        out[i] += z.real
        out[i + 1] += z.imag
    return out


def _overlay_symbolic(features: list[FeatureOverlay]) -> np.ndarray:
    """Słaby baseline: tylko Re=amplitude, Im ZAWSZE 0 — różne fazy tej
    samej kategorii cechy stają się IDENTYCZNYMI wektorami (nie tylko
    trudnymi do odróżnienia — dosłownie tym samym punktem)."""
    _check_slots(features)
    out = np.zeros(FEATURE_DIM, dtype=float)
    for f in features:
        out[2 * f.slot] += f.amplitude
    return out


def _check_slots_onehot(features: list[FeatureOverlay]) -> None:
    for f in features:
        if not (0 <= f.slot < FEATURE_DIM):
            raise ValueError(
                f"slot {f.slot} poza zakresem FEATURE_DIM={FEATURE_DIM} "
                f"(SYMBOLIC_ONEHOT adresuje wymiary bezpośrednio — brak fazy, "
                f"więc brak podziału na Re/Im jak w hologram/symbolic)"
            )


def _overlay_onehot(features: list[FeatureOverlay]) -> np.ndarray:
    """Twardy baseline: `slot` adresuje wymiar BEZPOŚREDNIO (0..FEATURE_DIM-1,
    pełne 64 — nie N_SLOTS=32 jak hologram/symbolic, bo tu nie ma fazy do
    zmieszczenia obok amplitude). Kod wymusza tylko zakres, NIE unikalność
    między niezależnymi wywołaniami — to odpowiedzialność wołającego przy
    budowie zbioru ablacji: każda odrębna wartość musi dostać własny slot,
    inaczej to już nie jest 'twardy' baseline tylko SYMBOLIC pod inną nazwą
    (patrz _run_onehot_baseline dla poprawnego użycia)."""
    _check_slots_onehot(features)
    out = np.zeros(FEATURE_DIM, dtype=float)
    for f in features:
        out[f.slot] += f.amplitude
    return out


def feature_vector(features: list[FeatureOverlay], encoding: AtomEncoding) -> np.ndarray:
    if encoding is AtomEncoding.HOLOGRAM:
        return _overlay_hologram(features)
    if encoding is AtomEncoding.SYMBOLIC:
        return _overlay_symbolic(features)
    if encoding is AtomEncoding.SYMBOLIC_ONEHOT:
        return _overlay_onehot(features)
    raise NotImplementedError(f"{encoding} jeszcze niezaimplementowany")


def _splice(feature_vec: np.ndarray, content_vec: Optional[np.ndarray],
            dim: int) -> np.ndarray:
    """Skleja [overlay cech (FEATURE_DIM)] + [reszta: content_vec substratu,
    albo zera gdy brak HRR]."""
    if len(feature_vec) != FEATURE_DIM:
        raise ValueError("feature_vec ma zły rozmiar")
    if dim < FEATURE_DIM:
        raise ValueError(f"dim={dim} musi być >= FEATURE_DIM={FEATURE_DIM}")
    tail_len = dim - FEATURE_DIM
    if content_vec is None:
        tail = np.zeros(tail_len, dtype=float)
    else:
        content_vec = np.asarray(content_vec, dtype=float)
        tail = np.zeros(tail_len, dtype=float)
        take = min(tail_len, max(0, len(content_vec) - FEATURE_DIM))
        tail[:take] = content_vec[FEATURE_DIM:FEATURE_DIM + take]
    return np.concatenate([np.asarray(feature_vec, dtype=float), tail])


class LinguaLayer:
    """Warstwa lingwistyczna nad realnym Store (karmazyn_substrate.Store
    albo dowolnym magazynem zgodnym z AtomStore z karmazyn_atomstore.py).
    Woła metody z kontraktu: create_atom, get_atom, has_atom, create_bubble,
    opcjonalnie atom_vector/resonance jeśli Store je ma. Kompozycję liczy
    przez karmazyn_hrr.py (bind/bundle), jeśli jest importowalny — inaczej
    degraduje się do zwykłej sumy (fallback, patrz compose())."""

    def __init__(self, store, dim: int = DEFAULT_VEC_DIM):
        self.store = store
        self.dim = dim
        # Osobna, MAŁA instancja HRR (D=FEATURE_DIM) tylko do wiązania
        # pozycyjnego wewnątrz overlay cech — nie miesza się z "dużym"
        # D=2048 substratu (Store.atom_vector generuje TĘ część osobno).
        self._pos_hrr = _hrr.HRROperations(D=FEATURE_DIM) if _HAS_HRR else None

    # -- pomocnicze ---------------------------------------------------------
    def _content_vector(self, atom) -> Optional[np.ndarray]:
        if hasattr(self.store, "atom_vector"):
            return self.store.atom_vector(atom)
        return None

    def _get_or_create(self, aid: str, S: str, E: str, fv: np.ndarray,
                        value: Optional[Any] = None):
        """Get-or-create: ten sam morfem/atom użyty w wielu słowach jest
        TWORZONY RAZ i ponownie wykorzystywany (idempotentne)."""
        if self.store.has_atom(aid):
            return self.store.get_atom(aid)
        self.store.create_atom(aid, S=S, E=E, value=value)
        atom = self.store.get_atom(aid)
        content = self._content_vector(atom)
        atom.vector = _splice(fv, content, dim=self.dim)
        return atom

    # -- API publiczne --------------------------------------------------------
    def make_morpheme(self, ident: str, features: list[FeatureOverlay],
                       encoding: AtomEncoding = AtomEncoding.HOLOGRAM):
        """Poziom 0: atom morfemu / glifu / sylaby."""
        fv = feature_vector(features, encoding)
        return self._get_or_create(f"lingua:{ident}", S="morpheme", E=ident, fv=fv)

    def compose(self, label: str, member_atoms: Sequence[Any], *,
                kind: str = "composite", bubble_root: bool = False):
        """Poziom n+1: składa atomy W KOLEJNOŚCI podanej listy: pozycja
        i = bind(P^i, cechy_i), suma przez bundle — jak
        HRROperations.encode_sequence, na MOICH wektorach cech (nie na
        wektorach z samej nazwy).

        WAŻNE: pozycje liczą się OD 1, nigdy od 0 — P^0 jest elementem
        identycznościowym splotu (bind(P^0,x)=x), więc unbind(...,P^0)
        oddaje całą sumę bez separacji zamiast wyciągać składnik
        (sprawdzone empirycznie, patrz nagłówek pliku, poprawka 1).

        Bez HRR (karmazyn_hrr.py niedostępny): fallback = zwykła suma
        (nie rozróżnia pozycji/ról — ograniczenie jawne, nie ukryte)."""
        if not member_atoms:
            raise ValueError("compose() wymaga co najmniej jednego atomu składowego")

        feature_vecs = []
        for atom in member_atoms:
            v = atom.vector
            if v is None or len(v) < FEATURE_DIM:
                raise ValueError(
                    f"atom {getattr(atom, 'id', '?')!r} nie ma poprawnego "
                    f"wektora cech (oczekiwano >= {FEATURE_DIM} wymiarów) — "
                    f"czy powstał przez make_morpheme/compose z tej warstwy?"
                )
            feature_vecs.append(np.asarray(v[:FEATURE_DIM], dtype=float))

        if self._pos_hrr is not None:
            terms = [_hrr.bind(self._pos_hrr.P_power(i), fv)
                     for i, fv in enumerate(feature_vecs, start=1)]  # start=1, NIGDY 0
            combined_fv = _hrr.bundle(*terms)
        else:
            combined_fv = np.sum(feature_vecs, axis=0)  # fallback bez HRR

        aid = f"lingua:{label}"
        member_ids = [a.id for a in member_atoms]
        composite = self._get_or_create(
            aid, S=kind, E=label, fv=combined_fv,
            value={"members": member_ids, "kind": kind, "order": member_ids},
        )

        bubble_label = f"bubble:{label}"
        self.store.create_bubble(bubble_label, atom_ids=member_ids + [composite.id],
                                  root=bubble_root)
        return composite

    def unbind_position(self, composite, position: int) -> np.ndarray:
        """Odzyskaj przybliżony wektor cech z pozycji `position` złożenia.
        position numerowana OD 1, zgodnie z compose() (position=1 to
        pierwszy element listy member_atoms przekazanej do compose)."""
        if self._pos_hrr is None:
            raise NotImplementedError("brak karmazyn_hrr — unbind_position niedostępny")
        if position < 1:
            raise ValueError(
                "position musi być >= 1 — P^0 jest elementem identycznościowym "
                "i unbind(..., P^0) nie separuje niczego (patrz nagłówek pliku)"
            )
        fv = np.asarray(composite.vector[:FEATURE_DIM], dtype=float)
        return _hrr.unbind(fv, self._pos_hrr.P_power(position))

    def nearest_morpheme(self, query_fv: np.ndarray, candidates: Sequence[Any],
                          top_k: int = 1):
        """r@k: posortowana [(similarity, atom), ...] wśród kandydatów wg
        podobieństwa wektora cech (tylko FEATURE_DIM, nie cały atom.vector)."""
        if self._pos_hrr is None:
            raise NotImplementedError("brak karmazyn_hrr — nearest_morpheme niedostępny")
        scored = []
        for atom in candidates:
            cand_fv = np.asarray(atom.vector[:FEATURE_DIM], dtype=float)
            s = _hrr.similarity(query_fv, cand_fv)
            scored.append((s, atom))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def query(self, name_or_vector, k: int = 5, threshold: float = 0.1):
        """Cienki wrapper na Store.resonance() (przeszukuje CAŁY wektor
        D=2048, nie tylko overlay cech) — do testów w stylu Holo-Phi."""
        if not hasattr(self.store, "resonance"):
            raise NotImplementedError("Ten store nie ma resonance() (brak HRR)")
        return self.store.resonance(name_or_vector, k=k, threshold=threshold)


# ---------------------------------------------------------------------------
# Przykład: polska rodzina aspektowa jako n+1 (odpalalne od razu, na
# prawdziwym Store — zastępuje starą przyklad_pisac_rodzina_bubbles, która
# budowała drzewo z osobnych klas MorphAtom/LinguisticBubble zamiast na
# realnym substracie).
# ---------------------------------------------------------------------------

def przyklad_pisac_rodzina(
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM,
) -> tuple["LinguaLayer", list[Any]]:
    """pisać / napisać / przepisać / wypisać jako compose(prefiks, rdzeń)
    na realnym Store. Rdzeń zajmuje slot 0. Wszystkie trzy prefiksy CELOWO
    dzielą slot 1 i mają tę samą amplitude=0.8 (różni je tylko faza) —
    wymusza pytanie wprost: czy faza rozróżnia prefiksy, skoro amplitude
    nie może. Zwraca (lingua, [słowo_na, słowo_prze, słowo_wy]) — lingua
    trzeba zwrócić razem ze słowami, bo store/atomy żyją w niej."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)

    rdzen = lingua.make_morpheme(
        "pisa", [FeatureOverlay("rdzen_pisac", amplitude=1.0, phase=0.0, slot=0)], encoding
    )

    prefiksy = {
        "na-": FeatureOverlay("aspekt_na", amplitude=0.8, phase=math.pi / 4, slot=1),
        "prze-": FeatureOverlay("aspekt_prze", amplitude=0.8, phase=math.pi / 3, slot=1),
        "wy-": FeatureOverlay("aspekt_wy", amplitude=0.8, phase=math.pi / 6, slot=1),
    }

    slowa: list[Any] = []
    for nazwa, feat in prefiksy.items():
        prefiks = lingua.make_morpheme(nazwa, [feat], encoding)
        slowa.append(lingua.compose(f"{nazwa}pisać", [prefiks, rdzen], kind="word"))
    return lingua, slowa


# ---------------------------------------------------------------------------
# Most z tego repo: eksport do kanonu PhiJSON (holo_phi / transcode)
# ---------------------------------------------------------------------------
#
# Most (silnik uczący tego repo) NIE czyta z Store wprost — czyta PhiJSON
# (dumps/<name>/{atoms,edges}.jsonl + manifest.json, zob. phi_dump.py).
# `to_phi_atoms` to most między nimi: atomy w tych samych kluczach co
# holo_phi.build_holo_graph / transcode.phi_build.build_from_spans, więc
# działa z export_graph_as_dump / phi_dump.load_dump / phimost_learn.py bez
# zmian w silniku. `T` z energii cech (norma L2 overlay) na wspólnym `ref` —
# ta sama konwencja co holo_phi §2b (energy_to_T), nie hash.
#
# W przeciwieństwie do starej wersji (MorphAtom/LinguisticBubble jako
# osobne dataclassy) to chodzi PO REALNYM STORE: drzewo odczytujemy z
# atom.value["members"] (ustawianego przez LinguaLayer.compose) — Store
# sam JEST strukturą, nie trzeba jej duplikować w Pythonie.

from holo_phi import energy_to_T  # noqa: E402  (import na dole: unika cyklu przy `python -m holo_phi`)
from transcode.phi_build import state_for_T  # noqa: E402


def _atom_energy(atom) -> float:
    """Energia atomu = norma L2 jego overlayu cech (FEATURE_DIM pierwszych
    wymiarów) — analog sumy amplitud ze starej wersji, ale liczony wprost
    z wektora, bez trzymania osobno listy FeatureOverlay na atomie."""
    fv = np.asarray(atom.vector[:FEATURE_DIM], dtype=float)
    return float(np.linalg.norm(fv))


def _atom_children(atom) -> Optional[list[str]]:
    """Odczytuje drzewo złożenia zapisane przez compose(..., value={...}).
    Store NIE eksponuje tego jako atom.value (tego atrybutu po prostu nie
    ma na karmazyn_atom.Atom) — kwarg `value=` z create_atom ląduje pod
    atom.metadata['v'] (sprawdzone empirycznie na realnym Store)."""
    meta = getattr(atom, "metadata", None)
    val = meta.get("v") if isinstance(meta, dict) else None
    if isinstance(val, dict) and "members" in val:
        return list(val["members"])
    return None


def to_phi_atoms(
    lingua: LinguaLayer,
    top_atoms: Sequence[Any],
    *,
    page_id: str = "lingua_page",
    corpus_note: str = "karmazyn_lingua: morfem -HRR bind-> slowo (n+1)",
) -> tuple[list[dict], list[dict], dict]:
    """Las złożonych atomów LinguaLayer (np. rodzina słów z compose()) ->
    (atoms, edges, meta) w kanonie PhiJSON tego repo. Rekurencyjne po
    realnym Store: obsługuje dowolną głębię zagnieżdżenia compose()-w-
    compose() (morfem -> słowo -> fraza -> klauzula), nie tylko płaski
    przypadek z `przyklad_pisac_rodzina`.

    depth: page=0, top-level=1, każdy poziom zagnieżdżenia +1
    (MAX_DEPTH_BINS=3 w tracerze n_dim=6 — patrz transcode/phi_build.py;
    głębsze drzewa nadal dostają poprawny int `depth`, ale trzeci+ bin
    tracer spłaszcza razem, jak w reszcie repo)."""
    store = lingua.store
    atoms: list[dict] = []
    edges: list[dict] = []

    def bind_edge(a: str, b: str, name: str) -> None:
        edges.append({"kind": "bind", "src": a, "dst": b, "name": name})
        edges.append({"kind": "bind", "src": b, "dst": a, "name": name})

    atoms.append(
        {
            "id": "corpus", "S": "lingua/corpus", "E": corpus_note, "T": 1.0,
            "state": "HOT", "bubble": "lingua", "depth": 0, "unit": "page",
            "payload": corpus_note, "mazur_tracer": 0.5, "topic": "corpus",
        }
    )
    atoms.append(
        {
            "id": page_id, "S": "lingua/page", "E": f"page n_bubbles={len(top_atoms)}",
            "T": 0.92, "state": "HOT", "bubble": "lingua", "depth": 0, "unit": "page",
            "payload": None, "mazur_tracer": 0.5, "topic": "page",
        }
    )
    bind_edge("corpus", page_id, "has_page")

    energies = [_atom_energy(a) for a in top_atoms]
    ref = max(energies) if energies and max(energies) > 0 else 1.0

    def emit(atom, node_id: str, parent_id: str, depth: int, topic: str) -> None:
        children = _atom_children(atom)
        if children:
            unit, label = "bubble", atom.E
            S = f"lingua:bubble:{label}"
            E = f"bubble depth={depth} n_members={len(children)}"
        else:
            unit, label = "morph", atom.E
            S = f"lingua:{atom.S}:{label}"
            E = f"morph S={atom.S}"

        T = energy_to_T(_atom_energy(atom), ref=ref)
        st = state_for_T(T)
        atoms.append(
            {
                "id": node_id, "S": S, "E": E, "T": T, "state": st,
                "bubble": f"lingua/{topic}", "depth": depth, "unit": unit,
                "payload": (label if st == "HOT" else None),
                "mazur_tracer": max(0.0, min(1.0, _atom_energy(atom) / ref)),
                "topic": topic,
            }
        )
        edges.append({"kind": "parent", "src": node_id, "dst": parent_id, "name": f"{unit}_of"})
        bind_edge(parent_id, node_id, f"has_{unit}:{label}")

        if children:
            member_ids = []
            for i, cid in enumerate(children):
                child_atom = store.get_atom(cid)
                mid = f"{node_id}/m{i}"
                emit(child_atom, mid, node_id, depth + 1, topic)
                member_ids.append(mid)
            for a, b in zip(member_ids, member_ids[1:]):
                bind_edge(a, b, "next")

    top_ids: list[str] = []
    for i, atom in enumerate(top_atoms):
        bid = f"bubble/{i}/{atom.E}"
        emit(atom, bid, page_id, 1, atom.E)
        top_ids.append(bid)
    for a, b in zip(top_ids, top_ids[1:]):
        bind_edge(a, b, "next_bubble")

    meta = {
        "method": "karmazyn_lingua",
        "contract": "phijson-v1",
        "n_top_bubbles": len(top_atoms),
        "ref_energy": ref,
        "recursion": "morfem->slowo->fraza->klauzula (n+1)",
    }
    return atoms, edges, meta


# ---------------------------------------------------------------------------
# Demo / test — WYMAGA prawdziwych karmazyn_atom.py + karmazyn_substrate.py
# + karmazyn_hrr.py dla pełnej ścieżki bind/bundle/unbind.
# ---------------------------------------------------------------------------

def _run_r1(tryb: AtomEncoding, prefiksy_faza: dict) -> float:
    """Buduje na-/prze-/wy-pisać z podanymi fazami prefiksów, próbuje
    odzyskać, KTÓRY prefiks był na pozycji 1, zwraca r@1."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)

    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], tryb)
    prefiks_atoms = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], tryb)
        for n, ph in prefiksy_faza.items()
    }
    candidates = list(prefiks_atoms.values())

    trafienia = 0
    for nazwa, prefiks in prefiks_atoms.items():
        slowo = lingua.compose(f"{nazwa}pisać", [prefiks, rdzen], kind="word")
        odzyskany = lingua.unbind_position(slowo, position=1)   # prefiks = 1. element
        ranking = lingua.nearest_morpheme(odzyskany, candidates, top_k=3)
        top1 = ranking[0][1].E
        ok = (top1 == nazwa)
        trafienia += ok
        print(f"    {nazwa}pisać -> top1={top1!r} sim={ranking[0][0]:+.3f} "
              f"{'OK' if ok else 'BLAD'}")
    return trafienia / len(prefiksy_faza)


def _run_onehot_baseline() -> float:
    """Twardy baseline: w przeciwieństwie do _run_r1 (gdzie wszystkie
    prefiksy CELOWO dzielą slot=1), tu każdy prefiks dostaje WŁASNY slot —
    kolizja jest strukturalnie niemożliwa. Oczekiwane r@1=1.0 zawsze,
    niezależnie od fazy (ONEHOT jej nie używa) — górna granica retrievalu
    na tym pipeline, nie dowód przewagi żadnej reprezentacji."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)

    rdzen = lingua.make_morpheme(
        "pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], AtomEncoding.SYMBOLIC_ONEHOT
    )
    prefiksy_slot = {"na-": 1, "prze-": 2, "wy-": 3}
    prefiks_atoms = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, 0.0, slot)],
                                 AtomEncoding.SYMBOLIC_ONEHOT)
        for n, slot in prefiksy_slot.items()
    }
    candidates = list(prefiks_atoms.values())

    trafienia = 0
    for nazwa, prefiks in prefiks_atoms.items():
        slowo = lingua.compose(f"{nazwa}pisać_onehot", [prefiks, rdzen], kind="word")
        odzyskany = lingua.unbind_position(slowo, position=1)
        ranking = lingua.nearest_morpheme(odzyskany, candidates, top_k=3)
        top1 = ranking[0][1].E
        ok = (top1 == nazwa)
        trafienia += ok
        print(f"    {nazwa}pisać(onehot) -> top1={top1!r} sim={ranking[0][0]:+.3f} "
              f"{'OK' if ok else 'BLAD'}")
    return trafienia / len(prefiksy_slot)


def _demo_retrieval():
    print("=== SYMBOLIC (Im=0 zawsze — oczekiwane: r@1 = przypadek, ZAWSZE) ===")
    r_sym = _run_r1(AtomEncoding.SYMBOLIC,
                     {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6})
    print(f"  r@1 symbolic = {r_sym:.3f}\n")

    print("=== HOLOGRAM, fazy stłoczone 30°-45°-60° ===")
    r_holo_close = _run_r1(AtomEncoding.HOLOGRAM,
                            {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6})
    print(f"  r@1 hologram (fazy blisko) = {r_holo_close:.3f}\n")

    print("=== HOLOGRAM, fazy rozstawione 0°-120°-240° ===")
    r_holo_far = _run_r1(AtomEncoding.HOLOGRAM,
                          {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3})
    print(f"  r@1 hologram (fazy rozstawione) = {r_holo_far:.3f}\n")

    print("=== SYMBOLIC_ONEHOT (twardy baseline — każdy prefiks własny slot) ===")
    r_onehot = _run_onehot_baseline()
    print(f"  r@1 onehot = {r_onehot:.3f}\n")

    print("PODSUMOWANIE:", {
        "symbolic": r_sym,
        "hologram_faz_blisko": r_holo_close,
        "hologram_faz_rozstawione": r_holo_far,
        "onehot_hard_baseline": r_onehot,
    })

    print("\n=== to_phi_atoms smoke (rodzina pisać -> PhiJSON) ===")
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(lingua, slowa)
    print(f"  {len(atoms)} atoms, {len(edges)} edges, ref_energy={meta['ref_energy']:.3f}")


if __name__ == "__main__":
    _demo_retrieval()
