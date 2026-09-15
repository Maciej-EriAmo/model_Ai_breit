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

WYNIK (patrz _demo_retrieval, uruchomione naprawdę; historia poprawek
w doswiadczenie/glify/LINGUA_PROBE_MLP.md — zostawiona, bo pokazuje JAK
się do tego doszło, nie tylko dokąd):
  - SYMBOLIC (Im zawsze 0, brak fazy): r@1 = 0.33 zawsze, niezależnie od
    separacji faz ANI metody dekodowania (sprawdzone też trenowanym MLP,
    mean_acc=0.35) — bo dla symbolic te warianty SĄ identycznymi
    wektorami. To jest twardy sufit informacyjny (informacja zniszczona
    na etapie kodowania), nie problem dekodera.
  - HOLOGRAM, fazy stłoczone w 30°-60°: r@1 = 1.0. Po drodze: pierwsza
    wersja unbind_position() (goły unbind + nearest_morpheme do
    prototypu) dawała tu r@1=0.33 — wyglądało jak "sygnał za słaby
    względem szumu". Trenowany MLP (lingua_probe_mlp.py) pokazał, że to
    nieprawda: dostawał 0.72, więc sygnał BYŁ tam, tylko źle dekodowany.
    Przyczyna: unbind_position dekodował SUROWY composite.vector, co
    miesza w wynik interferencję od KAŻDEGO innego bound-termu (tu:
    wspólnego rdzenia). Poprawka: compose() zna wszystkich swoich
    sąsiadów (metadata['v']['members']), więc unbind_position ODEJMUJE
    NAJPIERW ich dokładny, znany wkład (bind/bundle są liniowe, więc to
    ścisłe kasowanie, nie przybliżenie) — r@1 wraca do 1.0, LEPIEJ niż
    trenowany MLP, bo znajomość struktury bija uczenie z zaszumionych
    próbek. Patrz LinguaLayer.unbind_position i _composite_member_ids.
  - HOLOGRAM, fazy rozstawione 0°/120°/240°: r@1 = 1.0 (bez zmian —
    interferencja z rdzenia była tu i tak zdominowana przez separację faz).
  Wniosek: "hologram MOŻE nieść informację o fazie" i "hologram FAKTYCZNIE
  poprawia retrieval w tym zadaniu" to dwa różne twierdzenia, ale trzecie,
  ukryte w środku, to "dekoder MOŻE gubić informację, którą reprezentacja
  faktycznie niesie" — r@1 mierzy metodę dekodowania tak samo jak samą
  reprezentację, i łatwo pomylić jedno z drugim. To jest dokładnie ten sam
  typ pytania, które P3/P4a zadały Holo-Phi.

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
atom.metadata['v']['members'] (ustawiane przez compose(), patrz
_composite_member_ids), więc nie potrzebuje osobnych klas drzewa jak
stare MorphAtom/LinguisticBubble — sam Store JEST drzewem.

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
import unicodedata
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


def _composite_member_ids(atom) -> Optional[list[str]]:
    """Odczytuje listę id składników złożenia zapisaną przez
    compose(..., value={...}). Store NIE eksponuje tego jako atom.value
    (tego atrybutu po prostu nie ma na karmazyn_atom.Atom) — kwarg
    `value=` z create_atom ląduje pod atom.metadata['v'] (sprawdzone
    empirycznie na realnym Store). Współdzielone przez unbind_position()
    (odejmowanie znanych interferentów) i to_phi_atoms() (chodzenie po
    drzewie), żeby nie duplikować tej samej wiedzy o kontrakcie Store."""
    meta = getattr(atom, "metadata", None)
    val = meta.get("v") if isinstance(meta, dict) else None
    if isinstance(val, dict) and "members" in val:
        return list(val["members"])
    return None


class PhaseRegistry:
    """Przypisuje NOWYM, nieznanym wcześniej kluczom (np. kodpunktowi
    znaku łączącego Unicode) stabilną fazę na okręgu, z góry ustalonej
    pojemności `capacity`, w kolejności pierwszego zgłoszenia — lekcja
    z sekcji 9B raportu hologram-znaki: fazy MUSZĄ być świadomie
    rozstawione (2π/N), nie losowe, a rozstaw musi być STABILNY (nie
    przeliczany na nowo przy każdym nowym kluczu, bo to zmieniłoby fazy
    już zakodowanych atomów pod nimi — capacity jest więc rezerwowana z
    góry, nie dociągana na bieżąco)."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._phase_of: dict[Any, float] = {}

    def phase_for(self, key: Any) -> float:
        if key not in self._phase_of:
            if len(self._phase_of) >= self.capacity:
                raise ValueError(
                    f"PhaseRegistry capacity={self.capacity} wyczerpana przy "
                    f"próbie zarejestrowania {key!r} — zwiększ capacity świadomie "
                    f"(patrz limit pojemności w sekcji 9A raportu), nie zgaduj"
                )
            self._phase_of[key] = 2 * math.pi * len(self._phase_of) / self.capacity
        return self._phase_of[key]

    def __len__(self) -> int:
        return len(self._phase_of)


def glyph_features_from_unicode(
    char: str,
    base_registry: PhaseRegistry,
    mark_registry: PhaseRegistry,
    *,
    base_amplitude: float = 1.0,
    mark_amplitude: float = 0.8,
) -> list[FeatureOverlay]:
    """Ekstraktor cech dla NIEZNANEGO WCZEŚNIEJ znaku — brakujący
    komponent z sekcji 11.3 raportu hologram-znaki ("rozpoznawanie
    rzeczywistego, nieuprzednio zakodowanego znaku"). Zamiast ręcznej
    tabeli znak<->wiersz<->diakrytyk (jak hiragana w raporcie, gdzie
    przypisanie było "z wiedzy ugruntowanej", czyli ręcznie wpisane),
    używa KANONICZNEJ DEKOMPOZYCJI UNICODE (NFD): dowolny znak złożony
    rozkłada się na BAZĘ + 0..N znaków łączących — to ten sam mechanizm,
    którym Unicode już opisuje np. hiraganę z dakuten (ば = は +
    U+3099 COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK, dokładnie ta
    sama struktura co test bazowy/diakrytyk z sekcji 10 raportu) i
    litery narodowe alfabetu łacińskiego (é = e + U+0301 COMBINING
    ACUTE ACCENT) — jeden realny mechanizm, wiele pism, zgodnie z celem
    z sekcji 1. Generalizuje na znak, którego nikt nie wpisał ręcznie do
    tabeli, bo rozpoznaje jego CZĘŚCI SKŁADOWE, nie sam znak.

    Ograniczenie jawne, nie ukryte: znak bez kanonicznej dekompozycji
    (np. samo 'h', albo kanji bez dakuten) dostaje TYLKO cechę bazy,
    zero cech diakrytyku — ekstraktor nie zgaduje struktury, której
    Unicode nie opisuje. Diakrytyk / baza nigdy wcześniej niewidziane w
    danym rejestrze dostają nową fazę PRZY PIERWSZYM użyciu (rejestr
    rośnie), ale to oznacza, że bez choć jednego WCZEŚNIEJSZEGO
    przykładu z TYM SAMYM diakrytykiem/bazą nie ma z czym go odróżnić
    przy retrievalu — ekstraktor generalizuje na nowe KOMBINACJE znanych
    części, nie na całkowicie pierwszy-raz-widziane części."""
    norm = unicodedata.normalize("NFD", char)
    if not norm:
        raise ValueError(f"pusty znak: {char!r}")
    base_char, marks = norm[0], norm[1:]

    features = [FeatureOverlay(f"base:{base_char}", base_amplitude,
                                base_registry.phase_for(base_char), slot=0)]
    for i, mark in enumerate(marks, start=1):
        if i >= N_SLOTS:
            raise ValueError(
                f"znak {char!r} ma {len(marks)} znaków łączących — więcej niż "
                f"dostępnych slotów (N_SLOTS={N_SLOTS}) po odjęciu slotu bazy"
            )
        features.append(FeatureOverlay(f"mark:{unicodedata.name(mark, mark)}",
                                        mark_amplitude, mark_registry.phase_for(mark),
                                        slot=i))
    return features


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

    def unbind_position(self, composite, position: int, blind: bool = False) -> np.ndarray:
        """Odzyskaj wektor cech z pozycji `position` złożenia. position
        numerowana OD 1, zgodnie z compose() (position=1 to pierwszy
        element listy member_atoms przekazanej do compose).

        POPRAWKA (znaleziona przez lingua_probe_mlp.py — trenowany MLP
        dostawał r@1=0.72 tam, gdzie goły unbind+nearest_morpheme dostawał
        0.33 dla tych samych danych, czyli sygnał BYŁ odzyskiwalny, tylko
        nie tą metodą): compose() zna WSZYSTKICH swoich sąsiadów (zapisani
        w atom.metadata['v']['members'], patrz _composite_member_ids) —
        zamiast unbindować surowy composite.vector wprost (co miesza w
        wynik interferencję od KAŻDEGO innego bound-termu), odejmujemy
        NAJPIERW dokładny, znany wkład każdego sąsiada:
        bind(P^i, sąsiad_i.vector) dla i != position. Bind/bundle są
        liniowe, więc to dokładnie kasuje interferencję międzyterminową
        (nie jest to przybliżenie) — zostaje tylko "samo-szum" z tego, że
        P^position nie jest idealnie unitarny (stąd sim≈0.707, nie 1.0,
        nawet przy D=2048, patrz karmazyn_hrr.py). Empirycznie: r@1 dla
        hologram_close (fazy 30°-60°) skacze z 0.33 (bez tej poprawki) do
        1.0 (z nią) — patrz doswiadczenie/glify/LINGUA_PROBE_MLP.md.

        `blind=True` WYŁĄCZA tę poprawkę: nawet gdy `composite` MA
        metadane members (czyli oracle byłby dostępny), pomija odejmowanie
        znanych sąsiadów i unbinduje surowy composite.vector wprost — tak
        jak w pierwszej wersji tej funkcji, sprzed poprawki. Cel: kontrolowana
        ablacja (ten sam composite, oracle vs blind) do sprawdzenia, czy
        retrieval przeżywa BEZ znajomości struktury złożenia z góry (pytanie
        z NASTĘPNY KROK w notatkach projektu — Projector.scan zgaduje
        POZYCJĘ, ale nadal na oracle; to tu sprawdza REPREZENTACJĘ na
        ślepo). Gdy `composite` nie jest złożeniem compose() (brak
        metadanych members — np. gołe make_morpheme), oracle i tak nie ma
        czego odjąć, więc `blind=False` daje ten sam wynik co `blind=True`
        (dawny fallback, teraz jawny, nie przypadkowy)."""
        if self._pos_hrr is None:
            raise NotImplementedError("brak karmazyn_hrr — unbind_position niedostępny")
        if position < 1:
            raise ValueError(
                "position musi być >= 1 — P^0 jest elementem identycznościowym "
                "i unbind(..., P^0) nie separuje niczego (patrz nagłówek pliku)"
            )
        fv = np.asarray(composite.vector[:FEATURE_DIM], dtype=float)

        member_ids = None if blind else _composite_member_ids(composite)
        if member_ids:
            for i, mid in enumerate(member_ids, start=1):
                if i == position:
                    continue
                sibling = self.store.get_atom(mid)
                if sibling is None or sibling.vector is None:
                    continue
                sibling_fv = np.asarray(sibling.vector[:FEATURE_DIM], dtype=float)
                fv = fv - _hrr.bind(self._pos_hrr.P_power(i), sibling_fv)

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


class Projector:
    """Metafora: LinguaLayer.unbind_position() dziś to bierny projektor —
    świeci JEDNĄ, z góry podaną wiązką P^position, bo wywołujący już wie,
    gdzie szukać. Projector.scan() robi to, czego brakuje — a teraz w
    DWÓCH wymiarach naraz:

      GDZIE patrzeć: kolejne wiązki referencyjne P^1, P^2, ... (positions).
      JAK patrzeć: z korektą o znanych sąsiadów (oracle) czy bez niej
        (blind, patrz unbind_position) — blind_modes.

    Dla każdej pary (pozycja, tryb) obserwator (nearest_morpheme) ocenia
    jakość rekonstrukcji, a scan() zwraca wszystkie próby posortowane po
    similarity — [0] to najsilniejszy, najbardziej spójny widok. To domyka
    pętlę "nieznany sygnał na wejściu -> rozpoznany fragment" bez
    wcześniejszej wiedzy WYWOŁUJĄCEGO ani o pozycji, ani o tym, czy warto
    ufać korekcie strukturalnej.

    Uczciwie: sama korekta 'oracle' nadal wymaga, żeby `composite` MIAŁ
    metadane members (compose() je zapisuje, patrz _composite_member_ids)
    — scan() z blind_modes=(False, True) nie odpowiada na 'czy stać nas
    bez ZNAJOMOŚCI struktury w ogóle' (gdy metadanych nie ma, oracle i tak
    nie zadziała, patrz unbind_position), tylko na 'czy WARTO z niej
    skorzystać, skoro jest dostępna' — obserwator może np. wykryć, że dwa
    widoki dają bardzo różną siłę sygnału (duża przepaść similarity), co
    samo w sobie jest sygnałem, czy dana rekonstrukcja jest wiarygodna."""

    def __init__(self, lingua: LinguaLayer):
        self.lingua = lingua

    def scan(self, composite, candidates: Sequence[Any],
              positions: Optional[Sequence[int]] = None,
              blind_modes: Sequence[bool] = (False,),
              top_k: int = 1) -> list[tuple[int, bool, float, Any]]:
        """Próbuje unbind_position(composite, pos, blind=b) dla KAŻDEJ pary
        (pos, b) z iloczynu `positions` x `blind_modes`. `positions`
        domyślnie: 1..len(members) odczytane z metadanych złożenia (patrz
        _composite_member_ids), 1..3 gdy brak metadanych. `blind_modes`
        domyślnie tylko (False,) — czyli zachowanie identyczne jak przed
        dodaniem osi trybu; podaj (False, True) albo (True,), żeby
        obserwator próbował też/tylko widoku bez korekty strukturalnej.
        Dla każdej próby bierze najlepsze dopasowanie z `candidates`
        (nearest_morpheme) i zwraca [(position, blind, similarity, atom),
        ...] posortowane malejąco po similarity — [0] to najlepszy strzał
        projektora, uwzględniający zarówno GDZIE jak i JAK patrzył."""
        if positions is None:
            member_ids = _composite_member_ids(composite)
            n = len(member_ids) if member_ids else 3
            positions = range(1, n + 1)

        results: list[tuple[int, bool, float, Any]] = []
        for pos in positions:
            for blind in blind_modes:
                try:
                    fv = self.lingua.unbind_position(composite, position=pos, blind=blind)
                except (NotImplementedError, ValueError):
                    continue
                ranking = self.lingua.nearest_morpheme(fv, candidates, top_k=top_k)
                if ranking:
                    sim, atom = ranking[0]
                    results.append((pos, blind, sim, atom))
        results.sort(key=lambda r: -r[2])
        return results

    def best(self, composite, candidates: Sequence[Any],
              positions: Optional[Sequence[int]] = None,
              blind_modes: Sequence[bool] = (False,),
              ) -> Optional[tuple[int, bool, float, Any]]:
        """Wygodny skrót: sam wynik [0] ze scan(), albo None gdy nic się
        nie dało odzyskać (np. composite bez cech w żadnej próbowanej
        parze pozycja/tryb)."""
        results = self.scan(composite, candidates, positions=positions,
                             blind_modes=blind_modes)
        return results[0] if results else None


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
    na realnym Store. Rdzeń zajmuje slot 0. Pod HOLOGRAM/SYMBOLIC wszystkie
    trzy prefiksy CELOWO dzielą slot 1 i mają tę samą amplitude=0.8 (różni
    je tylko faza) — wymusza pytanie wprost: czy faza rozróżnia prefiksy,
    skoro amplitude nie może. Pod SYMBOLIC_ONEHOT to pytanie nie ma sensu
    (onehot ignoruje fazę), więc tam każdy prefiks dostaje WŁASNY slot
    (1,2,3) — współdzielony slot dałby trzy identyczne wektory, czyli
    fałszywy 'twardy baseline'. Zwraca (lingua, [słowo_na, słowo_prze,
    słowo_wy]) — lingua trzeba zwrócić razem ze słowami, bo store/atomy
    żyją w niej."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)

    rdzen = lingua.make_morpheme(
        "pisa", [FeatureOverlay("rdzen_pisac", amplitude=1.0, phase=0.0, slot=0)], encoding
    )

    if encoding is AtomEncoding.SYMBOLIC_ONEHOT:
        prefiksy = {
            "na-": FeatureOverlay("aspekt_na", amplitude=0.8, phase=0.0, slot=1),
            "prze-": FeatureOverlay("aspekt_prze", amplitude=0.8, phase=0.0, slot=2),
            "wy-": FeatureOverlay("aspekt_wy", amplitude=0.8, phase=0.0, slot=3),
        }
    else:
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
# osobne dataclassy) to chodzi PO REALNYM STORE: drzewo odczytujemy przez
# _composite_member_ids (atom.metadata['v']['members'], ustawiane przez
# LinguaLayer.compose) — Store sam JEST strukturą, nie trzeba jej
# duplikować w Pythonie.

from holo_phi import energy_to_T  # noqa: E402  (import na dole: unika cyklu przy `python -m holo_phi`)
from transcode.phi_build import state_for_T  # noqa: E402


def _atom_energy(atom) -> float:
    """Energia atomu = norma L2 jego overlayu cech (FEATURE_DIM pierwszych
    wymiarów) — analog sumy amplitud ze starej wersji, ale liczony wprost
    z wektora, bez trzymania osobno listy FeatureOverlay na atomie."""
    fv = np.asarray(atom.vector[:FEATURE_DIM], dtype=float)
    return float(np.linalg.norm(fv))


# Alias lokalny: to_phi_atoms chodzi po TYM SAMYM drzewie, które
# unbind_position() czyta do odejmowania znanych interferentów — jedna
# definicja (_composite_member_ids, przed class LinguaLayer) dla obu.
_atom_children = _composite_member_ids


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

def _run_r1(tryb: AtomEncoding, prefiksy_faza: dict, blind: bool = False,
            _verbose: bool = True) -> float:
    """Buduje na-/prze-/wy-pisać z podanymi fazami prefiksów, próbuje
    odzyskać, KTÓRY prefiks był na pozycji 1, zwraca r@1.

    `blind=True` woła unbind_position(..., blind=True) — dekoder NIE zna
    struktury złożenia (nie odejmuje znanego wkładu rdzenia), więc mierzy
    to samo pytanie co pierwsza, "zepsuta" wersja unbind_position() z
    nagłówka pliku, ale teraz jako świadomy wybór do ablacji, nie
    przypadkowy fallback."""
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
        odzyskany = lingua.unbind_position(slowo, position=1, blind=blind)   # prefiks = 1. element
        ranking = lingua.nearest_morpheme(odzyskany, candidates, top_k=3)
        top1 = ranking[0][1].E
        ok = (top1 == nazwa)
        trafienia += ok
        if _verbose:
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


def _run_projector_scan() -> None:
    """Smoke test Projector.scan(): buduje na-/prze-/wy-pisać (fazy blisko
    siebie, ten sam trudny przypadek co hologram_faz_blisko w _run_r1),
    ale w przeciwieństwie do _run_r1 NIE MÓWI scan()-owi, że prefiks jest
    na pozycji 1 — scan próbuje pozycji 1 i 2 sam, na puli kandydatów
    złożonej z PREFIKSÓW i RDZENIA razem (żeby sprawdzić, że projektor nie
    tylko trafia właściwy morfem, ale i właściwą pozycję)."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)
    projector = Projector(lingua)

    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)],
                                  AtomEncoding.HOLOGRAM)
    prefiksy_faza = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    prefiks_atoms = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
        for n, ph in prefiksy_faza.items()
    }
    candidates = list(prefiks_atoms.values()) + [rdzen]

    trafienia_pozycja = 0
    trafienia_morfem = 0
    for nazwa, prefiks in prefiks_atoms.items():
        slowo = lingua.compose(f"{nazwa}pisać_scan", [prefiks, rdzen], kind="word")
        pos, blind, sim, atom = projector.best(slowo, candidates, positions=(1, 2))
        ok_pos = (pos == 1)
        ok_morfem = (atom.E == nazwa)
        trafienia_pozycja += ok_pos
        trafienia_morfem += ok_morfem
        print(f"    {nazwa}pisać_scan -> best pos={pos} top1={atom.E!r} sim={sim:+.3f} "
              f"{'OK' if ok_pos and ok_morfem else 'BLAD'}")
    n = len(prefiksy_faza)
    print(f"  Projector.scan: pozycja r@1={trafienia_pozycja/n:.3f}, "
          f"morfem r@1={trafienia_morfem/n:.3f}\n")


def _run_observer_search() -> None:
    """Obserwator szuka najlepszego widoku na NAJTRUDNIEJSZYM przypadku z
    ablacji (hologram, fazy stłoczone 30°-60° — tam _run_blind_ablation
    pokazał, że unbind_position(position=1, blind=True) SAMO pada do r@1
    przypadku=0.333, bo traci korektę strukturalną). Dwa pytania:

    (A) Czy samo PRZESZUKANIE POZYCJI na ślepo (blind_modes=(True,), bez
        oracle w ogóle) ratuje cokolwiek względem stałej pozycji=1 blind?
        Zgadywanie 'gdzie' nie naprawia utraty 'jak' — spodziewane: nadal
        słabo, to inna oś.
    (B) Gdy obserwator ma OBIE opcje na stole (blind_modes=(False, True)),
        czy sam, bez podpowiedzi, wybiera oracle — i o ile mocniejszy jest
        ten widok (przepaść similarity oracle vs blind na tej samej
        pozycji), skoro to jedyny sygnał, jaki obserwator ma do dyspozycji
        przy realnym 'nie wiem które jest prawdziwe'."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)
    projector = Projector(lingua)

    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)],
                                  AtomEncoding.HOLOGRAM)
    prefiksy_faza = {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}
    prefiks_atoms = {
        n: lingua.make_morpheme(n, [FeatureOverlay(f"a_{n}", 0.8, ph, 1)], AtomEncoding.HOLOGRAM)
        for n, ph in prefiksy_faza.items()
    }
    candidates = list(prefiks_atoms.values())

    trafienia_blind_only = 0
    trafienia_mixed = 0
    gaps = []
    for nazwa, prefiks in prefiks_atoms.items():
        slowo = lingua.compose(f"{nazwa}pisać_obs", [prefiks, rdzen], kind="word")

        # (A) tylko blind, ale przeszukuje obie pozycje
        pos_b, _, sim_b, atom_b = projector.best(slowo, candidates, positions=(1, 2),
                                                   blind_modes=(True,))
        ok_b = (atom_b.E == nazwa)
        trafienia_blind_only += ok_b

        # (B) oba tryby na stole naraz — obserwator wybiera sam
        wszystkie = projector.scan(slowo, candidates, positions=(1, 2),
                                    blind_modes=(False, True))
        pos_m, blind_m, sim_m, atom_m = wszystkie[0]
        ok_m = (atom_m.E == nazwa) and not blind_m
        trafienia_mixed += ok_m
        # przepaść: najlepszy oracle-widok vs najlepszy blind-widok
        best_oracle = max((r for r in wszystkie if not r[1]), key=lambda r: r[2], default=None)
        best_blind = max((r for r in wszystkie if r[1]), key=lambda r: r[2], default=None)
        if best_oracle and best_blind:
            gap = best_oracle[2] - best_blind[2]
            gaps.append(gap)
            print(f"    {nazwa}pisać_obs -> (A) blind-only best pos={pos_b} "
                  f"top1={atom_b.E!r} sim={sim_b:+.3f} {'OK' if ok_b else 'BLAD'} | "
                  f"(B) mixed wybrał blind={blind_m} pos={pos_m} top1={atom_m.E!r} "
                  f"sim={sim_m:+.3f} {'OK' if ok_m else 'BLAD'} | "
                  f"przepasc(oracle-blind)={gap:+.3f}")

    n = len(prefiksy_faza)
    print(f"  (A) blind-only (przeszukane obie pozycje): r@1={trafienia_blind_only/n:.3f}")
    print(f"  (B) mixed (oracle+blind na stole, obserwator wybiera): r@1={trafienia_mixed/n:.3f}")
    if gaps:
        print(f"  srednia przepasc similarity oracle-vs-blind na tej samej pozycji: "
              f"{sum(gaps)/len(gaps):+.3f}\n")


def _run_blind_ablation() -> dict:
    """Kontrolowana ablacja oracle vs blind na TYCH SAMYCH trzech
    scenariuszach co reszta _demo_retrieval (symbolic, hologram fazy
    blisko, hologram fazy rozstawione) — jedyna zmienna to blind=True/False
    w unbind_position, wszystko inne (dane, fazy, słowa) identyczne.
    Odpowiada wprost na pytanie z NASTĘPNY KROK: czy retrieval przeżywa,
    gdy dekoder nie zna struktury złożenia z góry (nie odejmuje znanego
    wkładu rdzenia), czy to oracle (nie reprezentacja) robił robotę."""
    scenariusze = {
        "symbolic": (AtomEncoding.SYMBOLIC,
                     {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}),
        "hologram_faz_blisko": (AtomEncoding.HOLOGRAM,
                                 {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}),
        "hologram_faz_rozstawione": (AtomEncoding.HOLOGRAM,
                                      {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3}),
    }
    wyniki: dict = {}
    for nazwa, (tryb, fazy) in scenariusze.items():
        r_oracle = _run_r1(tryb, fazy, blind=False, _verbose=False)
        r_blind = _run_r1(tryb, fazy, blind=True, _verbose=False)
        wyniki[nazwa] = {"oracle": r_oracle, "blind": r_blind}
        print(f"    {nazwa:28s} oracle r@1={r_oracle:.3f}  blind r@1={r_blind:.3f}  "
              f"{'ROZNICA' if r_oracle != r_blind else 'bez zmian'}")
    return wyniki


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

    print("=== Projector.scan (pozycja NIEZNANA z góry, skanowana) ===")
    _run_projector_scan()

    print("=== BLIND vs ORACLE (dekoder nie zna struktury złożenia) ===")
    _run_blind_ablation()

    print("=== Obserwator szuka najlepszego widoku (pozycja x tryb, przypadek trudny) ===")
    _run_observer_search()

    print("\n=== to_phi_atoms smoke (rodzina pisać -> PhiJSON) ===")
    lingua, slowa = przyklad_pisac_rodzina(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(lingua, slowa)
    print(f"  {len(atoms)} atoms, {len(edges)} edges, ref_energy={meta['ref_energy']:.3f}")


if __name__ == "__main__":
    _demo_retrieval()
