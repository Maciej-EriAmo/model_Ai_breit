"""
karmazyn_lingua.py
===================

Warstwa lingwistyczna NA WIERZCHU zamrożonego substratu KarmazynOS —
Store/Atom/Bubble/HRR substratu pozostaje NIETKNIĘTY.

Substrat podłączony (RealSubstrateAdapter, sekcja 6) do prawdziwych modułów:
  C:\\Users\\drwis\\KarmazynOs\\...          -> karmazyn_substrate.Store
  C:\\Users\\drwis\\KarmazynOs\\archiwum\\kernel_python\\karmazyn_hrr.py
Kontrakt rdzenia = karmazyn_atomstore.AtomStore (create_atom/get_atom/
has_atom/create_bubble/import_to_bubble/get_bubble); bind/unbind/similarity
to CZYSTE funkcje karmazyn_hrr (nie metody Store — Store ma tylko
resonance() po zarejestrowanych nazwach). Degradacja łagodna: jeśli
KarmazynOs nie jest zainstalowany, connect_real_substrate() zwraca None
zamiast wyjątku (ten sam wzorzec co opcjonalny import _hrr w
karmazyn_substrate.py).

Most/PhiMost (ten repozytorium): sekcja 7 (`to_phi_atoms`) eksportuje
las MorphAtom/LinguisticBubble do kanonu PhiJSON tego repo (te same
klucze co `holo_phi.build_holo_graph` / `transcode.phi_build`), więc
wpina się w istniejący pipeline `make_dump_*.py -> dumps/... ->
phimost_learn.py` bez zmiany silnika Most. CLI: `make_dump_lingua.py`.

Dwa poziomy reprezentacji (ustalone w rozmowie):
  1) ATOM = morfem/glif, cechy kodowane na fragmencie wektora D=2048
     jako hologram-overlay: nakładanie A*e^(i*phi) na wymiarach
     zarezerwowanych pod konkretną cechę (diakrytyk, aspekt, przypadek...).
  2) BUBBLE = wiązanie atomów w słowo PRZEZ ISTNIEJĄCY MECHANIZM HRR
     substratu (bind), NIE przez nową interferencję fal — wniosek
     z porażki Holo-Phi P4a (faza nie niosła sygnału rodziny, jawne
     binds tak).
  3) Bubble.as_atom_vector() redukuje się do jednego wektora, który
     może wejść jako atom do Bubble WYŻSZEGO poziomu — rekurencja n+1:
     morfem -> słowo -> fraza -> klauzula.

Hooki pod ablację P3-style: każdy MorphAtom ma encoding=HOLOGRAM albo
encoding=SYMBOLIC, ten sam wymiar wektora, więc (b) vs (c) da się
porównać na identycznym pipeline.

ZASTRZEŻENIE DO WŁASNEGO KODU PONIŻEJ (celowo zostawione widoczne):
SYMBOLIC w obecnej wersji koduje cechę jako "amplitude na współdzielonym
slocie", więc różne morfemy o tej samej amplitude i różnej fazie (patrz
przykład na dole: trzy prefiksy, wszystkie amplitude=0.8) stają się
NIEROZRÓŻNIALNE w trybie symbolic. To NIE dowodzi przewagi hologramu —
to słaby baseline. Uczciwy test potrzebuje też "twardego" symbolic
baseline: jeden dedykowany wymiar na KAŻDY odrębny morfem (prawdziwe
one-hot), patrz TODO przy SYMBOLIC_ONEHOT niżej.
"""

from __future__ import annotations

import cmath
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# 1. Minimalny protokół substratu — zgodny z realnym kontraktem KarmazynOS
#    (karmazyn_atomstore.AtomStore); zweryfikowany na
#    C:\Users\drwis\DBase\karmazyn_{substrate,atomstore}.py i
#    C:\Users\drwis\KarmazynOs\archiwum\kernel_python\karmazyn_hrr.py.
# ---------------------------------------------------------------------------

@runtime_checkable
class SubstrateAtom(Protocol):
    """To, co realny karmazyn_atom.Atom faktycznie ma (nie `identity`/`thermal`
    — to były zgadnięte pola z wersji bez podpiętego substratu)."""
    id: str
    vector: object      # numpy.ndarray | None w prawdziwym substracie (D=2048)
    T: float


@runtime_checkable
class SubstrateProtocol(Protocol):
    """Rdzeń AtomStore (karmazyn_atomstore.CORE_METHODS) + hrr_bind/hrr_query
    — rozszerzenie TEJ warstwy, bo Store sam nie eksponuje "gołego" bind na
    dwóch dowolnych wektorach (tylko resonance() po zarejestrowanych nazwach).
    RealSubstrateAdapter (sekcja 6) mapuje to na Store + moduł karmazyn_hrr."""

    def create_atom(self, id: str, S: str, E: str, T: float, **kwargs) -> str: ...
    def get_atom(self, id: str) -> SubstrateAtom | None: ...
    def create_bubble(self, label: str, atom_ids: Sequence[str] | None = None) -> str: ...
    def hrr_bind(self, a: Sequence[float], b: Sequence[float]) -> list[float]: ...
    def hrr_query(self, composite: Sequence[float], probe: Sequence[float]) -> float: ...


# ---------------------------------------------------------------------------
# 2. Tryb reprezentacji atomu — potrzebny do ablacji P3-style
# ---------------------------------------------------------------------------

class AtomEncoding(Enum):
    HOLOGRAM = "hologram"           # nakładanie A*e^(i*phi) na cechy
    SYMBOLIC = "symbolic"           # amplitude na współdzielonym slocie (słaby baseline)
    SYMBOLIC_ONEHOT = "onehot"      # TODO: każdy odrębny morfem = własny wymiar (twardy baseline)


# ---------------------------------------------------------------------------
# 3. MorphAtom — poziom 0: morfem / glif / sylaba
# ---------------------------------------------------------------------------

FEATURE_DIM = 64  # ile wymiarów wektora D=2048 rezerwujemy pod cechy


@dataclass
class FeatureOverlay:
    """Jedna nałożona cecha (diakrytyk, aspekt, przypadek...).

    amplitude/phase — parametry hologramu (konwencja Holo-Phi: A*e^(i*phi)).
    slot — który z FEATURE_DIM kanałów zajmuje ta KATEGORIA cechy
    (np. 'diacritic', 'case', 'aspect' mają rozłączne zakresy slotów,
    żeby się nie mieszały ze sobą; różne WARTOŚCI w tej samej kategorii
    mogą celowo dzielić slot, żeby przetestować, czy faza je rozróżnia).
    """
    name: str
    amplitude: float
    phase: float          # w radianach
    slot: int             # indeks kanału w obrębie FEATURE_DIM


@dataclass
class MorphAtom:
    """Atom poziomu 0: morfem, glif bazowy, albo pojedyncza sylaba."""

    identity: str                       # np. "roz-", "pisa", "-ć", "ą"
    base_features: list[FeatureOverlay] = field(default_factory=list)
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM
    content_vector: list[float] = field(default_factory=list)  # reszta D=2048

    def feature_vector(self) -> list[float]:
        """Zwraca FEATURE_DIM-wymiarowy fragment wektora atomu."""
        if self.encoding is AtomEncoding.HOLOGRAM:
            return self._overlay_hologram()
        if self.encoding is AtomEncoding.SYMBOLIC:
            return self._overlay_symbolic_amplitude()
        raise NotImplementedError(
            f"{self.encoding} jeszcze nie zaimplementowany (patrz TODO SYMBOLIC_ONEHOT)"
        )

    def _check_slots(self) -> None:
        for f in self.base_features:
            if not (0 <= f.slot < FEATURE_DIM):
                raise ValueError(
                    f"Atom {self.identity!r}: slot {f.slot} poza zakresem "
                    f"FEATURE_DIM={FEATURE_DIM}"
                )

    def _overlay_hologram(self) -> list[float]:
        """Superpozycja A*e^(i*phi) per kanał; bierzemy część rzeczywistą
        (A*cos(phi)) — zgodnie z konwencją 'word = interferencja fal liter'
        z Holo-Phi. Wiele cech na tym samym slocie SUMUJE się (to jest
        właśnie 'nakładanie znaków')."""
        self._check_slots()
        out = [0.0] * FEATURE_DIM
        for f in self.base_features:
            z = cmath.rect(f.amplitude, f.phase)   # A*e^(i*phi)
            out[f.slot] += z.real
        return out

    def _overlay_symbolic_amplitude(self) -> list[float]:
        """Słaby baseline: obecność cechy = amplitude wprost, faza ignorowana.
        Jeśli dwie różne cechy dzielą slot i mają tę samą amplitude,
        stają się NIEROZRÓŻNIALNE — to jest zamierzone i pokazane w demo
        na dole, właśnie po to, żeby było widać, co faza ewentualnie
        dokłada ponad amplitude."""
        self._check_slots()
        out = [0.0] * FEATURE_DIM
        for f in self.base_features:
            out[f.slot] += f.amplitude
        return out

    def full_vector(self, dim: int = 2048) -> list[float]:
        if dim < FEATURE_DIM:
            raise ValueError(f"dim={dim} musi być >= FEATURE_DIM={FEATURE_DIM}")
        fv = self.feature_vector()
        rest = list(self.content_vector[: dim - FEATURE_DIM])
        rest += [0.0] * (dim - FEATURE_DIM - len(rest))
        return fv + rest


# ---------------------------------------------------------------------------
# 4. LinguisticBubble — poziom 1+: słowo / fraza / klauzula
# ---------------------------------------------------------------------------

@dataclass
class LinguisticBubble:
    """Bubble wiążący atomy (morfemy LUB inne Bubble-jako-atomy) w jednostkę
    wyższego poziomu. Wiązanie idzie przez HRR substratu (jeśli podpięty) —
    NIE przez nakładanie fal. To jest wprost lekcja z Holo-Phi P4a."""

    label: str
    members: list["MorphAtom | LinguisticBubble"] = field(default_factory=list)
    substrate: SubstrateProtocol | None = None

    def member_vectors(self, dim: int = 2048) -> list[list[float]]:
        out = []
        for m in self.members:
            if isinstance(m, LinguisticBubble):
                out.append(m.as_atom_vector(dim))
            else:
                out.append(m.full_vector(dim))
        return out

    def as_atom_vector(self, dim: int = 2048) -> list[float]:
        """Redukuje Bubble do jednego wektora, żeby mógł wejść jako atom
        do Bubble wyższego poziomu (rekurencja n+1: słowo->fraza->klauzula).

        Z podpiętym substratem -> prawdziwy HRR bind (rekomendowane).
        Bez substratu -> element-wise suma jako placeholder TYLKO do
        testów offline / tego demo — do zastąpienia realnym bind()."""
        vectors = self.member_vectors(dim)
        if not vectors:
            return [0.0] * dim
        if self.substrate is not None:
            acc = vectors[0]
            for v in vectors[1:]:
                acc = self.substrate.hrr_bind(acc, v)
            return acc
        acc = [0.0] * dim
        for v in vectors:
            for i, x in enumerate(v):
                acc[i] += x
        return acc

    def persist(self, dim: int = 2048) -> str | None:
        """Materializuje Bubble w PRAWDZIWYM Store (jeśli substrate ma
        create_atom/create_bubble — RealSubstrateAdapter): każdy liść
        MorphAtom -> atom Store z jego wektorem cech; ten Bubble -> bąbel
        Store wiążący ich id. Zwraca label bąbla albo None bez substratu.
        Zagnieżdżone LinguisticBubble są persystowane rekurencyjnie
        NAJPIERW (n+1), tak żeby ich HRR-bind-owany wektor trafił jako
        `vector` atomu-reprezentanta wyższego poziomu."""
        if self.substrate is None:
            return None
        atom_ids: list[str] = []
        for i, m in enumerate(self.members):
            if isinstance(m, LinguisticBubble):
                m.persist(dim=dim)
                aid = f"{self.label}/{m.label}"
                vec = m.as_atom_vector(dim)
            else:
                aid = f"{self.label}/{m.identity}/{i}"
                vec = m.full_vector(dim)
            self.substrate.create_atom(aid, S=aid, E=m.label if isinstance(m, LinguisticBubble) else m.identity, T=1.0, vector=vec)
            atom_ids.append(aid)
        return self.substrate.create_bubble(self.label, atom_ids=atom_ids)


# ---------------------------------------------------------------------------
# 5. Przykład: polska rodzina aspektowa jako n+1 (odpalalne od razu)
# ---------------------------------------------------------------------------

def przyklad_pisac_rodzina_bubbles(
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM,
    substrate: SubstrateProtocol | None = None,
) -> list[LinguisticBubble]:
    """pisać / napisać / przepisać / wypisać jako Bubble(prefiks + rdzeń).

    Rdzeń zajmuje slot 0. Wszystkie trzy prefiksy CELOWO dzielą slot 1
    i mają tę samą amplitude=0.8 (różni je tylko faza) — to wymusza
    pytanie wprost: czy faza rozróżnia prefiksy, skoro amplitude nie może."""

    rdzen = MorphAtom(
        identity="pisa",
        base_features=[FeatureOverlay("rdzen_pisac", amplitude=1.0, phase=0.0, slot=0)],
        encoding=encoding,
    )

    prefiksy = {
        "na-": FeatureOverlay("aspekt_na", amplitude=0.8, phase=math.pi / 4, slot=1),
        "prze-": FeatureOverlay("aspekt_prze", amplitude=0.8, phase=math.pi / 3, slot=1),
        "wy-": FeatureOverlay("aspekt_wy", amplitude=0.8, phase=math.pi / 6, slot=1),
    }

    slowa: list[LinguisticBubble] = []
    for nazwa, feat in prefiksy.items():
        prefiks = MorphAtom(identity=nazwa, base_features=[feat], encoding=encoding)
        slowa.append(
            LinguisticBubble(label=f"{nazwa}pisać", members=[prefiks, rdzen], substrate=substrate)
        )
    return slowa


def przyklad_pisac_rodzina(
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM, dim: int = 128
) -> dict[str, list[float]]:
    """Jak przyklad_pisac_rodzina_bubbles, ale zredukowane do wektorów —
    wygodne dla starych callerów / demo w __main__."""
    wyniki: dict[str, list[float]] = {}
    for slowo in przyklad_pisac_rodzina_bubbles(encoding):
        wyniki[slowo.label] = slowo.as_atom_vector(dim=dim)

    return wyniki


# ---------------------------------------------------------------------------
# 6. Adapter do prawdziwego substratu KarmazynOS
# ---------------------------------------------------------------------------

# Ten sam wzorzec sys.path co export_karmazyn_live.py._setup_path() —
# substrat NIE jest w tym repo, żyje w C:\Users\drwis\KarmazynOs (zamrożony,
# ta warstwa go tylko woła, nie modyfikuje).
KARMAZYN_ROOT = Path(r"C:\Users\drwis\KarmazynOs")


def _setup_karmazyn_path() -> None:
    for p in (KARMAZYN_ROOT, KARMAZYN_ROOT / "archiwum" / "kernel_python"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


class RealSubstrateAdapter:
    """Woła prawdziwy Store (karmazyn_substrate.py, kontrakt AtomStore z
    karmazyn_atomstore.py) + moduł karmazyn_hrr (bind/unbind/similarity,
    splot kołowy na D=2048). Substrat jest zamrożony — ta klasa go NIE
    modyfikuje, tylko go woła (jak karmazyn_econ.py dla ekonomii).

    `store`: obiekt zgodny z AtomStore (np. karmazyn_substrate.Store()).
    `hrr`: moduł/obiekt z bind(a,b)/similarity(a,b) na numpy ndarray
    (domyślnie prawdziwy karmazyn_hrr — patrz connect_real_substrate())."""

    def __init__(self, store: SubstrateProtocol, hrr) -> None:
        self.store = store
        self._hrr = hrr

    def create_atom(self, id: str, S: str, E: str, T: float, **kwargs) -> str:
        return self.store.create_atom(id, S, E, T, **kwargs)

    def get_atom(self, id: str):
        return self.store.get_atom(id)

    def create_bubble(self, label: str, atom_ids: Sequence[str] | None = None) -> str:
        return self.store.create_bubble(label, atom_ids=list(atom_ids) if atom_ids else None)

    def hrr_bind(self, a: Sequence[float], b: Sequence[float]) -> list[float]:
        import numpy as np

        v = self._hrr.bind(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        return v.tolist()

    def hrr_query(self, composite: Sequence[float], probe: Sequence[float]) -> float:
        import numpy as np

        return float(
            self._hrr.similarity(np.asarray(composite, dtype=float), np.asarray(probe, dtype=float))
        )


def connect_real_substrate(*, thermal: bool = True) -> RealSubstrateAdapter | None:
    """Best-effort podłączenie do żywego KarmazynOS. Łagodna degradacja:
    None gdy KarmazynOs (albo numpy) nie jest dostępny — ten sam wzorzec co
    opcjonalny `import karmazyn_hrr` w karmazyn_substrate.py, żeby ta warstwa
    dawała się testować/importować bez zewnętrznego repo."""
    try:
        _setup_karmazyn_path()
        from karmazyn_substrate import Store
        import karmazyn_hrr as hrr
    except Exception:
        return None
    return RealSubstrateAdapter(Store(thermal=thermal), hrr)


# ---------------------------------------------------------------------------
# 7. Most z tego repo: eksport do kanonu PhiJSON (holo_phi / transcode)
# ---------------------------------------------------------------------------
#
# Karmazyn substrate (sekcja 6) to zamrożony magazyn atomów KarmazynOS.
# Most (silnik uczący tego repo) NIE czyta z niego wprost — czyta PhiJSON
# (dumps/<name>/{atoms,edges}.jsonl + manifest.json, zob. phi_dump.py).
# `to_phi_atoms` to most między nimi: atomy w tych samych kluczach co
# holo_phi.build_holo_graph / transcode.phi_build.build_from_spans, więc
# działa z export_graph_as_dump / phi_dump.load_dump / phimost_learn.py bez
# zmian w silniku. `T` z energii cech (suma amplitud) na wspólnym `ref` —
# ta sama konwencja co holo_phi §2b (energy_to_T), nie hash.

from holo_phi import energy_to_T  # noqa: E402  (import na dole: unika cyklu przy `python -m holo_phi`)
from transcode.phi_build import state_for_T  # noqa: E402


def _feature_energy(atom: MorphAtom) -> float:
    return sum(f.amplitude for f in atom.base_features)


def _node_energy(node: "MorphAtom | LinguisticBubble") -> float:
    if isinstance(node, LinguisticBubble):
        return sum(_node_energy(m) for m in node.members)
    return _feature_energy(node)


def to_phi_atoms(
    bubbles: Sequence[LinguisticBubble],
    *,
    page_id: str = "lingua_page",
    corpus_note: str = "karmazyn_lingua: morfem -HRR bind-> slowo (n+1)",
) -> tuple[list[dict], list[dict], dict]:
    """Las LinguisticBubble (np. rodzina słów) -> (atoms, edges, meta) w
    kanonie PhiJSON tego repo. Rekurencyjne: obsługuje dowolną głębię
    zagnieżdżenia Bubble-w-Bubble (morfem -> słowo -> fraza -> klauzula),
    nie tylko płaski przypadek z `przyklad_pisac_rodzina_bubbles`.

    depth: page=0, top-level Bubble=1, każdy poziom zagnieżdżenia +1
    (MAX_DEPTH_BINS=3 w tracerze n_dim=6 — patrz transcode/phi_build.py;
    głębsze drzewa nadal dostają poprawny int `depth`, ale trzeci+ bin
    tracer spłaszcza razem, jak w reszcie repo)."""
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
            "id": page_id, "S": "lingua/page", "E": f"page n_bubbles={len(bubbles)}",
            "T": 0.92, "state": "HOT", "bubble": "lingua", "depth": 0, "unit": "page",
            "payload": None, "mazur_tracer": 0.5, "topic": "page",
        }
    )
    bind_edge("corpus", page_id, "has_page")

    energies = [_node_energy(b) for b in bubbles]
    ref = max(energies) if energies and max(energies) > 0 else 1.0

    def emit(node: "MorphAtom | LinguisticBubble", node_id: str, parent_id: str, depth: int, topic: str) -> None:
        if isinstance(node, LinguisticBubble):
            unit, label = "bubble", node.label
            S = f"lingua:bubble:{label}"
            E = f"bubble depth={depth} n_members={len(node.members)}"
        else:
            unit, label = "morph", node.identity
            enc = node.encoding.value
            S = f"lingua:{enc}:{label}"
            E = f"morph encoding={enc} n_features={len(node.base_features)}"

        T = energy_to_T(_node_energy(node), ref=ref)
        st = state_for_T(T)
        atoms.append(
            {
                "id": node_id, "S": S, "E": E, "T": T, "state": st,
                "bubble": f"lingua/{topic}", "depth": depth, "unit": unit,
                "payload": (label if st == "HOT" else None),
                "mazur_tracer": max(0.0, min(1.0, _node_energy(node) / ref)),
                "topic": topic,
            }
        )
        edges.append({"kind": "parent", "src": node_id, "dst": parent_id, "name": f"{unit}_of"})
        bind_edge(parent_id, node_id, f"has_{unit}:{label}")

        if isinstance(node, LinguisticBubble):
            member_ids = []
            for i, m in enumerate(node.members):
                mid = f"{node_id}/m{i}"
                emit(m, mid, node_id, depth + 1, topic)
                member_ids.append(mid)
            for a, b in zip(member_ids, member_ids[1:]):
                bind_edge(a, b, "next")

    top_ids: list[str] = []
    for i, b in enumerate(bubbles):
        bid = f"bubble/{i}/{b.label}"
        emit(b, bid, page_id, 1, b.label)
        top_ids.append(bid)
    for a, b in zip(top_ids, top_ids[1:]):
        bind_edge(a, b, "next_bubble")

    meta = {
        "method": "karmazyn_lingua",
        "contract": "phijson-v1",
        "n_top_bubbles": len(bubbles),
        "ref_energy": ref,
        "recursion": "morfem->slowo->fraza->klauzula (n+1)",
    }
    return atoms, edges, meta


if __name__ == "__main__":
    for tryb in (AtomEncoding.SYMBOLIC, AtomEncoding.HOLOGRAM):
        print(f"--- {tryb.value} ---")
        for slowo, wektor in przyklad_pisac_rodzina(tryb).items():
            print(f"{slowo:15s} slot0={wektor[0]:+.3f}  slot1={wektor[1]:+.3f}")

    bubbles = przyklad_pisac_rodzina_bubbles(AtomEncoding.HOLOGRAM)
    atoms, edges, meta = to_phi_atoms(bubbles)
    print(f"\nto_phi_atoms: {len(atoms)} atoms, {len(edges)} edges, ref_energy={meta['ref_energy']:.3f}")

    adapter = connect_real_substrate()
    if adapter is None:
        print("connect_real_substrate: KarmazynOs niedostępny (łagodna degradacja, OK).")
    else:
        bubbles_live = przyklad_pisac_rodzina_bubbles(AtomEncoding.HOLOGRAM, substrate=adapter)
        label = bubbles_live[0].persist()
        print(f"connect_real_substrate: OK -> persisted bubble {label!r} w żywym Store")
