"""
MemoryBus — cienka fasada M1↔M2: ingest → encode → recall.

Nie miesza Holon SE (M3). Live Store tylko przez istniejący live_store + flagi.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from checkpoint import DEFAULT_MOST, load_encoder
from dump_recall import graph_to_tensors, records_to_graph
from phi_dump import load_dump, warm_without_payload
from phi_schema import PhiGraph, text_as_content_vec

ROOT = Path(__file__).resolve().parent
DEFAULT_EPISODES = ROOT / "dumps" / "episodes.jsonl"


@dataclass
class RecallHit:
    atom_id: str
    score: float
    index: int
    depth: int
    state: str
    E: str


class MemoryBus:
    """ingest(snapshot) → encode → recall(q, k)."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        d_x: int = 8,
        n_dim: int = 6,
        device: str | torch.device = "cpu",
    ):
        self.d_x = d_x
        self.n_dim = n_dim
        self.device = torch.device(device)
        path = Path(checkpoint) if checkpoint else DEFAULT_MOST
        self.model, self.meta = load_encoder(path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.graph: PhiGraph | None = None
        self.content: torch.Tensor | None = None  # [N,d_x]
        self.tracer: torch.Tensor | None = None  # [N,n]
        self.mask: torch.Tensor | None = None  # [N]
        self.tok: torch.Tensor | None = None  # [N,d]
        self.source: str = ""

    # --- ingest ---

    def ingest_graph(self, g: PhiGraph, *, source: str = "graph") -> int:
        g = warm_without_payload(g)
        # blend_holo_proto zgodny z finetune_karmazyn_live (klaster hologramu w x)
        c, t, _, _ = graph_to_tensors(
            g,
            d_x=self.d_x,
            n_dim=self.n_dim,
            blend_holo_proto=0.35,
        )
        N = len(g.atoms)
        self.graph = g
        self.content = c
        self.tracer = t
        self.mask = torch.ones(N)
        self.tok = None
        self.source = source
        return N

    def ingest_dump(self, dump_dir: str | Path) -> int:
        dump_dir = Path(dump_dir)
        _, g = load_dump(dump_dir)
        return self.ingest_graph(g, source=str(dump_dir))

    def ingest_snapshot_rows(
        self, rows: list[dict], edges: list | None = None
    ) -> int:
        g = records_to_graph(rows, edges or [])
        return self.ingest_graph(g, source="rows")

    # --- encode ---

    @torch.no_grad()
    def encode(self, *, force: bool = False) -> torch.Tensor:
        if self.content is None or self.tracer is None or self.mask is None:
            raise RuntimeError("MemoryBus: najpierw ingest_*")
        if self.tok is not None and not force:
            return self.tok
        x = self.content.unsqueeze(0).to(self.device)
        t = self.tracer.unsqueeze(0).to(self.device)
        m = self.mask.unsqueeze(0).to(self.device)
        tok, _ = self.model(x, t, m)
        self.tok = tok.squeeze(0).cpu()
        return self.tok

    # --- recall ---

    def _require_encoded(self) -> tuple[PhiGraph, torch.Tensor]:
        if self.graph is None:
            raise RuntimeError("MemoryBus: brak grafu")
        tok = self.encode()
        return self.graph, tok

    @torch.no_grad()
    def recall_by_index(
        self,
        query_idx: int,
        k: int = 5,
        *,
        same_prefix: bool = False,
        exclude_nucleus: bool = True,
    ) -> list[RecallHit]:
        """
        exclude_nucleus: ukryj */nucleus w rankingu orbitali (domyślnie tak).
        Gdy query sam jest jądrem — nie wykluczaj.
        """
        g, tok = self._require_encoded()
        N = tok.shape[0]
        if query_idx < 0 or query_idx >= N:
            raise IndexError(query_idx)
        q = F.normalize(tok[query_idx : query_idx + 1], dim=-1)  # [1,d]
        all_h = F.normalize(tok, dim=-1)
        scores = (all_h @ q.squeeze(0)).clone()
        scores[query_idx] = -1e9
        if self.mask is not None:
            scores = scores.masked_fill(self.mask < 0.5, -1e9)
        qid = g.atoms[query_idx].atom_id
        if same_prefix:
            pref = qid.split("/", 1)[0]
            for j, a in enumerate(g.atoms):
                if not a.atom_id.startswith(pref):
                    scores[j] = -1e9
        if exclude_nucleus and not qid.endswith("/nucleus"):
            for j, a in enumerate(g.atoms):
                if a.atom_id.endswith("/nucleus"):
                    scores[j] = -1e9
        n_ok = int((scores > -1e8).sum().item())
        if n_ok <= 0:
            return []
        k = max(1, min(k, n_ok))
        topv, topi = torch.topk(scores, k)
        return [self._hit(int(i), float(v)) for i, v in zip(topi.tolist(), topv.tolist())]

    @torch.no_grad()
    def recall_by_id(
        self,
        atom_id: str,
        k: int = 5,
        *,
        same_prefix: bool = False,
        exclude_nucleus: bool = True,
    ) -> list[RecallHit]:
        g, _ = self._require_encoded()
        ix = {a.atom_id: i for i, a in enumerate(g.atoms)}
        if atom_id not in ix:
            raise KeyError(f"atom_id nie w snapshot: {atom_id!r}")
        return self.recall_by_index(
            ix[atom_id],
            k=k,
            same_prefix=same_prefix,
            exclude_nucleus=exclude_nucleus,
        )

    @torch.no_grad()
    def recall_by_text(self, text: str, k: int = 5) -> list[RecallHit]:
        """Query w połowie E contentu → najbliższy atom → expand w enkoderze."""
        g, tok = self._require_encoded()
        assert self.content is not None
        q = F.normalize(text_as_content_vec(text, self.d_x), dim=-1)
        c = F.normalize(self.content, dim=-1)
        sims = c @ q
        if self.mask is not None:
            sims = sims.masked_fill(self.mask < 0.5, -1e9)
        qi = int(sims.argmax().item())
        # ranking w przestrzeni encodera względem wybranego atomu-kotwicy
        return self.recall_by_index(qi, k=k)

    def _hit(self, idx: int, score: float) -> RecallHit:
        assert self.graph is not None
        a = self.graph.atoms[idx]
        return RecallHit(
            atom_id=a.atom_id,
            score=score,
            index=idx,
            depth=a.depth,
            state=a.state.name,
            E=a.E[:120],
        )

    # --- export / episodes ---

    def export_dump(self, path: str | Path) -> Path:
        """RO mirror M2: zapis atoms+edges+manifest (bez Holon)."""
        if self.graph is None:
            raise RuntimeError("brak grafu do exportu")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        atoms_p = path / "atoms.jsonl"
        edges_p = path / "edges.jsonl"
        with atoms_p.open("w", encoding="utf-8") as f:
            for a in self.graph.atoms:
                row = {
                    "id": a.atom_id,
                    "S": a.S,
                    "E": a.E,
                    "T": a.T,
                    "state": a.state.name,
                    "bubble": a.bubble_id,
                    "depth": a.depth,
                    "payload": a.payload,
                    "mazur_tracer": a.mazur_tracer,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with edges_p.open("w", encoding="utf-8") as f:
            for e in self.graph.edges:
                f.write(
                    json.dumps(
                        {"kind": e.kind, "src": e.src, "dst": e.dst, "name": e.name},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        man = {
            "format": "phijson-v1",
            "source": self.source,
            "n_atoms": len(self.graph.atoms),
            "n_edges": len(self.graph.edges),
            "exported_at": time.time(),
        }
        (path / "manifest.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8"
        )
        return path


def append_episode(
    record: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    """JSONL epizodów obok dumpów — mirror M2, nie Holon."""
    path = Path(path) if path else DEFAULT_EPISODES
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def hits_to_dicts(hits: Iterable[RecallHit]) -> list[dict]:
    return [
        {
            "atom_id": h.atom_id,
            "score": h.score,
            "index": h.index,
            "depth": h.depth,
            "state": h.state,
            "E": h.E,
        }
        for h in hits
    ]
