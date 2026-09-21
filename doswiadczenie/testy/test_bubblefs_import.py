"""P2.12 — BubbleFS fixture → PhiGraph → dump → MemoryBus opcjonalnie."""

from __future__ import annotations

import _labroot  # root labu na sys.path

from pathlib import Path

from bubblefs_import import (
    bubblefs_to_graph,
    export_graph_as_dump,
    write_fixture_bubblefs,
)
from phi_schema import PhiState

ROOT = _labroot.ROOT


def main():
    src = write_fixture_bubblefs(ROOT / "dumps" / "bubblefs_fixture")
    man, g = bubblefs_to_graph(src)
    assert man.get("bubblefs_version") == "1.0.0"
    assert len(g.atoms) >= 4
    assert any(a.state == PhiState.HOT for a in g.atoms)
    assert any(e.kind == "bind" for e in g.edges), "hologram bind expected"
    out = export_graph_as_dump(g, ROOT / "dumps" / "from_bubblefs")
    assert (out / "atoms.jsonl").is_file()
    print(
        f"OK — BubbleFS atoms={len(g.atoms)} edges={len(g.edges)} "
        f"bind={sum(1 for e in g.edges if e.kind == 'bind')} → {out}"
    )


if __name__ == "__main__":
    main()
