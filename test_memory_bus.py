"""Smoke MemoryBus: ingest dump → encode → recall + episode JSONL."""

from __future__ import annotations

from pathlib import Path

from checkpoint import DEFAULT_MOST
from memory_bus import MemoryBus, DEFAULT_EPISODES, append_episode, hits_to_dicts

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "textfile_phi"


def main():
    if not DEFAULT_MOST.is_file():
        import smoke_checkpoint

        smoke_checkpoint.train_and_save(epochs=6)

    if not (DUMP / "atoms.jsonl").exists():
        import make_dump_textfile

        make_dump_textfile.main()

    bus = MemoryBus(DEFAULT_MOST)
    n = bus.ingest_dump(DUMP)
    assert n > 2
    tok = bus.encode()
    assert tok.shape[0] == n

    hits = bus.recall_by_id("__main__", k=3)
    assert len(hits) == 3
    assert all(h.atom_id != "__main__" for h in hits)

    hits2 = bus.recall_by_text("recall bind", k=3)
    assert len(hits2) == 3

    # episode mirror (temp path żeby nie śmiecić kanonu w teście równoległym)
    ep = ROOT / "dumps" / "_test_episodes.jsonl"
    if ep.exists():
        ep.unlink()
    append_episode(
        {"kind": "test", "hits": hits_to_dicts(hits)},
        path=ep,
    )
    assert ep.is_file() and ep.read_text(encoding="utf-8").strip()
    ep.unlink()

    out = ROOT / "dumps" / "_bus_export_test"
    bus.export_dump(out)
    assert (out / "atoms.jsonl").is_file()
    # cleanup export
    for p in out.iterdir():
        p.unlink()
    out.rmdir()

    print(f"OK — MemoryBus n={n} top1={hits[0].atom_id} score={hits[0].score:.3f}")


if __name__ == "__main__":
    main()
