"""Kryterium #5 — RunDir zapisuje meta/command/hparams/loss/metrics."""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

import json
import tempfile
from pathlib import Path

from runlog import RunDir


def test_rundir_create_finalize(tmp_root: Path | None = None) -> None:
    cleanup = False
    if tmp_root is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="runlog_test_"))
        cleanup = True
    try:
        run = RunDir.create(
            "unit_test_runlog",
            argv=["python", "-m", "pytest", "test_runlog"],
            hparams={"epochs": 2, "lr": 1e-3},
            runs_root=tmp_root,
            extra_meta={"criterion": "success#5"},
        )
        assert run.path.is_dir()
        assert run.path.parent == tmp_root
        assert (run.path / "meta.json").is_file()
        assert (run.path / "command.txt").is_file()
        assert (run.path / "hparams.json").is_file()
        curve = [
            {"epoch": 1, "loss": 1.0, "r@1": 0.1, "r@3": 0.2, "mrr": 0.15},
            {"epoch": 2, "loss": 0.5, "r@1": 0.4, "r@3": 0.6, "mrr": 0.5},
        ]
        run.finalize(
            metrics={"r@1": 0.4, "ok": True},
            summary="unit ok",
            status="ok",
            curve=curve,
        )
        meta = json.loads((run.path / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "ok"
        assert meta["finished_utc"]
        assert meta.get("git") is not None
        loss = (run.path / "loss.csv").read_text(encoding="utf-8").strip().splitlines()
        assert loss[0].startswith("epoch,loss")
        assert len(loss) == 3
        metrics = json.loads((run.path / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["r@1"] == 0.4
        assert "unit ok" in (run.path / "summary.md").read_text(encoding="utf-8")
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_rundir_create_finalize()
    print("OK test_runlog")
