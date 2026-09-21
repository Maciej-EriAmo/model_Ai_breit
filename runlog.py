"""Run-dir odtwarzalności — kryterium sukcesu #5 (wzmocnione).

Jeden katalog na trening:
  runs/<UTC>_<name>/
    meta.json      start/stop, git, host, torch
    command.txt    argv
    hparams.json   hiperparametry
    loss.csv       epoch,loss,r@1,r@3,mrr  (opcjonalnie)
    metrics.json   metryki końcowe
    summary.md     krótki werdykt

Użycie:
  run = RunDir.create("phimost_learn", argv=sys.argv, hparams={...})
  run.write_curve(curve)
  run.finalize(metrics={...}, summary="...")
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_info(cwd: Path = ROOT) -> dict[str, Any]:
    info: dict[str, Any] = {"sha": None, "dirty": None, "branch": None}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["sha"] = sha
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["dirty"] = bool(dirty)
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = branch
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return info


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


@dataclass
class RunDir:
    path: Path
    name: str
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        argv: Sequence[str] | None = None,
        hparams: Mapping[str, Any] | None = None,
        runs_root: Path | None = None,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> "RunDir":
        root = Path(runs_root) if runs_root else RUNS_ROOT
        root.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
        path = root / f"{_utc_stamp()}_{safe}"
        path.mkdir(parents=True, exist_ok=False)

        meta: dict[str, Any] = {
            "name": name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "finished_utc": None,
            "status": "running",
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": _torch_version(),
            "cwd": str(Path.cwd()),
            "git": _git_info(),
            "pid": os.getpid(),
        }
        if extra_meta:
            meta.update(dict(extra_meta))

        run = cls(path=path, name=name, meta=meta)
        run._write_json("meta.json", meta)
        cmd = list(argv) if argv is not None else list(sys.argv)
        (path / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        if hparams is not None:
            run.write_hparams(hparams)
        (path / "README.md").write_text(
            "\n".join(
                [
                    f"# Run `{name}`",
                    "",
                    f"Started: {meta['started_utc']}",
                    f"Git: `{meta['git'].get('sha')}` dirty={meta['git'].get('dirty')}",
                    "",
                    "Pliki: `meta.json` · `command.txt` · `hparams.json` · "
                    "`loss.csv` · `metrics.json` · `summary.md`",
                    "",
                    "Kryterium sukcesu #5 — odtwarzalność (docs/KRYTERIA_SUKCESU_MAPA.md).",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return run

    def _write_json(self, filename: str, obj: Mapping[str, Any] | list) -> Path:
        p = self.path / filename
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return p

    def write_hparams(self, hparams: Mapping[str, Any]) -> Path:
        return self._write_json("hparams.json", dict(hparams))

    def write_curve(self, curve: Iterable[Mapping[str, Any]], *, filename: str = "loss.csv") -> Path:
        rows = list(curve)
        p = self.path / filename
        cols = ["epoch", "loss", "r@1", "r@3", "mrr"]
        lines = [",".join(cols)]
        for r in rows:
            lines.append(
                ",".join(
                    str(r.get(c, ""))
                    for c in cols
                )
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        return self._write_json("metrics.json", dict(metrics))

    def finalize(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        summary: str | None = None,
        status: str = "ok",
        curve: Iterable[Mapping[str, Any]] | None = None,
    ) -> Path:
        if curve is not None:
            self.write_curve(curve)
        if metrics is not None:
            self.write_metrics(metrics)
        self.meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.meta["status"] = status
        self._write_json("meta.json", self.meta)
        if summary is not None:
            (self.path / "summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
        elif metrics is not None:
            lines = [
                f"# {self.name}",
                "",
                f"status={status}",
                "",
                "```json",
                json.dumps(dict(metrics), indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ]
            (self.path / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        return self.path
