"""
U1 — żywy eksport z Karmazyn Store (Mazur bridge) → BubbleFS → PhiJSON.

Nie wymaga pełnego KarmazynOS.bubbles / GUI.
Używa python substrate + LorentzBridge (ten sam tor co mazur_crystal.test_bridge).

  python export_karmazyn_live.py
  python export_karmazyn_live.py --out dumps/karmazyn_live_bbl
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KARMAZYN = Path(r"C:\Users\drwis\KarmazynOs")
DEFAULT_OUT = ROOT / "dumps" / "karmazyn_live_bbl"


def _setup_path() -> None:
    for p in (
        KARMAZYN,
        KARMAZYN / "archiwum" / "kernel_python",
        KARMAZYN / "software",
        KARMAZYN / "native",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ.setdefault("KARMAZYN_SUBSTRATE", "python")
    os.environ.setdefault("KARMAZYN_MAZUR", "1")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def seed_store(*, rich: bool = True):
    """
    Żywa przestrzeń Φ-like ze Store Mazur.
    `rich=True` ≈ 28 atomów + kilka hologramów (dane pod fine-tune).
    """
    from karmazyn_substrate import Store
    from mazur_crystal import Tracer, attach_lorentz_bridge, set_tracer

    raw = Store(thermal=True)
    s = attach_lorentz_bridge(raw, mrc=True)

    ctx = s.create_atom(
        "ctx",
        "c",
        "Most Lorentz active context — softplus tracer",
        T=100.0,
        tracer=Tracer(energy=1.0),
    )
    notes = s.bubble_new("notes")
    lab = s.bubble_new("lab")
    phys = s.bubble_new("physics")
    arch = s.bubble_new("archive")
    s.set_root(notes)
    notes.bind("c", s.get_atom(ctx))
    s.set_context(ctx)

    # (id, S, E, T, energy, depth, bubble)
    peers = [
        ("main", "m", "Notatka: Most i przestrzeń Φ", 80.0, 1.0, 0, notes),
        ("intro", "m", "Wstęp — zamek Lorentza i DeltaE", 70.0, 1.1, 0, notes),
        ("toc", "m", "Spis: teza, most, recall, fizyka, zakazy", 65.0, 1.0, 0, notes),
        ("thesis", "m", "Teza: Most powstaje dopiero przy niezerowym DeltaE", 75.0, 0.95, 0, notes),
        ("glossary", "m", "Słownik: content x, tracer tau, softplus, ZERO", 60.0, 1.05, 0, notes),
        ("bridge", "m", "bridge formula softplus×2 + tracer", 58.0, 1.1, 1, lab),
        ("recall", "m", "InfoNCE bind same depth recall ranking", 55.0, 0.9, 1, lab),
        ("softmax", "m", "Softmax twin bez geometrii obserwatora", 52.0, 1.3, 1, lab),
        ("zero", "m", "ZERO tracer — degeneracja Mostu bez tau", 50.0, 1.4, 1, lab),
        ("encoder", "m", "PhiBridgeEncoder checkpoint load/save M1", 54.0, 1.0, 1, lab),
        ("bus", "m", "MemoryBus ingest encode recall fasada", 53.0, 1.05, 1, lab),
        ("mask", "m", "mask padding w InfoNCE i uwadze", 51.0, 1.15, 1, lab),
        ("solar", "m", "Układ Słoneczny star rocky giant + Moon", 48.0, 0.85, 1, phys),
        ("three", "m", "Trzy ciała Verlet warstwy masy", 47.0, 0.88, 1, phys),
        ("kepler", "m", "Energia Kepler w tracerze solar bench", 46.0, 0.92, 1, phys),
        ("nbody", "m", "Walidacja n-ciał Most bije Softmax", 49.0, 0.9, 1, phys),
        ("ema", "m", "EMA-S rejected negative result", 35.0, 2.5, 2, arch),
        ("obs", "m", "Obserwator w superpozycji FAIL 0/4", 32.0, 2.8, 2, arch),
        ("flat", "m", "Unix flat payload gorszy niż Φ-graph", 30.0, 2.2, 2, arch),
        ("archive", "m", "stary fragment COLD archive", 18.0, 4.0, 2, arch),
    ]
    if rich:
        peers += [
            ("canon", "m", "config.CANON softplus n_dim=6 mazur dim5", 56.0, 1.0, 1, lab),
            ("dump", "m", "PhiJSON atoms edges manifest offline dump", 54.0, 1.08, 1, lab),
            ("gen", "m", "gen head propose E warunkowany tracerem", 50.0, 1.2, 1, lab),
            ("lambda", "m", "lambda_neural OFF default Mazur blend", 48.0, 1.25, 1, lab),
            ("moon", "m", "Księżyc rocky nie giant — fix solar", 44.0, 0.95, 1, phys),
            ("verlet", "m", "Integracja Verlet three-body lab", 43.0, 0.97, 1, phys),
            ("warm", "m", "WARM bez payload — model musi działać na S/E/T", 40.0, 1.5, 2, arch),
            ("tomb", "m", "TOMB poza feed NN — GC Mazur", 12.0, 5.0, 2, arch),
        ]

    ids = {"ctx": ctx}
    bubbles = {"notes": notes, "lab": lab, "physics": phys, "archive": arch}
    for aid, S, E, T, energy, depth, bub in peers:
        i = s.create_atom(aid, S, E, T=T, tracer=Tracer(energy=energy, level=depth))
        ids[aid] = i
        atom = s.get_atom(i)
        set_tracer(atom, Tracer(energy=energy, level=depth))
        bub.bind(aid, atom)

    # grupy hologramów = bind same-topic / same-depth (dane pod InfoNCE)
    holo_groups = {
        "h_notes": ["main", "intro", "toc", "thesis", "glossary"],
        "h_most": ["bridge", "recall", "softmax", "zero", "encoder", "canon"],
        "h_fasada": ["bus", "mask", "dump", "gen", "lambda"],
        "h_phys": ["solar", "three", "kepler", "nbody", "moon", "verlet"],
        "h_neg": ["ema", "obs", "flat", "archive", "warm", "tomb"],
        "h_bridge_main": ["main", "bridge", "recall", "thesis"],
    }
    return s, ids, bubbles, holo_groups


def store_to_bubblefs(
    s,
    ids: dict,
    bubbles: dict,
    out: Path,
    holo_groups: dict[str, list[str]] | None = None,
) -> Path:
    """Atom + bubble bindings → layout BubbleFS (plaintext .bbl)."""
    bdir = out / "bubbles"
    hdir = out / "holograms"
    bdir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)

    depth_of_bubble = {
        "notes": 0,
        "lab": 1,
        "physics": 1,
        "archive": 2,
    }
    atom_depth: dict[str, int] = {}
    for blabel, bub in bubbles.items():
        depth = depth_of_bubble.get(blabel, 1)
        binds = getattr(bub, "bindings", None) or {}
        if callable(binds):
            binds = binds()
        if not isinstance(binds, dict):
            continue
        for _name, atom in binds.items():
            aid = str(getattr(atom, "id", atom))
            atom_depth[aid] = depth

    # Geometria Φ: mazur_tracer = klaster hologramu (bind-group).
    # Softmax tego nie widzi; Most używa ΔE — inaczej live recall ≈ los.
    label_group: dict[str, int] = {}
    groups_list = list((holo_groups or {}).items())
    for gi, (_hid, labs) in enumerate(groups_list):
        for lab in labs:
            label_group.setdefault(lab, gi)
    n_grp = max(len(groups_list), 1)

    bbls = []
    inode = 1
    for req_id, real_id in ids.items():
        a = s.get_atom(real_id)
        if a is None:
            continue
        from mazur_crystal import get_tracer

        tr = get_tracer(a)
        energy = float(getattr(tr, "energy", 1.0) or 1.0)
        level = int(getattr(tr, "level", 0) or 0)
        text = str(a.E or "")
        depth = atom_depth.get(str(a.id), level)
        parent = None
        if depth >= 1:
            parent = ids.get("main") or ids.get("ctx")
        gi = label_group.get(req_id, n_grp - 1)
        # środek binu grupy + lekki offset z energy (nie kasuje klastra)
        mazur = (gi + 0.5) / n_grp
        mazur = max(0.0, min(1.0, 0.85 * mazur + 0.15 * (1.0 / (1.0 + abs(energy - 1.0)))))
        meta = {
            "depth": depth,
            "state": "HOT" if float(a.T) >= 40 else "WARM",
            "requested_id": req_id,
            "holo_group": gi,
            "mazur_tracer": mazur,
        }
        if parent and str(parent) != str(a.id):
            meta["parent"] = str(parent)
        bbl = {
            "id": str(a.id),
            "label": req_id,
            "inode": inode,
            "epoch_born": 1,
            "recall_count": max(0, int(float(a.T) / 20.0)),
            "consolidated_from": [],
            "metadata": meta,
            "revoked": False,
            "content_encrypted": False,
            "content_b64": _b64(text.encode("utf-8")),
            "fingerprint_b64": _b64(f"{a.id}:{a.S}".encode()),
            "S_struct": [float(hash(str(a.S)) % 1000) / 1000.0, depth / 2.0, 0.1],
            "S_sem": [float(hash(str(a.E)[:40]) % 1000) / 1000.0, energy / 10.0, 0.2],
        }
        if float(a.T) < 25:
            bbl["decay_start_epoch"] = 10
            bbl["decay_rate"] = 0.35
            bbl["metadata"]["state"] = "WARM"
        if req_id == "tomb":
            bbl["revoked"] = True
            bbl["metadata"]["state"] = "TOMB"
        (bdir / f"{a.id}.bbl").write_text(
            json.dumps(bbl, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        bbls.append(bbl)
        inode += 1

    label_set = {b["label"] for b in bbls}
    groups = holo_groups or {
        "h_most": ["main", "bridge", "recall", "intro"],
    }
    n_h = 0
    for hid, labs in groups.items():
        use = [x for x in labs if x in label_set]
        if len(use) < 2:
            continue
        hgm = {
            "id": hid,
            "topic": hid,
            "proto": [0.2, 0.1],
            "generators": [[0.1], [0.2]],
            "weights": [0.5, 0.5],
            "bubble_labels": use,
            "epoch_created": 1,
            "decay_rate": 0.0,
            "metadata": {"source": "export_karmazyn_live"},
        }
        (hdir / f"{hid}.hgm").write_text(
            json.dumps(hgm, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        n_h += 1

    man = {
        "bubblefs_version": "1.0.0",
        "karmazyn_version": "live-store-export",
        "epoch": 1,
        "dim": 8,
        "t_vacuum()": 0.1,
        "n_bubbles": len(bbls),
        "n_holograms": n_h,
        "encrypted": False,
        "include_phi_vectors": False,
        "source": "mazur_crystal+Store",
        "integrity_sha256": "live-export-unsigned",
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--phi-out", type=Path, default=ROOT / "dumps" / "from_karmazyn_live")
    ap.add_argument("--recall", action="store_true", help="MemoryBus recall po imporcie")
    args = ap.parse_args()

    _setup_path()
    s, ids, bubbles, holo = seed_store(rich=True)
    out = store_to_bubblefs(s, ids, bubbles, args.out, holo_groups=holo)
    print(f"BubbleFS live → {out}  atoms≈{len(ids)}")

    # import adapter PhiMost
    sys.path.insert(0, str(ROOT))
    from bubblefs_import import bubblefs_to_graph, export_graph_as_dump

    man, g = bubblefs_to_graph(out)
    phi = export_graph_as_dump(
        g, args.phi_out, description="live Karmazyn Store → BubbleFS → PhiJSON"
    )
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    print(f"PhiJSON → {phi}  atoms={len(g.atoms)} bind={n_bind}")

    if args.recall:
        from checkpoint import DEFAULT_MOST
        from memory_bus import MemoryBus

        if not DEFAULT_MOST.is_file():
            import smoke_checkpoint

            smoke_checkpoint.train_and_save(epochs=6)
        bus = MemoryBus(DEFAULT_MOST)
        bus.ingest_dump(phi)
        # prefer label main / __main__ / first
        labels = [a.atom_id for a in g.atoms]
        q = "main" if "main" in labels else labels[0]
        # atom_id in export is Store id (may be 'main')
        try:
            hits = bus.recall_by_id(q, k=5)
        except KeyError:
            hits = bus.recall_by_index(0, k=5)
        print("recall:", [(h.atom_id, round(h.score, 3), h.E[:40]) for h in hits])


if __name__ == "__main__":
    main()
