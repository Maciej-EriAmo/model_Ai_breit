"""
Buduje realistyczny dump Φ: „plik tekstowy” Karmazyn ≠ płaski .txt.

Scenariusz:
  Bubble doc/notes zawiera:
    - __main__ (HOT) — pełny tekst
    - title, tags (HOT/WARM)
    - sekcje outline (WARM — bez payload, tylko S/E/T)
    - peer atoms w pod-bańkach (depth 1–2)
  Bind: __main__ ↔ sekcje; parent bubbles.

Zapis: dumps/textfile_phi/
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "textfile_phi"

TEXT = """# Notatka: Most i przestrzeń Φ

KarmazynOs nie jest systemem plików Unix.
Informacja = stabilizacja(H ∘ P ∘ A).
Atom A = (S, E, T). Plik tekstowy w Φ to wiele atomów.

## Most Lorentza

Zamek: A = γ C / (γ² + ΔE²).
Obserwator (tracer) niesie geometrię energii.
Bez ΔE most degeneruje do Softmax.

## Superpozycja

EMA-S odrzucona (2026-09-08/09).
Baseline: czysty most + tracer.

## Recall

Encoder Most rankinguje sąsiadów bind w warstwie depth.
"""


def main():
    DUMP.mkdir(parents=True, exist_ok=True)

    # sekcje z tekstu (outline) — WARM, bez pełnego payload
    sections = [
        ("sec_intro", "Notatka: Most i przestrzeń Φ", 0, "intro"),
        ("sec_most", "Most Lorentza", 1, "most"),
        ("sec_super", "Superpozycja", 1, "superpozycja"),
        ("sec_recall", "Recall", 1, "recall"),
    ]

    atoms = []
    edges = []

    # --- depth 0: dokument ---
    atoms.append(
        {
            "id": "__main__",
            "S": "doc/notes/__main__",
            "E": "text/markdown; note about Most and Phi space",
            "T": 1.0,
            "state": "HOT",
            "bubble": "doc/notes",
            "depth": 0,
            "payload": TEXT,
            "mazur_tracer": 1.0,
            "topic": "doc",
        }
    )
    atoms.append(
        {
            "id": "title",
            "S": "doc/notes/title",
            "E": "title: Notatka Most i przestrzeń Φ",
            "T": 0.95,
            "state": "HOT",
            "bubble": "doc/notes",
            "depth": 0,
            "payload": "Notatka: Most i przestrzeń Φ",
            "mazur_tracer": 0.95,
            "topic": "doc",
        }
    )
    atoms.append(
        {
            "id": "tags",
            "S": "doc/notes/tags",
            "E": "tags: most, lorentz, phi, karmazyn, recall",
            "T": 0.85,
            "state": "WARM",
            "bubble": "doc/notes",
            "depth": 0,
            "payload": None,  # WARM — index only
            "mazur_tracer": 0.8,
            "topic": "doc",
        }
    )

    for sid, heading, depth, topic in sections:
        atoms.append(
            {
                "id": sid,
                "S": f"doc/notes/outline/{sid}",
                "E": f"section/{topic}: {heading}",
                "T": 0.9 - 0.1 * depth,
                "state": "WARM",
                "bubble": "doc/notes",
                "depth": depth,
                "payload": None,
                "mazur_tracer": 0.7 - 0.1 * depth,
                "topic": topic,
            }
        )
        edges.append(
            {"kind": "bind", "src": "__main__", "dst": sid, "name": f"outline:{topic}"}
        )
        edges.append(
            {"kind": "bind", "src": sid, "dst": "__main__", "name": f"outline:{topic}"}
        )

    edges.append({"kind": "bind", "src": "__main__", "dst": "title", "name": "has_title"})
    edges.append({"kind": "bind", "src": "title", "dst": "__main__", "name": "has_title"})
    edges.append({"kind": "bind", "src": "__main__", "dst": "tags", "name": "has_tags"})

    # --- depth 1: peer bańka lab (powiązane pojęcia) ---
    peers = [
        ("peer_bridge", "lab/most", "bridge formula softplus", "most", 1, 0.7),
        ("peer_tracer", "lab/tracer", "observer DeltaE n-D", "most", 1, 0.65),
        ("peer_softmax", "lab/baseline", "softmax twin control", "baseline", 1, 0.55),
        ("peer_ema", "lab/rejected", "EMA-S rejected negative", "superpozycja", 1, 0.5),
        ("peer_recall", "lab/recall", "InfoNCE bind same depth", "recall", 1, 0.6),
        ("peer_flat", "lab/ablation", "flat payload fails", "recall", 1, 0.45),
    ]
    for pid, S, E, topic, depth, T in peers:
        atoms.append(
            {
                "id": pid,
                "S": S,
                "E": E,
                "T": T,
                "state": "WARM" if T < 0.7 else "HOT",
                "bubble": "lab/phi",
                "depth": depth,
                "payload": (f"note:{E}" if T >= 0.7 else None),
                "mazur_tracer": T,
                "topic": topic,
            }
        )
        # bind peer ↔ matching section by topic
        for sid, _, _, stopic in sections:
            if stopic == topic or (topic == "baseline" and stopic == "most"):
                edges.append(
                    {"kind": "bind", "src": sid, "dst": pid, "name": f"rel:{topic}"}
                )
                edges.append(
                    {"kind": "bind", "src": pid, "dst": sid, "name": f"rel:{topic}"}
                )

    edges.append({"kind": "parent", "src": "lab/phi", "dst": "doc/notes", "name": ""})

    # --- depth 2: COLD archive crumbs ---
    for i, (cid, topic) in enumerate(
        [("arch_old_s", "superpozycja"), ("arch_old_r", "recall")]
    ):
        atoms.append(
            {
                "id": cid,
                "S": f"archive/{cid}",
                "E": f"cold crumb about {topic}",
                "T": 0.25,
                "state": "COLD",
                "bubble": "archive",
                "depth": 2,
                "payload": None,
                "mazur_tracer": 0.2,
                "topic": topic,
            }
        )
        edges.append({"kind": "parent", "src": "archive", "dst": "lab/phi", "name": ""})

    # same-depth binds (Faza D gate — geometria Most jak w C)
    same_depth_binds = [
        ("peer_bridge", "peer_tracer"),
        ("peer_tracer", "peer_softmax"),
        ("peer_bridge", "peer_softmax"),
        ("peer_recall", "peer_flat"),
        ("peer_ema", "peer_flat"),
        ("peer_recall", "peer_ema"),
        ("sec_most", "sec_super"),
        ("sec_super", "sec_recall"),
        ("sec_most", "sec_recall"),
        ("title", "tags"),
        ("__main__", "title"),
        ("__main__", "tags"),
        ("arch_old_s", "arch_old_r"),
    ]
    for a, b in same_depth_binds:
        edges.append({"kind": "bind", "src": a, "dst": b, "name": "same_depth"})
        edges.append({"kind": "bind", "src": b, "dst": a, "name": "same_depth"})

    manifest = {
        "name": "textfile_phi",
        "version": "1.0",
        "description": "Plik tekstowy jako przestrzeń Φ (nie Unix .txt)",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "bubbles": sorted({a["bubble"] for a in atoms}),
        "source": "synthetic-realistic Karmazyn-style (offline; not live Store)",
        "canon": "PhiJSON dump; atomstore not canonical",
    }

    (DUMP / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (DUMP / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (DUMP / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # case study doc
    case = DUMP / "CASE_TEXT.md"
    hot_payload_atoms = [a for a in atoms if a.get("payload")]
    warm_no_pay = [a for a in atoms if a["state"] == "WARM"]
    case.write_text(
        "\n".join(
            [
                "# Case study: plik tekstowy nie jest płaski",
                "",
                f"**Dump:** `{DUMP}`",
                "",
                "## Unix fiction",
                "",
                "Jeden plik `notatka.md` = ciąg bajtów / tokenów.",
                "",
                "## Φ (ten dump)",
                "",
                f"- atomów: **{len(atoms)}**",
                f"- krawędzi: **{len(edges)}**",
                f"- HOT z payload: {[a['id'] for a in hot_payload_atoms]}",
                f"- WARM bez payload: {[a['id'] for a in warm_no_pay]}",
                f"- bańki: {manifest['bubbles']}",
                "",
                "### `__main__`",
                "",
                "Pełny tekst notatki (HOT). Peer atoms niosą outline, tagi, lab,",
                "archive — **osie T/state/depth**, nie tylko bajty.",
                "",
                "### Wniosek",
                "",
                "Tokenizacja samego `__main__.payload` gubi WARM sekcje, bind do labu",
                "i geometrię depth. PhiMost czyta graf, nie `.txt`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Zapisano dump: {DUMP}")
    print(f"  atoms={len(atoms)} edges={len(edges)}")
    print(f"  case: {case}")


if __name__ == "__main__":
    main()
