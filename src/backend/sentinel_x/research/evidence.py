"""Freeze FEVER-only retrieval candidates and prepare independent reviewer forms."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone

from .audit import digest, write_json


def freeze_fever(index, cases, output):
    index, cases, output = Path(index).resolve(), Path(cases), Path(output)
    if output.exists():
        raise ValueError("Choose a new evidence-pack directory; frozen packs are immutable")
    with closing(sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        partial = bool(db.execute("SELECT 1 FROM index_progress LIMIT 1").fetchone())
        passages, selections = {}, {}
        for case in json.loads((cases / "manifest.json").read_text()):
            text = (
                (cases / case["case"] / "input/source_tweets.txt")
                .read_text()
                .split("\t", 1)[1]
                .strip()
            )
            words = sorted(
                set(re.findall(r"\w+", text.casefold()))
                - {
                    "url",
                    "https",
                    "http",
                    "what",
                    "when",
                    "that",
                    "this",
                    "have",
                    "been",
                    "with",
                    "from",
                    "they",
                    "their",
                    "would",
                    "which",
                    "will",
                    "were",
                    "there",
                    "about",
                    "just",
                },
                key=lambda w: (-len(w), w),
            )
            query = " OR ".join('"' + w + '"' for w in words[:8] if len(w) > 3)
            # Only FEVER Wikipedia sentences with imported file provenance are eligible.
            rows = db.execute(
                """SELECT c.*, f.hash AS file_hash FROM search
                JOIN chunks c ON c.id=search.rowid LEFT JOIN files f ON f.path=c.path
                WHERE search MATCH ? ORDER BY rank LIMIT 200""",
                (query,),
            ).fetchall()
            selected = []
            for row in rows:
                provenance = json.loads(row["provenance"])
                if (
                    not {"page", "sentence"} <= provenance.keys()
                    or "fever" not in row["path"].casefold()
                ):
                    continue
                ident = f"F{row['id']}"
                selected.append(ident)
                passages[ident] = {
                    "id": ident,
                    "corpus": "FEVER",
                    "text": row["text"],
                    "sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "provenance": provenance,
                    "source_path": row["path"],
                    "indexed_file_sha256": row["file_hash"],
                }
                if len(selected) == 40:
                    break
            selections[case["case"]] = selected
    if not passages:
        raise ValueError("No indexed FEVER sentences found; index FEVER before freezing a pack")
    pack = {
        "version": 1,
        "corpus": "FEVER",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval": "Frozen keyword candidate subset; not the entire FEVER corpus",
        "source_index_partial": partial,
        "cases": selections,
        "passages": list(passages.values()),
    }
    pack["content_sha256"] = digest(pack)
    write_json(output / "pack.json", pack)
    for reviewer in ("reviewer-a", "reviewer-b", "adjudication"):
        write_json(
            output / "reviewers" / (reviewer + ".json"),
            {
                "reviewer_id": None,
                "run_manifest_sha256": None,
                "budget": None,
                "pack_sha256": pack["content_sha256"],
                "cases": [
                    {
                        "case": case,
                        "claims": [],
                        "relevant_passage_ids": [],
                        "factual_verdict": None,
                        "emotional_language": None,
                        "scores": {
                            k: None for k in ("claim", "evidence", "framing", "interpretation")
                        },
                        "fabrication": None,
                        "rationale": None,
                    }
                    for case in selections
                ],
            },
        )
    return pack


def validate_pack(path):
    pack = json.loads(Path(path).read_text())
    expected = pack.pop("content_sha256")
    if digest(pack) != expected or pack.get("corpus") != "FEVER":
        raise ValueError("Evidence pack hash/corpus mismatch")
    ids = set()
    for p in pack["passages"]:
        if (
            p["id"] in ids
            or p["corpus"] != "FEVER"
            or hashlib.sha256(p["text"].encode()).hexdigest() != p["sha256"]
        ):
            raise ValueError("Invalid or duplicate FEVER passage")
        if not {"page", "sentence"} <= p["provenance"].keys():
            raise ValueError("FEVER Wikipedia provenance required")
        ids.add(p["id"])
    if any(not set(values) <= ids for values in pack["cases"].values()):
        raise ValueError("Unknown passage reference")
    pack["content_sha256"] = expected
    return pack


def install_pack(service, pack):
    # Copy only allowlisted evidence fields, never reviewer notes or labels.
    with closing(service.retrieval.connection()) as db, db:
        if db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
            raise ValueError("Evidence installation requires a fresh isolated index")
        for p in pack["passages"]:
            db.execute(
                "INSERT INTO chunks(path,key,text,provenance) VALUES(?,?,?,?)",
                (
                    "FEVER/" + p["provenance"]["page"],
                    p["id"],
                    p["text"],
                    json.dumps(
                        {
                            **p["provenance"],
                            "pack_passage_id": p["id"],
                            "pack_sha256": pack["content_sha256"],
                            "corpus": "FEVER",
                        }
                    ),
                ),
            )
