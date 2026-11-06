"""Post-inference scoring; hidden outcomes are only read by this module."""

import json
from pathlib import Path
import math

from ..learning import scores
from .audit import write_json, digest

DIMENSIONS = ("claim", "evidence", "framing", "interpretation")


def reliability_svg(metrics, path):
    points = [
        (b["mean_probability"], b["observed_frequency"])
        for b in metrics["reliability"]
        if b["count"]
    ]
    circles = "".join(
        f'<circle cx="{40 + 300 * x:.1f}" cy="{340 - 300 * y:.1f}" r="5" fill="#2764ae"/>'
        for x, y in points
    )
    Path(path).write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">'
        '<rect width="400" height="400" fill="white"/><path d="M40 40V340H340 M40 340L340 40" fill="none" stroke="#999"/>'
        + circles
        + '<text x="90" y="380">Mean predicted probability</text><text x="45" y="25">Observed frequency (0–1)</text></svg>'
    )


def reviewer_rows(path, pack_hash, run_hash=None):
    data = json.loads(Path(path).read_text())
    if data.get("pack_sha256") != pack_hash:
        raise ValueError("Reviewer pack version mismatch")
    if data.get("reviewer_id") and (
        data.get("run_manifest_sha256") != run_hash or type(data.get("budget")) is not int
    ):
        raise ValueError("Reviewer scores require the matching run hash and a budget")
    if not data.get("reviewer_id"):
        return {}
    rows = {}
    for row in data["cases"]:
        if row["case"] in rows:
            raise ValueError("Duplicate reviewer case")
        values = row.get("scores", {})
        if any(values.get(k) is None for k in DIMENSIONS):
            continue
        if any(type(values[k]) is not int or values[k] not in (0, 1, 2) for k in DIMENSIONS):
            raise ValueError("Scores must be integers 0, 1 or 2")
        intensity = row.get("emotional_language")
        if intensity is not None and (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not math.isfinite(intensity)
            or not 0 <= intensity <= 1
        ):
            raise ValueError("Invalid reviewer semantic intensity")
        if type(row.get("fabrication")) is not bool or not row.get("rationale"):
            raise ValueError("Completed scores require fabrication judgment and rationale")
        rows[row["case"]] = row
    return rows


def score_run(output, cases, reviewers=None):
    output, cases = Path(output), Path(cases)
    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "data-audit/audit.json").read_text())
    eligibility = {r["case"]: r["heldout_eligible"] for r in audit["cases"]}
    latest = {}
    for path in sorted((output / "attempts").glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["case"], row["budget"])
        if key not in latest or latest[key]["attempt"] < row["attempt"]:
            latest[key] = row
    results = {}
    for budget in sorted({r["budget"] for r in latest.values()}):
        rows = [r for r in latest.values() if r["budget"] == budget]
        for group, selected in [
            ("all_development", rows),
            ("overlap_screened", [r for r in rows if eligibility[r["case"]]]),
        ]:
            measured = []
            for row in selected:
                if not row.get("prediction"):
                    continue
                p = row["prediction"]
                reference = json.loads((cases / row["case"] / "EVALUATOR_ONLY.json").read_text())
                target = reference[f"viral_target_{p['size_threshold']}"]
                measured.append((int(target), p["viral_probability"], p["decision_cutoff"]))
            if measured:
                cutoffs = {m[2] for m in measured}
                if len(cutoffs) != 1:
                    raise ValueError("Do not pool differing decision cutoffs")
                metrics = scores([m[0] for m in measured], [m[1] for m in measured], measured[0][2])
                metrics.update(
                    expected=len(selected),
                    complete_predictions=len(measured),
                    completed_answers=sum(r["status"] == "completed" for r in selected),
                    interpretation="Exploratory; lexical overlap screening is not event adjudication",
                )
                results[f"{budget}-{group}"] = metrics
                reliability_svg(metrics, output / f"reliability-{budget}-{group}.svg")
    human = {"status": "PENDING independent annotations"}
    if reviewers:
        headers = [json.loads(Path(p).read_text()) for p in reviewers]
        if headers[0].get("reviewer_id") and headers[0].get("reviewer_id") == headers[1].get(
            "reviewer_id"
        ):
            raise ValueError("Two distinct independent reviewers are required")
        if len({h.get("budget") for h in headers}) != 1:
            raise ValueError("Reviewers must assess the same Thinking budget")
        human["budget"] = headers[0].get("budget")
        a, b, adjudicated = [
            reviewer_rows(p, manifest["pack_hash"], digest(manifest)) for p in reviewers
        ]
        common = set(a) & set(b)
        human["agreement"] = {
            d: sum(a[k]["scores"][d] == b[k]["scores"][d] for k in common) / len(common)
            if common
            else None
            for d in DIMENSIONS
        }
        human["independently_scored_cases"] = len(common)
        human["adjudicated"] = {}
        for key, row in adjudicated.items():
            if key not in common:
                raise ValueError("Adjudication requires both independent reviews")
            total = sum(row["scores"].values())
            human["adjudicated"][key] = {
                "score": total,
                "pass": total >= 7 and min(row["scores"].values()) > 0 and not row["fabrication"],
            }
        if human["adjudicated"]:
            human["status"] = (
                "PARTIAL" if len(human["adjudicated"]) < len(eligibility) else "COMPLETE"
            )
    report = {
        "prediction_metrics": results,
        "human_review": human,
        "policy": "No inferred factual verdicts from rumor labels; assisted follow-ups must be separate runs.",
    }
    write_json(output / "scores.json", report)
    return report
