"""Create 15 user-workflow evaluations from the prepared native Twitter16 corpus."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/backend"))
from sentinel_x.graphs import native_twitter, observed_prefix

SCENARIOS = [
    (
        "Early spread assessment",
        "Assess whether this post is likely to spread widely from its first hour of propagation. Explain the prediction and what it does and does not tell me.",
    ),
    (
        "Claim and evidence review",
        "Identify the main factual claim, assess the available evidence, and separately estimate its early virality. Cite the evidence you use.",
    ),
    (
        "Fast growth without truth assumptions",
        "Does the early propagation suggest this will spread widely? Does that tell us whether the post is true? Give separate conclusions.",
    ),
    (
        "Limited spread without dismissal",
        "Assess the claim and likely spread. Explain whether limited early spread is a reason to dismiss the claim.",
    ),
    (
        "Emotional framing",
        "Describe the emotional framing with short quotations from the post. Explain how the selected model uses semantic information alongside propagation to predict spread.",
    ),
    (
        "Evidence insufficient to decide",
        "Check the claim against the available evidence. If you cannot establish the answer, explain what is missing. Still report the separate virality assessment.",
    ),
    (
        "Research briefing",
        "Write a short research briefing: claim, evidence assessment, early virality result, and limitations. Keep factual confidence separate from spread probability.",
    ),
    (
        "Numerical interpretation",
        "Report the early virality probability and decision. Explain the cutoff, observation period, and what reaching the target means in this dataset.",
    ),
    (
        "Rumor framing and uncertainty",
        "Analyze this post without assuming that calling something a rumor makes it false. Assess evidence and predict spread separately.",
    ),
    (
        "Grounded explanation",
        "Explain the prediction using the observed propagation and the post content. Distinguish measured facts from hypotheses about why people might share it.",
    ),
    (
        "Scope interpretation",
        "Give the early virality result. Explain what the saved Scope measurements can tell us about the language-model response, including what they cannot establish.",
    ),
    (
        "Claim decomposition",
        "Separate any distinct factual claims in the source post. Assess each against available evidence, then report one virality prediction for the selected cascade.",
    ),
    (
        "Decision support",
        "I am reviewing this post for further investigation. Summarize the evidence, likely recorded spread, and what additional information I should obtain before making a factual judgment.",
    ),
    (
        "Conflicting or irrelevant evidence",
        "Assess whether retrieved passages actually address this specific claim. Describe any conflict or gaps, and separately provide the early virality prediction.",
    ),
    (
        "Plain-language final assessment",
        "Explain to a nontechnical researcher what this post claims, whether the evidence warrants believing it, and how likely its cascade is to reach the recorded-spread target. Be precise about uncertainty.",
    ),
]


def prepare(suite, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    raw = suite / "original/twitter16"
    graphs = [g for g in native_twitter(raw) if "error" not in g]
    checkpoint = json.loads(Path("model weights/fe44637dfc4d49fe9cd402c4ce6cc21d.json").read_text())
    excluded = set(checkpoint["splits"]["train"]) | set(checkpoint["splits"]["validation"])
    # Some checkpoints store hashes instead of source IDs; record the limitation.
    used = set()
    manifest = []
    labels = ["false", "true", "unverified", "non-rumor"]
    for index, (title, prompt) in enumerate(SCENARIOS):
        label = labels[index % 4]
        viral = (index // 4) % 2 == 0
        candidates = [
            g
            for g in graphs
            if g["cascade_id"] not in used
            and g["cascade_id"] not in excluded
            and g["rumor_label"] == label
            and (len(g["nodes"]) >= 464) == viral
        ]
        if not candidates:
            raise ValueError(f"No candidate for {title}: {label}, viral={viral}")
        # Spread selected sizes across each eligible stratum, with deterministic ties.
        candidates.sort(key=lambda g: (len(g["nodes"]), g["cascade_id"]))
        graph = candidates[min(len(candidates) - 1, int(len(candidates) * ((index % 3) + 1) / 4))]
        ident = graph["cascade_id"]
        used.add(ident)
        folder = output / f"case-{index + 1:02d}"
        inputs = folder / "input"
        (inputs / "tree").mkdir(parents=True)
        shutil.copyfile(raw / "tree" / f"{ident}.txt", inputs / "tree" / f"{ident}.txt")
        (inputs / "source_tweets.txt").write_text(f"{ident}\t{graph['source_text']}\n")
        (inputs / "label.txt").write_text("")
        (folder / "USER_TASK.md").write_text(
            f"# Case {index + 1:02d}: {title}\n\nSource ID: {ident}\n\nSource post:\n\n> {graph['source_text']}\n\nImport `input/`, select this cascade, and ask:\n\n{prompt}\n\nSave the complete response and research export for evaluation.\n"
        )
        prefix = observed_prefix(graph, 60)
        expected = {
            "source_id": ident,
            "original_rumor_label": graph["rumor_label"],
            "final_recorded_nodes": len(graph["nodes"]),
            "observed_nodes_60min": len(prefix["nodes"]),
            "viral_target_464": len(graph["nodes"]) >= 464,
            "viral_target_475": len(graph["nodes"]) >= 475,
            "expected_probability": None,
            "expected_factual_verdict": "Human-adjudicated from the fixed evidence pack; never infer from rumor label alone",
            "semantic_reference": "Two reviewers independently annotate framing and emotional-language intensity from the source text before seeing outputs",
            "focus": title,
            "training_overlap_check": "Excluded literal source IDs in shipped train/validation lists; cross-dataset semantic duplicates still require review",
        }
        (folder / "EVALUATOR_ONLY.json").write_text(json.dumps(expected, indent=2) + "\n")
        (folder / "SCORECARD.md").write_text("""# Evaluator scorecard

Run/model/checkpoint/evidence-pack IDs:

- Virality predicted class / target / correct (0 or 1):
- Probability / saved cutoff / Brier error (p - target)^2:
- Claim understanding (0–2), reason and quotation:
- Evidence and calibrated uncertainty (0–2), passage IDs and reason:
- Semantic/framing analysis (0–2), reference annotations and reason:
- Interpretation and usefulness (0–2), reason:
- Any invented fact/citation/probability or truth–spread conflation:
- Complete / incomplete / not run, with reason:
- Case LLM PASS/FAIL and prediction CORRECT/INCORRECT:
- Reviewer IDs / adjudication notes:

Apply the shared rubric in docs/TWITTER16_VALIDATION.md. Do not score preparation
or application mechanics as model quality. Keep this file and expected outcomes
out of model context and retrieval.
""")
        manifest.append({"case": folder.name, "focus": title, "source_id": ident})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} unique workflow cases in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("datasets/twitter16-suite"))
    parser.add_argument("--output", type=Path, default=Path("datasets/twitter16-model-cases"))
    args = parser.parse_args()
    prepare(args.suite, args.output)
