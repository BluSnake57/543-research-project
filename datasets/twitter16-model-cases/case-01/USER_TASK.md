# Case 01: Early spread assessment

Source ID: 643103859729166337

Source post:

> unreal: watch what happens when desperate muslim refugees are offered red cross food URL URL

Prompt version: 2 (expanded reporting requirements after inspection of a pilot run).
Freeze this version before a new evaluation; report version-1 and version-2 results separately.

Open a fresh conversation. Choose **Import posts → Twitter15 / Twitter16 input folder…**,
select `input/`, and keep only this cascade attached. Record the selected checkpoint,
model/SAE revisions, evidence-pack version, and Thinking setting (default On).
Do not attach evaluator references to the conversation or evidence index.

## Primary prompt

Copy the source post above and the following request into the conversation:

Assess the attached source post and its likely recorded spread using only its first hour of propagation. Complete all of the following in one response:

1. Identify the central claim and any ambiguity. Do not invent the contents of the URL placeholders or infer what happened in an unavailable video.
2. Report the selected checkpoint/model, observation window, number of observed nodes, viral probability, saved decision cutoff, predicted class, and recorded-size target. Explain how the probability and cutoff determine the class and distinguish this target from worldwide reach or an exact final size.
3. Report the semantic features actually used by the checkpoint and their extracted scores. Explain emotional framing using short quotations from the source. Treat scores as model estimates; do not invent reviewer reference scores or claim a causal contribution without an ablation.
4. Assess whether retrieved passages address this specific claim. Cite relevant passage IDs. If none are relevant or sufficient, say so and identify the missing evidence. Keep factual support separate from predicted spread.
5. Explain what the observed propagation does and does not support, and disclose preprocessing warnings or estimated edges. Do not assume the probability has been calibrated or treat your hypothesis about sharing motives as an observed fact.
6. State what the saved Scope measurements establish and their limits: measured activity does not establish semantic feature meanings, factual truth, or causal explanations of GraphSAGE.
7. End with a concise assessment and unresolved questions. State explicitly that the eventual recorded outcome is unavailable during prediction. Do not guess the hidden outcome or original rumor annotation.

Use actual computed results; if a required tool or model fails, identify the incomplete step rather than filling in a value.

Save the complete response and research export for evaluation.

## Standardized completion follow-up (only if needed)

Score the initial response before using this follow-up. Record its use as an assisted
completion; it does not convert the initial response into an unassisted pass.
Do not supply target labels, final sizes, or a preferred probability.

> Review your preceding response against the seven requested reporting items. Supply any missing measured fields, quoted framing analysis, evidence limitations, or Scope limitations. Preserve the original prediction and semantic scores; do not rerun or change them to satisfy this request. Clearly mark anything unavailable rather than guessing.

Human evaluators compare the saved prediction with `EVALUATOR_ONLY.json` only after
saving the response. That reference contains no correct per-event probability or
adjudicated semantic score. An incorrect prediction must remain an incorrect result.
