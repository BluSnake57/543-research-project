# Case 01 pilot review and reporting revision

This is an engineering review of the saved completed pilot job
`0fae38a0fe644cdcba349e6300fd08be`, not an adjudicated human evaluation. The
reference is `datasets/twitter16-model-cases/case-01/EVALUATOR_ONLY.json`.
Keep this review out of model prompts and retrieval indexes.

| Item | Saved pilot output | Reference / assessment |
| --- | --- | --- |
| Cascade | 643103859729166337 | Correct case |
| Probability | 0.21661721169948578 | No expected per-event probability exists |
| Cutoff | 0.712752640247345 | Matches selected checkpoint |
| Predicted class | NOT VIRAL | Internally correct because probability < cutoff; **INCORRECT against recorded outcome** |
| Recorded outcome | Not reported | 564 final nodes, viral at both 464 and 475; withheld during prediction |
| Target and observation | 464 nodes, first 60 minutes | Correct for shipped checkpoint; 475 is for a different checkpoint |
| First-hour size | Omitted | 135 normalized nodes; now computed from the same prefix used by GraphSAGE |
| Brier error | Not reported | (0.21661721169948578 − 1)² = 0.6136885930, evaluator-only metric |
| Emotional-language estimate | 0.95 in auxiliary model output; omitted from visible answer | No adjudicated human reference score; numerical correctness cannot be claimed |
| Claim and framing explanation | Omitted | Does not satisfy the requested qualitative analysis |
| Evidence | Retrieved passages included unrelated material about post offices/software | No claim-specific justification or adjudicated evidence assessment |
| Uncertainty | Spread/truth distinction in reasoning summary | Missing discussion of calibration, unknown URL contents and exact-size limits |
| Scope | Auxiliary capture saved | Measurements alone do not establish semantic meanings or causality |

The old application returned its fixed numeric report immediately after GraphSAGE.
The language model only extracted semantic scores; it did not write a research
explanation for the final answer. This is why asking for additional explanation
in the initial prompt could not make that answer complete.

## Revision

The app now reports observed-prefix nodes/edges and actual semantic inputs with
the checkpoint's probability, cutoff and target. It then asks Qwen to finish the
claim, framing, evidence and limitations analysis. Exact numerical results remain
in an application-generated report alongside the model analysis and raw tool
records. Final cascade size, original rumor labels, evaluator references, and
later post text are excluded from the initial model context. Retrieved passages
are searched using the source post rather than the evaluation instructions.
Full-import repair counts are also withheld from that context: they describe the
entire supplied graph, not just the observation window. The final report labels
them as import diagnostics and preserves the original warnings for auditing.

Thinking defaults to On and its control is in Settings → Models. Semantic
extraction follows that setting, so the new run may produce different semantic
inputs and probabilities. Those changes must be reported, not selected to make
this case pass. Checkpoint weights, cutoff, target and evaluator outcomes have
not been changed to correct the pilot's prediction error.

Case 01's `USER_TASK.md` contains a version-2 primary prompt and a standardized
completion follow-up. The follow-up is scored separately as assisted completion.
The generator reproduces the versioned prompt and the provenance hash is updated.
Because the prompt was revised after observing a pilot failure, report this as a
development revision. Freeze it before new evaluations; do not pool the pilot
with the revised run as if they used an unchanged protocol.

A complete report can still make an incorrect prediction. Human evidence/framing
references and two-reviewer adjudication are still required for the research
rubric. Do not claim all information is correct merely because generation finishes.

## GPU validation and remaining research work

An isolated run on the RX 9070 XT used the installed Qwen3.5-9B / layer-16 SAE,
ROCm, NF4 quantization, SAE CPU offload, and Thinking On with a 256-token budget.
It completed in 222 seconds, including model verification/loading. Both semantic
extraction and response generation ended at a recognized stop token, without
output truncation. Both reached the Thinking budget. No evidence corpus was
loaded in this isolated run, so evidence sufficiency/retrieval quality was not
validated.

The run reported 135 observed nodes, 136 observed edges, emotional-language score
0.90, probability 0.21750508248806, cutoff 0.712752640247345, and NOT VIRAL. The
checkpoint and recorded-size target remained unchanged. This is still a false
negative against the withheld 564-node outcome. The original 0.95 semantic score
produces the original probability in a regression test; neither score is an
adjudicated semantic reference.

Live Qwen responses still confused graph counts with Scope measurements despite
explicit prompt instructions. The final application therefore generates the
Scope section from saved records, including model/SAE, layer, saved token counts,
capture status, and whether the Thinking budget was reached. It retains raw Qwen
output for auditing. The saved real response was replayed through this final
reporting path: the Scope section used 455 auxiliary and 3,036 response token
records, preserved the prediction, and removed the generated claim that graph
counts were Scope measurements. This replay verifies reporting using real saved
outputs; it is not another fresh generation after the last prompt revision.

The remaining qualitative interpretation needs human review. For example, the
live response inferred that the post invited skepticism about the Red Cross;
the wording alone does not establish the target of criticism or audience motives.
The revised instructions explicitly require these interpretations to be marked
as hypotheses. The bounded Thinking budget is another condition to record and
evaluate, not evidence that reasoning was exhaustive.

Before research evaluation, freeze the extraction and reporting protocols, model
revisions, evidence pack and generation settings. The cached training features
and current interactive extraction use different prompt/Thinking conventions;
measure or eliminate that mismatch before attributing errors to GraphSAGE
capacity. Establish independent framing/evidence annotations, then evaluate the
full protocol with assisted follow-ups scored separately. Assess calibration,
cross-dataset generalization and larger-model comparisons on appropriate
validation splits; do not tune to this inspected case.

Verification commands: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest
discover -s tests -v` (193 passing tests) and `.venv/bin/ruff check src/backend
tests scripts`. No trained weights, evaluator outcomes or decision cutoff were
modified.
