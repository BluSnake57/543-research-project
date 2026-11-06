# Twitter16 model evaluation through the user workflow

The primary evaluation is **15 research scenarios**, each with one real source
post and its observed propagation. These evaluate whether Qwen and GraphSAGE do
their jobs in Sentinel-X's normal workflow. Import checks, worker cleanup,
exports, and other software mechanics are preparation, not model-quality scores.

Cases are in `datasets/twitter16-model-cases/case-01` through `case-15`:

- `USER_TASK.md`: the actual source post and request to give Sentinel-X.
- `input/`: a single native-format cascade to import and select.
- `EVALUATOR_ONLY.json`: hidden recorded-spread outcomes and reviewer guidance.
- `SCORECARD.md`: the human evaluation form.

The 15 requests cover early spread, claim/evidence review, emotional framing,
uncertainty, research briefings, numerical interpretation, rumor framing, grounded
explanations, Scope interpretation, claim decomposition, and decision support.
Cases include viral and nonviral cascades across the original rumor classes.
The input label file is empty so original rumor annotations do not become answers.

## Expected user workflow

1. Choose a fixed checkpoint and Qwen model/SAE pair. Record their revisions,
   checkpoint cutoff/window/target, device/offload, and generation settings.
   Run 9B and 2B as separate evaluations using the same cases and evidence.
2. Prepare a fixed evidence corpus appropriate to the historical claims. Before
   viewing model outputs, two reviewers identify the claims and agree on what the
   available passages support, refute, or leave unresolved. Store passage IDs,
   short reference rationales, and corpus version outside model context.
   Include some genuinely unresolved cases; an arbitrary empty corpus cannot
   establish fact-checking ability. If no evidence pack has been prepared, evidence
   quality is NOT EVALUATED; only abstention behavior can be assessed.
3. Open a fresh conversation in the main application. Choose **Import posts →
   Twitter15 / Twitter16 input folder…** and select the case's `input/` folder,
   for example `datasets/twitter16-model-cases/case-01/input`. Selecting the case
   folder itself also works. Wait for the import to finish; successful cascades
   are automatically checked in the dataset drawer and attached to that
   conversation. The Trainer is not needed. Choose **JSON / JSONL file…** only
   for those post-file formats. Use the source text and request in `USER_TASK.md`,
   and let Sentinel-X perform
   its normal evidence retrieval, semantic analysis, and GraphSAGE prediction.
   Do not supply the expected class, final size, rumor annotation, or scorecard.
4. Evaluate the final answer, cited passages, and saved model/tool analysis using
   the scorecard. Judge whether the answer helps a researcher understand the
   claim and likely recorded spread. Do not prescribe an exact paragraph or
   require a particular confidence value.
5. Repeat independently for all 15 cases. Keep failures and incomplete responses
   in the results. Report model-loading limitations as INCOMPLETE rather than
   treating them as successful answers. Do not change prompts or cutoffs after
   seeing outcomes; revised evaluations require a new run version.

## What the models are expected to do

**GraphSAGE:** predict whether the cascade will reach the checkpoint's recorded
size target from the permitted early observation period. A correct classification
is a model success; an incorrect one is a prediction error even if every tool ran
correctly. Score the probability as well as the class.

**Qwen:** understand the post, identify factual claims, retrieve and interpret
relevant evidence, extract meaningful semantic features, and explain the resulting
spread assessment with calibrated uncertainty. Fluent prose or a completed tool
call alone earns no quality credit.

**Hybrid semantic contribution:** for the shipped emotional-language checkpoint,
have reviewers annotate the intensity of emotional language using the same scale
as the semantic extraction schema. Require a text-based rationale with quotations.
Compare the saved extracted score with the agreed reference and report absolute
error and rank agreement. The semantic intensity itself is subjective; establish
reviewer agreement first. Do not invent a numeric reference from a rumor label.
To claim the LLM improves GraphSAGE, compare hybrid and graph-only checkpoints
trained on the same event splits, observation window, target and training budget.
Report the difference in held-out F1 and average precision; a good hybrid answer
by itself does not demonstrate added value from semantic features.

**Scope:** sparse activations may describe measured model activity. There is no
Twitter16 ground truth for feature meanings or causal explanations. Evaluate
whether Qwen interprets these measurements honestly; do not score activation
presence as evidence of model accuracy.

## Per-case pass/fail

Score the Qwen response against the reviewer references, with two independent
reviewers and adjudication of disagreements. Use 0 = fails, 1 = partly meets,
2 = meets the requirement:

| Dimension | 2: meets | 1: partial | 0: fails |
| --- | --- | --- | --- |
| Claim understanding | Faithfully identifies the central claim and relevant qualifications | Main claim recognizable but misses a qualification | Changes/invents the claim or misses its central meaning |
| Evidence and uncertainty | Uses passages that address the claim; conclusion follows evidence; abstains on gaps | Relevant evidence but incomplete coverage or weak uncertainty wording | Unsupported confident conclusion, fabricated citation, or irrelevant evidence treated as proof |
| Semantic/framing analysis | Describes framing with source quotations and agrees with reviewer reference | Plausible but weakly grounded or misses a salient feature | Invented framing, unsupported motive, or semantic analysis inconsistent with text |
| Interpretation and usefulness | Explains the prediction and limitations accurately and answers the requested research question | Useful answer with an omission | Misstates prediction, target or evidence, conflates spread with truth, or fails the task |

If a request does not elicit semantic analysis, ask the same fixed follow-up for
all cases: “Describe the post's emotional framing using quotations from its text.”
Keep it separate from the initial prediction; do not adjust that prediction.

Proposed pilot acceptance gate (freeze before the first inference): **Qwen PASS**
requires at least 7/8, no zero dimension, and no invented factual evidence,
citation, or prediction probability. Any such fabrication, or treating spread as
proof of truth, is a case FAIL regardless of total. These are project evaluation
criteria, not validated universal benchmarks.

For GraphSAGE, **prediction CORRECT** means predicted class equals the hidden
recorded-spread target. Report incorrect results as INCORRECT, not as software
failures. A case's **end-to-end PASS** requires both Qwen PASS and prediction
CORRECT. Mark incomplete responses explicitly and report completion rate as well
as quality on completed cases; never silently drop them.

## Expected outcomes and numerical scoring

The shipped checkpoint uses a **464-node** target, a **60-minute** window, and its
saved probability cutoff. Use `viral_target_464` for it. A Twitter16 checkpoint
trained for **475** nodes uses `viral_target_475`. Class = probability >= the saved
cutoff. The expected probability is deliberately unspecified: no known correct
per-event probability exists. Compute Brier error `(probability - target)^2`;
lower is better. Accuracy, precision, recall, F1 and average precision summarize
performance across cases. The target means final *recorded* normalized cascade
size, not worldwide lifetime reach.

Original `true`, `false`, `unverified`, and `non-rumor` annotations are evaluator
context only. `non-rumor` does not prove every statement true, and `unverified`
does not mean false. Factual-answer scoring uses the independently reviewed
available evidence, not automatic comparison to these annotations. If original
labels disagree with the evidence review, document and adjudicate that difference.

Pilot suite report:

- Qwen: total score distribution, per-dimension scores, pass rate, fabrication
  count, and agreement between reviewers. Proposed pilot gate: at least 12/15
  cases pass with zero fabricated evidence/citations and all 15 completed.
- GraphSAGE: correct/incorrect counts, confusion matrix, F1, average precision,
  and mean Brier error. Report all 15 predictions; do not impose an arbitrary
  exact probability or call a threshold fitted on these cases a held-out result.
- End to end: number satisfying both model requirements, plus representative
  error analyses showing what a researcher would have misunderstood.
- Hybrid versus graph-only: paired results on identical events, if comparable
  checkpoints are available; otherwise semantic added value is NOT EVALUATED.

Fifteen deliberately selected examples provide a qualitative model acceptance
exercise, not reliable proof of generalization. The earlier 15 batches under
`datasets/twitter16-suite/` partition all **818** events and are supporting data
for larger prediction evaluation. Of these, **816** are usable observed cascades;
2 are disconnected and excluded from model scoring with the reason recorded.
Their import-rejection behavior is outside this model evaluation.

For a model-quality release claim, evaluate the larger corpus after removing
training/validation and cross-dataset event overlap. Compare F1 and average
precision to majority and early-observed-size baselines selected on training or
validation data only. Pre-register the required improvement and confidence
interval rule before running evaluation. Until that reference evaluation exists,
report GraphSAGE aggregate quality as EXPLORATORY; a model that makes a prediction
has not thereby demonstrated predictive skill.

The scenario selector excludes literal IDs in the shipped checkpoint's train and
validation lists. This is not a semantic duplicate audit; verify cross-dataset
matches before held-out claims. The shipped checkpoint is trained on Twitter15,
so Twitter16 is a cross-dataset evaluation. Checkpoint changes require rechecking
overlap and target definitions. Fixed evidence packs may contain real evidence
about the claims, but must exclude evaluator labels, final cascade outcomes,
scorecards, and expected responses.

## Data and reproduction

Tree/label mirror: https://github.com/gszswork/Twitter15_16_dataset
Revision: `6a33ebd91d4cafeb5a42eabc5727226fec7bb0e5`.
Source-text mirror: https://github.com/OwenLeng/rumor-detection-include-twitter15-twitter16data-
(`dataset/twitter16/twitter16.{train,dev,test}`). Text is preprocessed, including
URL placeholders. Original dataset research: https://aclanthology.org/P17-1066/.
The committed cases include download hashes, source attribution, and per-file
checksums in `datasets/twitter16-model-cases/PROVENANCE.json`. The selected
15 cases are versioned; full downloads and intermediate batches remain ignored.
Additional local download provenance is in
`datasets/twitter16-suite/download-provenance.json`. Preserve upstream attribution.

To generate a fresh set of the same scenarios from the downloaded corpus:

```sh
.venv/bin/python scripts/prepare_twitter16_workflows.py --output datasets/twitter16-model-cases-new
.venv/bin/python -m ruff check scripts/prepare_twitter16_workflows.py
.venv/bin/python -m ruff format --check scripts/prepare_twitter16_workflows.py
```

No model inference or evidence-reference adjudication has run as part of preparing
these materials. Expected spread targets exist; actual model scores remain blank.

## Case 01 reporting revision

Case 01 now has a version-2 primary prompt and an optional standardized completion
follow-up in `USER_TASK.md`. Record the prompt version and score any assisted
follow-up separately. The revision followed inspection of a pilot failure; freeze
it before a new run rather than treating earlier and revised outputs as one fixed
protocol. [The pilot review](CASE01_REVIEW.md) distinguishes reporting omissions
from an actual false-negative prediction. Keep that review out of model context.
