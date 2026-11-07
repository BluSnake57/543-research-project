# Research workflow

This workflow uses **FEVER only** as factual evidence. It performs no web search
at inference time and imports no externally researched articles. The normal app
continues to use its configured index; research runs use isolated application
storage and the versioned FEVER candidate pack in this directory.

## Data audit findings

All 15 case trees, source texts, original annotations, recorded sizes and first-hour
counts match the recorded source files and an independent graph calculation.
All original provenance hashes match. The ZIP downloaded by commit has different
ZIP wrapper bytes from the original branch archive; the source lock deliberately
checks the original archive hash rather than silently replacing that reference.

Case 12 has an exact normalized-text match in checkpoint training/validation data.
Case 03 has a near match (lexical similarity >= 0.85). Both are excluded from
**overlap-screened** metrics. All 15 remain available as development/qualitative
cases. No original cases or reference outcomes were rewritten. Human event-level
duplicate review is still required before calling any subset held-out.

The prepared full corpora contain 1,487 usable Twitter15 cascades out of 1,490,
and 816 usable Twitter16 cascades out of 818. Import reports retain the disconnected
cascades and rejection reasons. See `data-audit/audit.json` and `audit.md` for the
case-level checks and matched IDs.

## Evidence and reviewers

`fever-v1/pack.json` freezes 595 candidate Wikipedia sentences from the existing
FEVER index. It is a **keyword-retrieved subset**, not a copy of the entire FEVER
corpus and not an adjudicated set of relevant evidence. The source index had no
unfinished import records. Candidate selection uses up to eight distinct long
source-text terms, excludes common function words, and takes up to 40 FEVER
sentences per case from the first 200 keyword matches. Every sentence retains its
page, sentence number, imported-file hash and text hash. These are indexed-file
provenance records; the original FEVER archive was not rehashed in this workflow.

Only passage text and provenance enter the research index. Reviewer forms, case
labels, reference sizes, scores and expected answers never enter model context.
The restricted corpus may not cover a claim; lack of evidence is not proof of
falsehood. Citation-ID validation alone does not establish citation relevance.

Two humans independently complete the reference claim, relevant-passage and
emotional-intensity fields before seeing outputs. They then score saved model
outputs using the existing four-dimension rubric. Use copies of the blank forms
under `fever-v1/reviewers/`, fill distinct reviewer IDs, the run manifest hash and
the same Thinking budget, and adjudicate disagreements in a third form. Keep
private reviewer identities and completed annotations out of Git unless explicitly
approved. Automated reports leave missing human judgments **PENDING**.

## Commands

Use the configured project `.venv`. Run commands from the repository root.
Generated datasets, captures, SQLite files, checkpoints and long experiment outputs
belong under `.cache/research/`, which Git ignores. Model files are read through
symlinks into isolated storage; user settings, conversations and the selected
checkpoint are never changed.

```bash
.venv/bin/python scripts/research.py fetch-sources
.venv/bin/python scripts/research.py audit --output .cache/research/audit
.venv/bin/python scripts/research.py prepare-corpus --dataset twitter15 --output .cache/research/twitter15
.venv/bin/python scripts/research.py prepare-corpus --dataset twitter16 --output .cache/research/twitter16
```

Preparation refuses to overwrite an existing corpus. The checked-in FEVER pack
can be used offline. To make a new pack from an installed index:

```bash
.venv/bin/python scripts/research.py freeze-fever --index /path/to/index/evidence.sqlite --output .cache/research/fever-v2
```

Development pilot (three cases, budgets 256/512/1024; not a held-out comparison):

```bash
.venv/bin/python scripts/research.py evaluate --config docs/research/pilot.json --output .cache/research/pilot-v2
.venv/bin/python scripts/research.py report --output .cache/research/pilot-v2
.venv/bin/python scripts/research.py score --output .cache/research/pilot-v2
```

The pilot configuration records the paths used on this machine. For another
machine, copy it and change `model_data` and the FEVER-pack path. Choose a fresh
output directory when changing any frozen setting. Repeating the same command
skips completed cases and retries failures/interrupted cases in new attempt files.
Ctrl-C stops generation and preserves partial records. The default per-case
1200-second timeout is configurable. Do not run two processes on the same output
directory simultaneously.

Read the pilot completion, latency, memory and budget-hit reports before choosing
full-run settings. Qwen quality still requires reviewer scores; a higher budget is
not automatically better. The `full.json` example retains budget 256 as a starting
configuration, **not a measured recommendation**. Full runs are explicit:

```bash
.venv/bin/python scripts/research.py evaluate --config docs/research/full.json --output .cache/research/full-v1
.venv/bin/python scripts/research.py extract --config docs/research/twitter15-features.json --output .cache/research/twitter15-features
.venv/bin/python scripts/research.py extract --config docs/research/twitter16-features.json --output .cache/research/twitter16-features
.venv/bin/python scripts/research.py train-matrix --config docs/research/train-matrix.json --output .cache/research/matrix-v1
```

The full-run command paths are covered by automated fixtures; full 9B feature
regeneration and the actual 18-run training matrix have **not** been executed as
part of implementation. Cached legacy features remain unchanged. New extraction
uses the same prompt/validator as interactive extraction, Thinking On, and the
single emotional-language feature. Set the same budget in both extraction configs
and the full evaluation config after protocol selection.

The matrix uses widths 32/64/128, graph-only and hybrid variants, and initialization
seeds 42/43/44. All runs use split seed 42, threshold 464, first 60 minutes, and up
to 30 epochs with existing early stopping. Validation F1 determines configuration
ranking; test results are reported separately. PyTorch's `cuda` device selects the
AMD GPU when using the installed ROCm build. No checkpoint is automatically promoted.

For broader cross-dataset scoring, create a configuration with `checkpoint`
(metadata JSON from the matrix), `dataset` (prepared Twitter16), `training_dataset`
(prepared Twitter15), and `feature_cache` (matching versioned Twitter16 features;
omit for graph-only). Run `cross-evaluate --config CONFIG --output NEW_DIRECTORY`.
It preserves the checkpoint target/cutoff, excludes ID and lexical overlap
candidates, records rejections, and reports F1, average precision, Brier error,
reliability SVG and training/validation-selected baselines. These results remain
exploratory pending event-level overlap review and a preregistered improvement rule.

## Verification

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src/backend scripts tests
```

Regression coverage includes graph repair and 60-minute boundaries, source/hash/
reference mismatches, overlap, shared extraction, legacy cache detection,
FEVER-only indexing, resume manifest changes, retained failures, fixed training
splits, numerical metrics and pending human judgments. Live pilot results are
reported separately from software-test success.

## Unattended pilot and automatic logging

The baseline pilot writes `.cache/research/pilot-v2/run.log`, `status.json`,
`results.csv`, `summary.json`, and per-attempt JSON automatically. At the end it
also writes `scores.json` and reliability SVGs when predictions are available.
`completed_with_failures` means execution finished with failed/incomplete cases;
it does not claim those cases passed. Raw Scope outputs and conversations remain
in each isolated session database. No full feature extraction or training matrix
is started automatically when this pilot finishes.

Monitor without modifying the run:

```bash
tail -f .cache/research/pilot-v2/run.log
cat .cache/research/pilot-v2/status.json
```

The preliminary pilot-v1 was stopped after discovering that thinking and answers
shared a total generation cap. Its failed/interrupted attempts are preserved under
`.cache/research/pilot-v1`. Cases 01–03 failed response-format validation
at budget 256; these are findings, not successful analyses. Pilot-v2 reserves
1,600 answer tokens independently of the Thinking budget and uses stable graph
IDs across budgets. Semantic protocol version 2 records the revised budget
semantics. The total prompt plus reserved thinking/answer space must still fit
the configured context; excess input fails explicitly rather than truncating evidence.

Long-running commands acquire an output-directory lock and record terminal status
even on exceptions or Ctrl-C. A machine shutdown cannot execute cleanup: in that
case a stale `running` status is possible, but the OS lock releases and a subsequent
identical invocation preserves the old attempt and resumes remaining work.

Pilot-v2 finished with 2/9 accepted answers: Case 02 at budgets 512 and 1024.
Four attempts failed JSON parsing and three failed factual citation validation.
All nine retained GraphSAGE predictions; increasing Thinking did not change the
three semantic estimates or predicted probabilities. None of these failures was
an output-token truncation or GPU memory error. Accepted formatting alone did
not establish evidence quality: the Case 02 answer still misdescribed a passage
and used unsupported rumor wording. Human scoring remains pending.

The response repair uses a shared, nested JSON example for both initial answers
and the bounded retry. The retry starts a fresh message context containing the
grounding instructions, request, current evidence, failed draft and specific
validation error. Follow-up context and previous tool results are preserved as
quoted context. The failed draft is quoted data. It is never silently split or
trimmed into a successful answer. Duplicate keys, nonfinite values, incomplete
analysis and unsupported factual citations fail validation. Raw outputs remain
in Scope records, and retry messages and their outcome remain in the conversation
audit. Formatting is prompted and then validated; the worker does not implement
grammar-constrained decoding.

Evidence instructions distinguish irrelevant search results from refutation,
require source-based descriptions, and prohibit inferring a rumor/falsehood label
from missing corroboration. These instructions do not replace human relevance
and factuality review. The FEVER pack, evaluation references and checkpoint remain
fixed for the repair comparison.

The first repair smoke run (`.cache/research/pilot-repair-v3-512`) completed all
three cases on their first response at budget 512, up from 1/3 at that budget in
pilot-v2. It took about 12 minutes. The raw outputs still exposed quality issues:
Cases 01 and 02 conflated node and edge counts; Case 01 assumed linked video
content. This is a formatting improvement, not a claim that all content is correct.
The three changed implementation files for that preliminary run are retained in
its ignored `implementation/` directory alongside the frozen run manifest.

The subsequent changes check explicit node/edge count restatements in a single
cascade's limitations/assessment against the actual prediction and request a
repair on mismatches. This check is intentionally narrow: it does not judge
semantic entailment or guess attribution among multiple cascades. The prompt now
asks for qualitative interpretation alongside the application's exact numeric
report, distinguishes probability cutoffs from final-size targets, and avoids
assuming a media type from URL placeholders. Test the final version separately:

```bash
.venv/bin/python -u scripts/research.py evaluate \
  --config docs/research/repair-smoke.json \
  --output .cache/research/pilot-repair-v4-512
```

Inspect that directory's `status.json`, `results.csv`, `summary.json` and
`run.log` for the final retest outcome; do not infer its result from v3. The
512-token smoke test does not select an optimal Thinking budget. The checkpoint
and its Case 01 false negative have not been changed.

Implementation verification: 220 tests passed; Ruff and `git diff --check` passed.
