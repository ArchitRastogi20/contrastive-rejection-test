# Does a model's stated reason for rejecting an option do any work?

Code and per-item records for the experiments in the accompanying paper. Everything here runs
without a GPU except the generation itself, and every measurement is a deterministic rule over
strings. No language model judges any output at any point.

## The experiment in one paragraph

A model is shown a question and several candidate entities, each with a short profile of real
corpus sentences, and is asked to choose one and explain the choice. It often says why it ruled a
specific rival out, naming a concrete fact that rival's profile lacks: no date of death, no listed
director, no stated parentage. That is a claim about the text in front of it, so it can be
falsified. We insert a sentence carrying the named fact and ask again. A 2x2 crosses the content
of the inserted sentence, relevant against irrelevant and length-matched, with where it lands,
the rival the model named against a third option it never mentioned.

## Layout

```
code/
├── pilot/                          the experiment harness — everything importable, no script runs on import
│   ├── config.py                     paths, hardware profiles, UTC logging, pre-committed constants
│   ├── data.py                       2WikiMultihopQA loading, multiple-choice item construction
│   ├── prompts.py                    the two prompts: spontaneous arm vs. elicited arm
│   ├── models.py                     vLLM / plain-transformers / stub backends, forced-choice logprob probe
│   ├── extract.py                    deterministic extractor: rejections, named attributes, truth of the complaint
│   ├── repair.py                     builds conditions R0–R4 from a rejection, runs the eight integrity gates
│   ├── run_pilot.py                  entry point: spontaneous vs. elicited rejection rates
│   ├── run_experiment.py             entry point: the three-stage repair experiment (S3)
│   ├── r3_recency.py                 offline recheck of what condition R3 actually measured in run 1
│   ├── rescore.py                    re-scores saved raw responses under a corrected extractor, no GPU
│   ├── audit_probe_missingness.py    offline audit of letter-probe completion by model/part/condition
│   ├── analyze_run3.py               offline analyses of run 3 → writes a report outside code/
│   └── watchdog.py                   progress display, wall-clock budget, VRAM guard, GPU-second ledger
├── tests/                          176 tests, no GPU, no network, seconds to run
│   ├── conftest.py                    shared fixtures
│   ├── fixtures/twowiki_sample.jsonl  a small offline slice of the corpus
│   └── test_*.py                      test_logprob.py covers models.py's probe, test_pipeline_dryrun.py
│                                       covers run_experiment.py's pipeline; the rest are named after the
│                                       pilot module they test
├── figures/
│   └── make_figures.py               builds the paper's two figures from results/ and the run-3 analysis report
├── scripts/
│   ├── setup_env.sh                   builds the pinned venv
│   ├── prefetch_models.sh             downloads model weights ahead of a run, so a GPU run isn't
│   │                                   spent waiting on a download
│   └── smoke.sh                       three fast checks (self-check, dry-run, pytest) to run before
│                                       spending any GPU time
├── results/                        run outputs only — JSON / JSONL / CSV / run logs, never prose
│   ├── raw_*.jsonl                    pilot output, one row per (model, arm) pair
│   ├── pilot_summary.json, rescored_summary.json, r3_recency_summary.json
│   ├── gpu_ledger.csv, run.log, console.log, prefetch.log, heartbeat.json
│   │                                   GPU-second accounting and run logs — committed by design, not
│   │                                   clutter
│   ├── exp/, exp2/                    two earlier full runs of the repair experiment, superseded but kept
│   │                                   because the corrections below are only checkable against them
│   ├── exp3a/, exp3b/, exp3c/         the three parts (A, B, C) of the final run this paper reports
│   └── exp_rebuild*/                  intermediate re-derivations kept for the audit trail
├── requirements.txt                the 3090 Ti / vLLM environment
├── requirements-colab.txt          the T4 fallback environment (plain transformers, no vLLM)
└── .env.example                    names of the environment variables the code reads; no values
```

The narrative write-ups this code produces or was checked against, the experiment design, the
environment notes, and the experiment ledger live outside this tree. Nothing under `code/` is
prose about the results; `results/` holds only the machine-readable records those write-ups were
computed from.

## Environment

All runs in `results/` were generated on one RTX 3090 Ti (24 GB), on the
Python 3.11, PyTorch 2.4.0,
CUDA 12.4.1, Ubuntu 22.04. `requirements.txt` pins `vllm==0.6.3.post1`, the last vLLM release
built against torch 2.4.x. `requirements-colab.txt` is a T4 fallback (see Limitations for what
that roster cannot show).

## Conditions

Every condition sends a byte-identical prompt except for the one profile it edits. Option order,
letters and the question never change.

| | inserted content | edited profile |
|---|---|---|
| R0 | nothing | nothing, so the choice must reproduce exactly under greedy decoding |
| R1 | carries the named attribute | the rival the model complained about |
| R2 | a different attribute, length-matched | the rival |
| R3 | the same sentence as R1 | a third option, never mentioned |
| R4 | the same sentence as R2 | that third option |

R0 is an integrity check. Any flip there means generation is not deterministic and nothing else is
interpretable. It returned zero flips across 1264 rows in nine model-part cells.

Inserted sentences are real corpus sentences with the subject substituted. Nothing is composed.
The inserted fact need not be true of the target entity: the claim under test is that the profile
*gives* no date of death, and supplying any date of death falsifies that claim. This scopes the
result to the text-level claim and not to the semantic relation; see Limitations.

Eight integrity gates run before any generation, all deterministic. Among them: the named
attribute must be absent before the edit and present after; R2 must not accidentally supply it;
R1 and R2 sentences must match in length within 20%; an inserted date must not contradict a date
already present; the gold answer's profile is never touched; and the option R3 and R4 edit must
itself lack the named attribute, without which the location contrast compares supplying new
information against duplicating existing information.

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # 176 tests, no GPU
python -m pilot.run_experiment --self-check     # all eight gates on fixtures
python -m pilot.run_experiment --dry-run --limit 6 --out-dir /tmp/smoke
```

The real runs need one 24 GB card. `--self-check` and `--dry-run` exercise everything except the
weights, so a failure there is a code problem rather than an environment one. Decoding is greedy
throughout with a fixed seed, both recorded in each run's summary.

## The records

`results/exp3a`, `exp3b` and `exp3c` hold the three parts of the final run; `exp` and `exp2` hold
the two earlier runs, kept because the corrections below are only checkable against them.

**`stage1_*.jsonl`**, one row per item: `question`, `option_titles`, `gold_letter`, the model's
`choice` and `choice_correct`, the parsed `rejections`, and the full raw `response`. `selected`,
`selected_rival_letter` and `selected_attribute` record which rejection was carried forward.

**`stage2_*.jsonl`**, one row per candidate item: `built` true or false, and when false
`drop_category`, `drop_reason` and any `gate_failures`. Nothing is silently discarded.

**`stage3_*.jsonl`**, one row per item per condition: `condition`, `edited_letter` and
`chosen_is_edited` (the primary outcome), `p_edited` and `delta_p_edited` (the probability
outcome, the second measured against that item's own R0 baseline on the same letter),
`r0_p_target`, the full `letter_probe` including per-letter log-probabilities and whether the read
was `complete`, `still_names_same_defect`, and the full raw `response`.

Raw responses are kept everywhere. The extractor has been revised twice and both times the
existing runs were re-scored offline rather than regenerated.

## Headline numbers

Odds ratios for the two content contrasts, with 95% intervals, computed per part:

| part | R1 vs R2, at the named rival | R3 vs R4, at the unnamed third option |
|---|---|---|
| A, four options | 2.40 [1.31, 4.38] | 3.22 [1.53, 6.81] |
| B, four options, fresh roster | 1.75 [0.86, 3.56] | 1.07 [0.52, 2.22] |
| C, six options | 2.09 [1.27, 3.43] | 4.00 [1.93, 8.30] |

The interaction between content and location, which asks whether the named location receives more
of the effect, crosses zero in every part on both measures: continuous +0.0008 [-0.0292, 0.0306],
-0.0200 [-0.0595, 0.0183] and +0.0138 [-0.0054, 0.0340] for A, B and C. Equivalence holds only to
within roughly 3 to 7.5 percentage points, so the data distinguish neither equality nor
difference.

## Corrections, and values that appear in older files

The runs were analysed as they completed and several figures were withdrawn on recomputation. The
superseded values survive in the earlier `results/exp` and `results/exp2` directories and in the
summaries written at the time. They are listed here so nothing in this release is mistaken for a
current claim.

1. **A ceiling-effect explanation for the disagreement between the two measures, once quoted as a
   24-fold difference in baseline, is withdrawn.** The figure came from a median rather than a
   mean, a subset restricted to the majority direction, and one part checked without the other.
   Recomputed on all items it is 1.45 and 0.91 for the two parts, neither significant, and a
   dose-response test over 950 items rejects the explanation outright.
2. **A claim that six options materially changes the location result is withdrawn.** The effect
   size is the same at four and six options, odds ratio 1.63 against 1.64; the sample grew by 32%.
3. **A GPU cost figure double-counted an earlier run.** The summary field it read was already a
   cumulative total.
4. **An earlier framing, that the named defect is "not necessary", is withdrawn** as stronger than
   the design supports. What the four cells license is that content moves the choice at an option
   nobody named, that it also moves the choice at the named one, and that the comparison between
   the two is unresolved.
5. **The pilot's claim that spontaneous complaints are more accurate than elicited ones is
   withdrawn**, having come from comparing against two different denominators. On a common
   denominator the two are 68.4% and 68.2%.

## Limitations of the artifact

The probability probe returns nothing usable on a reasoning model, whose single decoded token is
always the opening of its reasoning block, and completes on 55% then 33% of one other model's rows
as the option count grows. Figures using that measure cover only the models it can read, and the
per-row `complete` flag makes the coverage checkable.

The stratum that would separate sensitivity to the stated criterion from sensitivity to a matching
attribute template, items where the corpus supplies the target entity's own true value, returned
zero usable items across roughly 475 built items in three runs.

A matched-baseline subsample, which would neutralise the difference in starting probability
between the two edit locations, exists at 39 and 36 items under a band tight enough to be useful,
against roughly 728 pairs needed for 80% power at the measured effect size.

## Corpus

2WikiMultihopQA, validation split, used unmodified as the source of entities, profiles and
evidence triples.

## License 

MIT