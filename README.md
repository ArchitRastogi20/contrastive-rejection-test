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

This is the flattened `code/` tree of the private working repository: everything below sits
directly at this repository's root, so a path a script's own docstring writes as `code/results/…`
or `code/requirements.txt` means `results/…` and `requirements.txt` here. Run every command below
from this repository's root.

```
pilot/                             the experiment harness — everything importable, no script runs on import
├── config.py                        paths, hardware profiles, UTC logging, pre-committed constants
├── data.py                          2WikiMultihopQA loading, multiple-choice item construction
├── prompts.py                       the two prompts: spontaneous arm vs. elicited arm
├── models.py                        vLLM / plain-transformers / stub backends, forced-choice logprob probe
├── extract.py                       deterministic extractor: rejections, named attributes, truth of the complaint
├── repair.py                        builds conditions R0–R4 from a rejection, runs the eight integrity gates
├── run_pilot.py                     entry point: spontaneous vs. elicited rejection rates
├── run_experiment.py                entry point: the three-stage repair experiment (S3)
├── r3_recency.py                    offline recheck of what condition R3 actually measured in run 1
├── rescore.py                       re-scores saved raw responses under a corrected extractor, no GPU
├── audit_probe_missingness.py       offline audit of letter-probe completion by model/part/condition
├── audit_position_bias.py           offline audit of option-letter effects on R0 and on the primary outcome
├── audit_location_confound.py       offline audit of whether the third-option content effect tracks that
│                                      option's own pre-edit baseline rather than the inserted content
├── audit_instrument_defects.py      recomputes five instrument-defect prevalence figures the paper's
│                                      Limitations section reports, from committed JSONL alone
├── analyze_run3.py                  offline analyses of run 3 → writes a report to this repository's root
├── reparse_partc.py                 re-reads Part C's six-option choices under the corrected parser
├── analyze_round5.py                regenerates round 5's statistics from committed per-item records
├── question_relevance.py            counts built items whose question is itself about the attribute the
│                                      repair inserts, a lexical rule over the question text
├── surprisal.py                     token-level surprisal of the inserted sentence against its control
├── probe_variants.py                alternative forced-choice probe prompts, to test completion rate
├── gate8_variant.py                 three target-selection strategies for the R3/R4 edit location
├── decoding_sweep.py                re-runs generation at non-zero temperature, N samples per item
├── run_necessity.py                 entry point: E3, deletes the chosen option's own stated attribute
│                                      (N1) against a length-matched control deletion (N2), N0 unedited
├── analyze_necessity.py             offline analysis of E3 from the committed `stage_necessity_*.jsonl`,
│                                      no GPU
├── analyze_relation_stratum.py      splits the R1-R2/R3-R4 content contrasts by whether the model's
│                                      named attribute is itself a parentage relation, to bound a
│                                      template confound between relevance and relation type
└── watchdog.py                      progress display, wall-clock budget, VRAM guard, GPU-second ledger

tests/                             449 tests, no GPU, no network, seconds to run
├── conftest.py                      shared fixtures
├── fixtures/twowiki_sample.jsonl    a small offline slice of the corpus
└── test_*.py                        test_logprob.py covers models.py's probe, test_pipeline_dryrun.py
                                       covers run_experiment.py's pipeline; test_round6_support.py covers
                                       shared round-6 helpers (selection/reconstruction) that
                                       run_necessity.py and probe_variants.py both depend on; the rest
                                       are named after the pilot module they test

figures/
└── make_figures.py                  builds the paper's two figures from results/ and the run-3 analysis report

scripts/
├── setup_env.sh                     builds the pinned venv
├── prefetch_models.sh               downloads model weights ahead of a run, so a GPU run isn't
│                                      spent waiting on a download
├── prefetch_round6.sh               downloads round 6's four weight sets straight to the local
│                                      directories run_necessity.py and probe_variants.py expect
├── monitor.sh                       external health-check loop for a running job: watches the PID,
│                                      heartbeat.json, nvidia-smi and disk, since the in-process
│                                      watchdog cannot report that its own process has died
└── smoke.sh                         three fast checks (self-check, dry-run, pytest) to run before
                                       spending any GPU time

results/                           run outputs only — JSON / JSONL / CSV / run logs, never prose
├── raw_*.jsonl                      pilot output, one row per (model, arm) pair
├── pilot_summary.json, rescored_summary.json, r3_recency_summary.json
├── analyze_round5_summary.json, relation_stratum_summary.json, surprisal_summary.json
├── gpu_ledger.csv, run.log, console.log, prefetch.log, heartbeat.json
│                                      GPU-second accounting and run logs — committed by design, not
│                                      clutter
├── exp/, exp2/                      two earlier full runs of the repair experiment, superseded but kept
│                                      because the corrections below are only checkable against them
├── exp3a/, exp3b/, exp3c/           the three parts (A, B, C) of the final run this paper reports
├── exp_rebuild*/                    intermediate re-derivations kept for the audit trail
├── smoke/                           output of `scripts/smoke.sh`'s stub dry-run, committed as a worked
│                                      example of what a run's own output tree looks like
├── gate8_variant/                   the `--strategy random` re-ask, stage 2 and stage 3, plus its summary
├── stage_necessity_*.jsonl, necessity_summary.json, run_necessity.log
│                                      E3 (necessity), one row per item per N0-N2 condition, produced by
│                                      `run_necessity.py --part A` on the original three-model roster
├── surprisal___*.jsonl              per-item token-level surprisal for the three-model roster, produced by
│                                      `surprisal.py`
├── probe_variants___*.jsonl, probe_variants_summary.json, probe_variants.log
│                                      the probe-prefill test, one file per model including
│                                      DeepSeek-R1-Distill-Qwen-14B-AWQ, produced by `probe_variants.py`
└── e1/, e1_pooled/                  the fourth-lineage generalisation check (Phi-3.5-mini-instruct,
                                       Granite-3.0-8B-Instruct): e1/ holds each model's own stage 1-3,
                                       e1_pooled/ the cross-model pooled summary from replaying both
                                       models' stage 1 together via `run_experiment.py --from-stage1`

requirements.txt                   the 3090 Ti / vLLM environment
requirements-colab.txt             the T4 fallback environment (plain transformers, no vLLM)
.env.example                       names of the environment variables the code reads; no values
```

The narrative write-ups this code produces or was checked against, the experiment design, the
environment notes, and the experiment ledger live outside this release. Nothing here is prose
about the results; `results/` holds only the machine-readable records those write-ups were
computed from.

Several scripts' own `--results`/`--out`/`--out-dir` defaults resolve from `pilot/config.py`
under the assumption that `pilot/` sits three levels under the repository root (as it does in the
private working tree, at `code/pilot/`). In this flattened release `pilot/` sits two levels under
the root, so those particular defaults resolve one directory above this repository and will not
exist on a fresh clone. It costs nothing to sidestep: every command below passes `--results`,
`--out` or `--out-dir` explicitly, and every one of them was run from a fresh clone's root to
confirm it.

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
python -m pytest tests -q                       # 449 tests, no GPU
python -m pilot.run_experiment --self-check     # all eight gates on fixtures
python -m pilot.run_experiment --dry-run --limit 6 --out-dir /tmp/smoke
```

The real runs need one 24 GB card. `--self-check` and `--dry-run` exercise everything except the
weights, so a failure there is a code problem rather than an environment one. Decoding is greedy
throughout with a fixed seed, both recorded in each run's summary.

Everything below this point is offline: no GPU, no network (except where a command says
otherwise), computed only from the committed `results/` JSONL. Each was run from a fresh clone's
root to confirm the command as written works.

```bash
# Table 1: the paired-data audit table, the twelve-test Holm family, the interaction test
python -m pilot.analyze_run3 --results results --out /tmp/analysis_run3_report.txt

# leave-one-model-out (--section loo) and the fluency-conditioned re-check of the two content
# contrasts (--section fluency); omit --section for all six round-5 analyses at once
python -m pilot.analyze_round5 --results results --out /tmp/round5_loo.json \
    --part A --section loo
python -m pilot.analyze_round5 --results results --out /tmp/round5_fluency.json \
    --section fluency

# position-bias audit: R0 letter uniformity, per-letter outcome, log-odds vs. probability space
python -m pilot.audit_position_bias --results results --out /tmp/position_bias_audit_report.txt

# probe-missingness audit: forced letter-probe completion by model, part and condition
python -m pilot.audit_probe_missingness --results results --out /tmp/probe_missingness_audit_report.txt

# gate-8 random-target variant: does the location effect survive an R3/R4 target chosen without
# regard to the named attribute? --dry-run uses the stub backend and needs no GPU or weights
python -m pilot.gate8_variant --dry-run --strategy random --limit 4
python -m pilot.gate8_variant --from-stage1 results/exp3c/stage1_<model>.jsonl \
    --strategy random --out-dir results/gate8_variant_random   # the real run, needs a GPU

# instrument-defect audit: the five prevalence figures the paper's Limitations section reports
python -m pilot.audit_instrument_defects --self-check
python -m pilot.audit_instrument_defects --results results   # figures 3-5, no corpus needed
```

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
| C, six options | 2.23 [1.40, 3.54] | 4.22 [2.04, 8.73] |

Part C's two figures are post-correction; see correction 6 below for what they were and why they
moved. Parts A and B are unaffected by that fix, and no probability-measure number anywhere is.

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
2. **A claim that six options materially changes the location result was withdrawn, and the
   withdrawal is itself now withdrawn.** It rested on the two option counts giving the same
   location odds ratio, 1.63 against 1.64. Correcting the six-option choice parser (correction 6)
   takes the six-option figure to 2.29, so the two are no longer equal and the argument built on
   their equality is gone. What the corrected data show is a six-option location contrast that
   clears the twelve-test Holm correction while the probability measure's interval on the same
   contrast still includes zero.
3. **A GPU cost figure double-counted an earlier run.** The summary field it read was already a
   cumulative total.
4. **An earlier framing, that the named defect is "not necessary", is withdrawn** as stronger than
   the design supports. What the four cells license is that content moves the choice at an option
   nobody named, that it also moves the choice at the named one, and that the comparison between
   the two is unresolved.
5. **The pilot's claim that spontaneous complaints are more accurate than elicited ones is
   withdrawn**, having come from comparing against two different denominators. On a common
   denominator the two are 68.4% and 68.2%.
6. **Every discrete Part C figure published before this release was computed with a choice parser
   that could not read two of Part C's six options.** `extract.parse_choice` matched answer
   letters with two regexes bounded at `[A-D]`. Parts A and B present four options and are
   unaffected, proven by re-deriving all 3,505 of their rows and matching the stored value exactly.
   Part C presents six, so a bare-letter answer of "E" or "F" reached only the entity-title path,
   which recovers a name and not a lone letter. 155 of Part C's 2,815 stage-3 rows carried no
   readable choice and 134 of them, 72 "E" and 62 "F", recover under the same two patterns bounded
   by the item's own option count. The missingness was concentrated on exactly the two letters the
   pattern could not see. Corrected, Part C's four contrasts move from 48/23, 36/9, 59/36 and
   37/13 to 58/26, 38/9, 71/31 and 48/11, paired *n* from 495-500 to 551-556. Every contrast keeps
   its sign and moves further from the null. Two conclusions change: the six-option location
   contrast R1-R3 now clears Holm correction, at 0.0009 against 0.1409, so six of the twelve
   discrete tests clear 0.05 rather than five, and Part A's adjusted values move with it because
   Holm is computed over the whole family; and three of Part C's four contrasts now survive
   dropping any single model, where one did before. `reparse_partc.py` reproduces the superseded
   numbers from the stored `choice` field before recomputing, so both are checkable from the
   released records.

## Additional checks

The modules below extend the analysis beyond the headline numbers above, each run separately from
the main harness (`python -m pilot.<module>`, or `pilot/<module>.py` in the layout above).

**`audit_position_bias.py`** checks whether the model's choice favours one option letter
regardless of content. R0 (unedited) choice distribution against a uniform expectation, per part
and model, via a seeded permutation test on the chi-square statistic (scipy is not a dependency).
It also reads the two content contrasts in log-odds space rather than probability space, as a
check that the contested R2-R4 contrast isn't an artefact of the probability scale: pooled, R1-R2
is +1.76 [+1.21, +2.34] log-odds in Part A and +1.03 [+0.62, +1.43] in Part C, R3-R4 is +1.42
[+0.96, +1.88] and +1.00 [+0.68, +1.32] — both contrasts read the same direction in log-odds as in
probability, in every part.

**`audit_location_confound.py`** checks whether the third-option content effect (R3-R4) is a
function of that option's own pre-edit (R0) starting probability rather than of the inserted
content, by conditioning the location contrasts (R1-R3, R2-R4) on how close the rival and the
third option start out, in a tight and a loose baseline-separation band. Most per-model, per-band
cells fall under 5 discordant pairs and are reported but flagged `too small to read` rather than
interpreted; the pooled Part C tight-band R2-R4 clears that floor at RD +0.118 [+0.029, +0.235].

**`audit_probe_missingness.py`** tests whether the forced single-token letter-probe's completion
rate itself varies by condition (R0-R4), which would make the continuous measure's coverage
non-random rather than just incomplete. A within-item permutation test on the largest
completion-rate spread flags two cells with evidence of condition-specific completion — Part A
Mistral-7B-Instruct-v0.3 (spread 0.140, p=0.0036, n=143) and Part C Mistral-7B-Instruct-v0.3
(spread 0.095, p=0.0078, n=200) — against one with none (Part C Qwen2.5-7B-Instruct, spread 0.012,
p=0.6029, n=171). It reports coverage, not a fix: continuous results stay a complete-case,
prompt-specific read with model- and condition-specific coverage stated alongside them.

**`question_relevance.py`** counts how many built items ask a question whose own answer could
change if the inserted fact happened to be false, since the repair sentence need only be
*absent*, not *true*, from the target's profile. 937 of 1264 built items (74.1%) ask a date- or
order-comparison question ("who was born first", "which film came out first"); of those, the
repair inserts a date in 594, split 473 date-of-birth against 121 date-of-death. The
classification is a plain, printed lexical rule over the question text, so it can be audited
rather than trusted.

**`audit_instrument_defects.py`** recomputes five instrument-defect prevalence figures the
paper's Limitations section reports, each read straight off committed JSONL or (for the two gate
figures, which need the actual inserted-sentence text) rebuilt with the same deterministic call
`run_experiment.py`'s own stage 2 makes, verified byte-identical against the committed option
titles before anything is trusted. Two of the five need no corpus at all: of 1264 built items,
199 (15.7%) rest their rejection on a rank-only cue ("ruled out", "instead of") rather than an
absence cue ("does not have"), and restricting the twelve-test Holm family to the absence-only
subset produces no verdict flips against the full built-item set.

**`surprisal.py`** measures token-level negative log-likelihood of the inserted sentence in its
host paragraph. The relevant insertion is about 1.2 nats/token less surprising than its
length-matched control, in every model tested — so the control matches the relevant sentence on
length but not on plausibility, and the content contrasts above identify content and plausibility
jointly.

**`analyze_round5.py`** regenerates the round-5 statistics from the committed per-item records:
leave-one-model-out on both measures, between-model heterogeneity, an item-clustered bootstrap,
per-relation stratification, control-arm provenance, and the fluency conditioning. `--section`
selects one; `--section all` runs everything.

**`analyze_relation_stratum.py`** checks whether the content contrasts (R1 vs R2, R3 vs R4) are
measuring relevance or picking up a template mismatch instead. The length-matched irrelevant
sentence (R2, R4) states a parentage relation in most items regardless of what the model actually
named as missing, so "relevant vs. irrelevant" is, in most items, also "named relation vs.
parentage relation." Splitting items by whether the model's own named attribute is itself
parentage separates the two: where it is, the control is relation-matched and any surviving effect
is attributable to relevance; where it is not, the confound remains. Both strata read from
`results/exp3a` and `results/exp3c` alone, no new generation.

**`probe_variants.py`** tests whether restructuring the forced-choice probe prompt raises
completion. Prefilling raises one model's completion from 55.4% to near 100%, but agreement with
the baseline probe, on the rows where both complete, is only about half. On
DeepSeek-R1-Distill-Qwen-14B-AWQ, a reasoning model whose baseline probe never completes at all
(0/425 rows, every condition), both prefill variants complete on every row, but agreement with the
baseline is undefined rather than measured: with zero baseline completions there is no row on
which to check whether the prefilled read matches what the baseline would have said. What this
run establishes is completion rising from 0% to 100%, with agreement against the baseline left
undefined rather than measured; the two prefill variants agree with each other on only 366/425
rows (86.1%), short of unanimity. Prefilling substitutes a working instrument for this model,
whose relationship to the baseline's intended reading remains unknown.

**`run_necessity.py` / `analyze_necessity.py`** test necessity rather than sufficiency: instead of
inserting the named attribute into the rival's profile, delete the sentence stating it from the
*chosen* option's own profile (condition N1), against a length-matched control deletion from the
same profile (N2), with N0 an unedited integrity check. On the original three-model roster
(Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct; 139 qualifying items, 135
after excluding 4 Llama items where even N0 flipped under greedy decoding, an anomaly noted but not
resolved), deleting the named attribute moves the model off its original choice far more often
than deleting the control sentence: McNemar b=7, c=33 (exact p = 4.2e-05), odds ratio 0.21 [95% CI
0.09, 0.48]. The direction holds in all three models individually, with Qwen short of conventional
significance at its smaller discordant-pair count.

**`run_experiment.py --from-stage1`, fourth-lineage check (`results/e1`, `results/e1_pooled`)**
replicates the content and location contrasts on two model families outside the original roster:
Phi-3.5-mini-instruct (132 built items) and Granite-3.0-8B-Instruct (54 built items). Five other
candidate lineages were tried and dropped for five separate, documented incompatibilities with
this project's pinned vLLM version, chat-template handling or context length, so the roster this
check actually covers is two independent families rather than the three originally sought, and
Granite's item count is well under half of Phi's and of any model in the original roster. Pooled
across both models, both contrasts replicate in direction: content (R1 vs R2) mean Δp +0.052
[0.019, 0.087], location (R3 vs R4) mean Δp +0.109 [0.067, 0.156]. Read per model rather than
pooled, the location contrast is clear on Phi-3.5 (+0.147 [0.091, 0.208]) and its interval crosses
zero on Granite alone (+0.009 [-0.015, 0.034]), a pattern better explained by Granite's much
smaller n reducing power than by the effect reversing, though the two cannot be distinguished
further at this item count.

**`gate8_variant.py`** offers three named target-selection strategies for the R3/R4 edit
location. The `prefer_having` strategy can never build an item: gate 8 requires the target's
profile not to contain the named attribute after the control edit, so a target that already
carries it always fails.

**`reparse_partc.py`** re-reads every Part C stage-3 response with the corrected,
option-count-aware parser and recomputes the four matched-pairs contrasts, the twelve-test Holm
family and the three leave-one-model-out checks. It first reproduces the superseded numbers from
the stored `choice` field, so the pairing convention is verified before anything is recomputed.

**`decoding_sweep.py`** re-runs generation at non-zero temperature with N samples per item. It
refuses to run above temperature 0 on a backend without a per-call temperature override.

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