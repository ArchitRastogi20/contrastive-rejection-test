# Does a model's stated reason for rejecting a candidate do any work?

Code, per-item records and every raw model response behind the paper of the same title
(Archit Rastogi, 2026).

## The question

Asked to choose between candidates and explain itself, a language model often rejects a rival
by naming a fact its profile lacks: *"It is Bruno. It is not Anna, because Anna's profile never
says she directed anything."* That second sentence is a claim about the text in front of the
model, so it can be tested mechanically. We insert a sentence stating the named fact into the
rival's profile, borrowed verbatim from a sibling profile elsewhere in the same corpus, and ask
again under greedy decoding. If the choice moves toward the rival, the stated reason was doing
work; if it does not, it was decoration that read like a reason.

Two controls separate *what* the edit says from *where* it lands: a length-matched irrelevant
sentence at the same profile, and the same two sentences at a third option the model never
mentioned. **No model or person judges any output**: every measurement is a deterministic rule
over strings, a re-ask, or a forced single-token read.

## Results at a glance

- **Content at the named rival moves the choice.** Supplying the named fact beats the irrelevant
  control, OR 3.57 [1.54, 8.26], Holm p = 0.0210, and survives dropping any single model.
- **The contrast the design was built to detect does not clear correction.** The same fact at the
  option nobody named: Holm p = 0.2428.
- **Placement matters at least as much as content.** The strongest result carries no content
  claim at all: the identical irrelevant sentence moves the choice more at the named rival than
  at the third option, Holm p = 0.0008.
- **The chosen option's own reason is necessary.** Deleting it moves the model off its choice far
  more than a matched control deletion, OR 0.21 [0.09, 0.48].
- **The pipeline audited itself and found eight defects.** Three are fixed, five are quantified;
  the largest, a choice parser that returned the option a model had just rejected in 17.1% of
  responses, changes which contrasts survive once fixed.

Details, tables and every caveat are below.

## Quick start

Every number this repository claims to reproduce regenerates **on CPU alone, offline**, from the
committed per-item records in `results/`: no GPU, no weights, no network.

```bash
git clone https://github.com/ArchitRastogi20/contrastive-rejection-test.git
cd contrastive-rejection-test
pip install -r requirements.txt
python -m pytest tests -q                                          # 458 tests, seconds
python -m harness.analyze_run3 --results results --out table1.txt  # the paper's Table 1
```

Only the original generation runs needed a GPU. The one exception to "offline" is the
rebuilt-sentence audits (`audit_instrument_defects.py`'s figures 1 and 2, and
`audit_date_correctness.py`'s correctness-direction check), which also need a local JSONL dump of
the `framolfese/2WikiMultihopQA` validation split passed as `--data-file`. To build it once:

```python
from datasets import load_dataset
import json

ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
with open("dump.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(dict(row)) + "\n")
```

## Layout

Everything sits at the repository root:

```
harness/                       the experiment harness
├── extract.py                  deterministic extractor: rejections, named attributes, choice parser
├── repair.py                   builds conditions R0-R4, runs the eight integrity gates
├── data.py, prompts.py, models.py, config.py, watchdog.py    corpus loading, prompts, backends, paths
├── run_experiment.py           entry point: the repair experiment
├── run_necessity.py            entry point: necessity check (delete the chosen option's own attribute)
├── analyze_run3.py              -> Table 1: paired counts, OR/CI, twelve-test Holm ladder
├── analyze_round5.py            -> --section loo/heterogeneity/cluster/fluency/relation/control
├── audit_position_bias.py       -> R0 letter-uniformity chi-square, log-odds cross-check
├── audit_probe_missingness.py   -> forced-choice probe completion by model/part/condition
├── audit_location_confound.py   -> R3-R4 effect vs. the third option's own R0 baseline
├── audit_instrument_defects.py  -> the four quantified-not-fixed defect prevalences below
├── question_relevance.py        -> share of built items whose question is about the inserted attribute
├── audit_date_correctness.py    -> content contrasts inside/outside the date-repair stratum; does a switch land on the option the inserted date makes correct
├── analyze_relation_stratum.py  -> content contrasts split by whether the named attribute is parentage
├── analyze_necessity.py         offline analysis of the necessity run
└── surprisal.py, probe_variants.py, gate8_variant.py, decoding_sweep.py, reparse_partc.py,
    r3_recency.py, rescore.py, run_pilot.py    additional checks, see below

tests/                        458 tests, no GPU, no network, seconds to run
figures/make_figures.py       builds the paper's two figures from results/ (needs matplotlib, numpy)
scripts/                      setup_env.sh, prefetch scripts, monitor.sh, smoke.sh
results/                      run outputs only, one JSON/JSONL record per item per condition
├── exp/, exp2/                 two earlier runs, superseded but kept for checkable corrections
├── exp3a/, exp3b/, exp3c/      Parts A, B, C of the run this paper reports
├── e1/, e1_pooled/             fourth-lineage check (Phi-3.5-mini-instruct, Granite-3.0-8B-Instruct)
├── stage_necessity_*.jsonl     the necessity run (N0-N2)
└── surprisal___*.jsonl, probe_variants___*.jsonl, gate8_variant/, smoke/    other check outputs
requirements.txt              the 3090 Ti / vLLM environment
requirements-colab.txt        the T4 fallback (plain transformers, no vLLM)
.env.example                  names of environment variables the code reads, no values
```

`harness/config.py` resolves its paths relative to `harness/` itself, so the commands below
work from any checkout. Each one passes `--results`/`--out` explicitly so it stands alone, and
each was run from a fresh clone.

## Environment

Generation ran on one RTX 3090 Ti (24 GB), Python 3.11, PyTorch 2.4.0, CUDA 12.4.1, Ubuntu
22.04. `requirements.txt` pins `vllm==0.6.3.post1`, the last release built against torch 2.4.x.
**It predates the Qwen3 architecture, so no Qwen3 checkpoint runs on this stack.**
`requirements-colab.txt` is a T4 fallback (plain transformers, no vLLM, no bfloat16, a smaller
roster); none of the analysis below needs either environment.

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # 458 tests, no GPU, no network

python -m harness.analyze_run3 --results results --out /tmp/analysis_run3_report.txt   # Table 1
python -m harness.analyze_round5 --results results --out /tmp/round5.json --section loo            # leave-one-model-out
python -m harness.analyze_round5 --results results --out /tmp/round5.json --section heterogeneity
python -m harness.analyze_round5 --results results --out /tmp/round5.json --section fluency
python -m harness.audit_position_bias --results results --out /tmp/position_bias_report.txt        # letter effects
python -m harness.audit_probe_missingness --results results --out /tmp/probe_missingness_report.txt
python -m harness.audit_instrument_defects --self-check
python -m harness.audit_instrument_defects --results results   # the four quantified defects, no corpus needed
python -m harness.question_relevance --results results          # share of questions about the inserted attribute
python -m harness.audit_date_correctness --results results      # stratified content contrasts; add --data-file <2Wiki dump> for the correctness-direction check
```

Real generation needs one 24 GB card. `python -m harness.run_experiment --self-check` exercises
the eight integrity gates on fixtures with no GPU; `--dry-run --limit 6` exercises the pipeline
with a stub backend.

## Conditions

| | inserted content | edited profile |
|---|---|---|
| R0 | nothing | nothing: the choice must reproduce exactly under greedy decoding |
| R1 | the named attribute | the rival the model complained about |
| R2 | a different attribute, length-matched | the rival |
| R3 | the same sentence as R1 | a third option, never mentioned |
| R4 | the same sentence as R2 | that third option |

R0 is an integrity check: any flip there means generation was not deterministic. It returned zero
flips across 1,264 rows in nine model-part cells, so generation is deterministic **within a
run** at this scale. Re-asking the same items in a later session on byte-identical prompts is
not covered by that claim.

## Headline result

Of the twelve discrete contrasts across three runs (A: original roster, four options; B: three
untried checkpoints, four options; C: original roster, six options), four clear Holm correction:

| Part | Contrast | OR [95% CI] | Holm p |
|---|---|---|---|
| A | R1-R2 (content at the named rival) | 3.57 [1.54, 8.26] | **0.0210** |
| C | R3-R4 (content at the unnamed third option) | 4.40 [1.67, 11.62] | **0.0167** |
| C | R2-R4 (irrelevant control across both locations) | 4.00 [1.93, 8.30] | **0.0008** |
| C | R1-R3 (relevant content, named vs. unnamed location) | 2.04 [1.26, 3.29] | **0.0345** |

**The contrast this design was built to detect, content at the option nobody named (Part A's
R3-R4), does not clear correction**, Holm p = 0.2428. C R3-R4 clears it but rests on one model:
dropping Mistral-7B-Instruct-v0.3 alone takes it to p = 0.30. A R1-R2 is the most defensible
survivor, holding under every single-model exclusion. Part A supports a content effect at the
named location and Part C at the unnamed one; no run supports both. `harness/analyze_run3.py`
reproduces this table exactly, seeded, from `results/exp3a`, `exp3b`, `exp3c`.

The two measures disagree: the discrete (choice) and continuous (forced-choice probability)
reads agree in sign on all six content-contrast cells but diverge on four of six location cells.
Three candidate explanations for the disagreement (a ceiling effect, an argmax discarding
within-ranking movement, a few large moves dominating a mean) were tested; none is supported.
A second roster of three untried checkpoints shares a Qwen lineage with the model driving most of
the original effect, so no independently trained third model has yet reproduced it on the choice
measure; a separate check on two more independent lineages (Phi-3.5-mini-instruct,
Granite-3.0-8B-Instruct) reproduces the content effect only on the continuous measure.

## Known defects

An adversarial audit of the pipeline, run against itself, found eight defects. Three were
corrected and are reflected in every number above; five are quantified but not fixed, since
fixing them needs fresh generation runs.

**Corrected:**
1. The choice parser returned the option a model had just rejected as its chosen one, in 17.1%
   of adjudicable responses (497/2,909).
2. Its letter-match regex was capped at four options while Part C offers six, leaving 155 of
   Part C's stage-3 rows unreadable; widening it recovers 134 of them.
3. It required a chosen option's trailing parenthetical disambiguator ("(director)") to appear
   in the model's restated answer; when the model dropped it, a rejected rival's unqualified
   title could win instead: 21 of 2,960 adjudicable stage-3 rows, 16.2% of the 130 exposed.

**Quantified, not fixed:**
4. Gate 3's word-count length check cannot fail on a built item by construction, since the
   integrity check tests the same metric the search that builds the item already minimises
   (confirmed on 1,070 independently rebuilt items; a real-tokenizer check needs a tokenizer
   this project cannot cache offline).
5. Gate 5's verbatim-title check misses a co-candidate mention at 28.0% on the relevant-content
   axis against 4.3% on the irrelevant one, since the relevant sentence is drawn from a sibling
   profile and the irrelevant one is not.
6. The attribute-naming cue misclassifies 88.2% (112/127) of parentage-attribute items,
   matching a word with no family sense or matching the wrong generation (grandfather or
   grandmother).
7. 15.7% (199/1,264) of built items rest a rejection on ranking a rival below the choice rather
   than asserting an absence. Excluding them and recomputing the twelve-test Holm ladder flips
   one verdict: Part C's R1-R3 no longer clears correction (Holm 0.0345 to 0.3524).
8. `extract.attribute_in_profile` matches its cue as an unbounded substring, so a sentence like
   "studied ... 1934" passes as a date of death: 6 of 648 date-repair sentences (R1 and R3, over
   the 1,070 rebuildable built items) carry the cue only that way.

`harness/audit_instrument_defects.py` recomputes 4-7 from committed JSONL alone (4 and 5 also
replay `repair.py`'s own condition-building call, verified byte-identical against the committed
option titles first). `harness/audit_date_correctness.py` recomputes 8, given the local corpus
dump described above. 1 and 2 are in `harness/extract.py`'s own `parse_choice`/`_letter_pick_re`;
3 is `_base_title`/`_TRAILING_PAREN` in the same file; 8 is `attribute_in_profile`'s unbounded
`cue in text` test, also in `extract.py`.

## Other checks in `harness/`

- **`audit_position_bias.py`**: R0 rejects a uniform letter distribution in Parts A and C
  ($\chi^2$, p<0.001); content contrasts read the same direction in log-odds as in probability.
- **`audit_location_confound.py`**: tests whether R3-R4 tracks the third option's own lower
  starting probability rather than the inserted content. Most matched bands are too small to
  read (34-94 items against roughly 728 needed for 80% power).
- **`audit_probe_missingness.py`**: the forced-choice probe never completes for one reasoning
  model and completes on only 55.4% to 32.8% of one other model's rows as option count grows,
  non-randomly by condition in two model-part cells.
- **`question_relevance.py`**: 937/1,264 built items (74.1%) ask a date- or order-comparison
  question, so the inserted fact's absence, not its truth, is usually what the question turns on.
- **`audit_date_correctness.py`**: recomputes the content contrasts inside and outside the
  stratum of order questions with a date repair (594 built items), and, given a local corpus
  dump, rebuilds each inserted sentence and asks whether the inserted year makes the edited
  option correct under the question, and separately counts date-repair sentences whose
  attribute cue is only embedded in a longer word (defect 8 above). Every rule is a printed
  regular expression with a sample.
- **`analyze_relation_stratum.py`**: the length-matched irrelevant control usually states a
  parentage relation regardless of what the model named, confounding relevance with relation
  type except where the model's own named attribute is itself parentage.
- **`run_necessity.py` / `analyze_necessity.py`**: deleting the chosen option's own stated
  attribute moves the model off its choice far more than a length-matched control deletion,
  McNemar b=7, c=33, OR 0.21 [0.09, 0.48], holding in all three models individually.
- **`surprisal.py`**: the relevant insertion is about 1.2 nats/token less surprising than its
  control in every model, so the control matches length but not plausibility.
- **`gate8_variant.py`, `decoding_sweep.py`, `probe_variants.py`, `reparse_partc.py`,
  `r3_recency.py`, `rescore.py`**: alternate target strategies, non-zero-temperature resampling,
  alternate probe prompts, a Part C reparse, and offline rechecks against saved raw responses.

## Limitations of the artifact

This work spans one corpus (2WikiMultihopQA, validation split) and one language. Four relation
types (dates of birth/death, parentage, director credits) account for 98% of built items. Every
rejection tested is elicited, not spontaneous. The probability probe's coverage is model- and
condition-dependent (above), so continuous results are complete-case under a non-random
missingness pattern for at least one model.

## Citation

```bibtex
@misc{rastogi2026rejection,
  author = {Rastogi, Archit},
  title  = {Does a model's stated reason for rejecting a candidate do any work?},
  year   = {2026},
  url    = {https://github.com/ArchitRastogi20/contrastive-rejection-test}
}
```

## License

MIT, see `LICENSE`.
