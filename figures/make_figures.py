"""Regenerate every figure this paper uses.

Run from the repository root or from ``code/``::

    python code/figures/make_figures.py
    python code/figures/make_figures.py --results code/results --out paper_full/figures

Figure 1 is a forest plot recomputed from the committed stage-3 JSONL under
``code/results/exp3a``, ``exp3b``, ``exp3c`` via ``harness.analyze_run3``; no network needed.

Figure 2 is a design schematic whose worked example is reconstructed end to end from a real
item's stage-1/stage-2 records (``code/results/exp2``, ``code/results/exp3a``) using
``harness.data`` and ``harness.repair``, and asserts the reconstruction matches the committed
record. This step needs network access once, to pull the source dataset
(``framolfese/2WikiMultihopQA``) from the Hugging Face Hub.

Output: vector PDF, sized to the paper template's single-column body text width (452.9679pt = 6.2679in).

Dependencies: matplotlib, numpy, and (for Figure 2's worked example only) ``datasets``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------- typography
# The paper template's body font is Libertinus, not available in this environment's font
# cache; STIXGeneral is the closest metrics-compatible serif actually installed
# (it was built as a Times-metric-compatible math/text face), with DejaVu Serif
# and the system serif fallback behind it so the script degrades gracefully
# elsewhere. text.usetex is deliberately left off: it would need a LaTeX install
# with the *same* Libertinus fonts on the rendering machine to help at all, and
# would fail loudly (not gracefully) if that is not this machine's LaTeX.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.3,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,   # embed as real (subsettable) fonts, not paths
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

TEXTWIDTH_IN = 452.9679 / 72.27  # measured against paper_full/ceurart.cls, see module docstring

# bbox_inches="tight" crops to the actual ink extent, which for the forest plot runs
# ~1.6% wider than the nominal figsize because the bottom legend's end labels overhang
# the axes box slightly. Shrinking the canvas by this measured factor before that crop
# lands the final, saved PDF at (not under) TEXTWIDTH_IN, so \includegraphics with no
# width= renders it at exactly the body text width -- no LaTeX-side rescaling, hence no
# font-size mismatch with the surrounding text. Re-measure this if the plot layout changes.
FOREST_WIDTH_SCALE = 0.9845

# --------------------------------------------------------------------------- data plumbing

CONTRASTS = [("R1", "R2", "content, rival"),
             ("R3", "R4", "content, third"),
             ("R1", "R3", "location, relevant"),
             ("R2", "R4", "location, irrelevant")]
PARTS = ["A", "B", "C"]


def _load_harness():
    """Import code/harness/analyze_run3.py regardless of the caller's cwd."""
    here = Path(__file__).resolve()
    code_dir = here.parent.parent  # code/
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from harness import analyze_run3 as ar3  # noqa: E402
    return ar3


def compute_contrast_table(results_dir: Path):
    """Pooled OR/RD/Δp for the 4 contrasts x 3 parts, plus the 12-test Holm correction.

    Mirrors exactly the "pooled" rows of the run-3 analysis report's section 1: same
    ``load_part``/``discrete_outcomes``/``discrete_contrast``/``paired_mean`` calls,
    same seed (module default, ``SEED = 20260822``), same resample count.
    """
    ar3 = _load_harness()
    rows = []
    raw_p = {}
    for part in PARTS:
        items = ar3.load_part(results_dir, part)
        pooled_items = {f"{m}::{i}": conds for m, its in items.items() for i, conds in its.items()}
        disc = ar3.discrete_outcomes(pooled_items)
        cont = {
            item_id: {c: {"delta_p_edited": row.get("delta_p_edited")} for c, row in conds.items()}
            for item_id, conds in pooled_items.items()
        }
        for a, b, why in CONTRASTS:
            d = ar3.discrete_contrast(disc, a, b)
            cm = ar3.paired_mean(cont, a, b, "delta_p_edited")
            key = f"{part} {a}-{b}"
            raw_p[key] = d["p_exact_two_sided"]
            rows.append({
                "part": part, "contrast": f"{a}-{b}", "why": why,
                "or": d["or"]["or"], "or_lo": d["or"]["lo"], "or_hi": d["or"]["hi"],
                "b": d["b_a_only"], "c": d["c_b_only"], "p_exact": d["p_exact_two_sided"],
                "dp": cm["mean"], "dp_lo": cm["ci_low"], "dp_hi": cm["ci_high"],
                "n_disc": d["n_paired"], "n_cont": cm["n"],
            })
    adj = ar3.holm(raw_p)
    for r in rows:
        r["p_holm"] = adj[f"{r['part']} {r['contrast']}"]
        r["holm_survives"] = r["p_holm"] < 0.05
    return rows


# --------------------------------------------------------------------------- Figure 1


def make_forest_plot(rows: list[dict], out_path: Path):
    """Odds ratio (log2 scale) and Δp, 4 contrasts x 3 parts, pooled across models.

    Replaces Table 3 (which covers only Parts A and C) and folds in Part B's
    numbers, currently reported only in running prose in Section 4.3. Solid
    markers on the odds-ratio panel are the five contrasts that survive Holm
    correction over the twelve-test family reported in the same section; open
    markers do not. The Δp panel carries no such flag -- the paper's Holm
    correction is declared over the discrete family only.
    """
    order = [(a, b) for a, b, _ in CONTRASTS]
    row_h = 1.0
    group_gap = 0.55
    y = 0.0
    ys = {}
    for a, b in order:
        for part in PARTS:
            ys[(a, b, part)] = y
            y -= row_h
        y -= group_gap
    y_top = row_h * 0.65
    y_bottom = y + group_gap - row_h * 0.35

    by_key = {(r["contrast"].split("-")[0], r["contrast"].split("-")[1], r["part"]): r for r in rows}

    fig = plt.figure(figsize=(TEXTWIDTH_IN * FOREST_WIDTH_SCALE, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.30, 0.35, 0.35], wspace=0.06)
    ax_lab = fig.add_subplot(gs[0, 0])
    ax_or = fig.add_subplot(gs[0, 1])
    ax_dp = fig.add_subplot(gs[0, 2], sharey=ax_or)

    for ax in (ax_lab, ax_or, ax_dp):
        ax.set_ylim(y_bottom, y_top)

    # ---- label column
    ax_lab.axis("off")
    group_labels = {
        ("R1", "R2"): "R1$-$R2\ncontent, rival",
        ("R3", "R4"): "R3$-$R4\ncontent, third",
        ("R1", "R3"): "R1$-$R3\nlocation, relevant",
        ("R2", "R4"): "R2$-$R4\nlocation, irrelevant",
    }
    for a, b in order:
        rowys = [ys[(a, b, part)] for part in PARTS]
        ycenter = sum(rowys) / len(rowys)
        ax_lab.text(1.0, ycenter, group_labels[(a, b)], ha="right", va="center",
                    fontsize=7.6, fontweight="bold", linespacing=1.35, transform=ax_lab.transData)
        # thin separator above each group (except the first)
    for part in PARTS:
        pass
    for a, b in order:
        for part in PARTS:
            ax_lab.text(1.35, ys[(a, b, part)], part, ha="left", va="center", fontsize=7.0,
                        transform=ax_lab.transData)
    ax_lab.set_xlim(0, 3)

    # group separators across all three axes
    for i in range(1, len(order)):
        sep_y = (ys[(order[i][0], order[i][1], "A")] + row_h * 0.65 +
                 ys[(order[i-1][0], order[i-1][1], "C")] - row_h * 0.65) / 2
        for ax in (ax_lab, ax_or, ax_dp):
            ax.axhline(sep_y, color="0.75", lw=0.5, xmin=-10, xmax=10, clip_on=False)

    # ---- odds ratio panel (log2 x-axis)
    ax_or.axvline(1.0, color="0.4", lw=0.7, ls="--", zorder=1)
    for a, b in order:
        for part in PARTS:
            r = by_key[(a, b, part)]
            yy = ys[(a, b, part)]
            filled = r["holm_survives"]
            ax_or.plot([r["or_lo"], r["or_hi"]], [yy, yy], color="black", lw=0.9, zorder=2)
            ax_or.plot([r["or"]], [yy],
                       marker="o", markersize=4.6,
                       markerfacecolor="black" if filled else "white",
                       markeredgecolor="black", markeredgewidth=0.8, zorder=3)
    ax_or.set_xscale("log", base=2)
    ax_or.set_xlim(0.42, 11.5)
    ax_or.set_xticks([0.5, 1, 2, 4, 8])
    ax_or.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax_or.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_or.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_or.spines[spine].set_visible(False)
    ax_or.spines["bottom"].set_position(("data", y_bottom))
    ax_or.tick_params(axis="x", labelsize=7.0)
    ax_or.set_title("odds ratio (choice measure)\nlog scale, matched-pairs", fontsize=7.6, pad=4)
    ax_or.set_xlabel("odds ratio, 95% CI", fontsize=7.0, labelpad=2)

    # ---- delta-p panel (linear x-axis)
    ax_dp.axvline(0.0, color="0.4", lw=0.7, ls="--", zorder=1)
    for a, b in order:
        for part in PARTS:
            r = by_key[(a, b, part)]
            yy = ys[(a, b, part)]
            ax_dp.plot([r["dp_lo"], r["dp_hi"]], [yy, yy], color="black", lw=0.9, zorder=2)
            ax_dp.plot([r["dp"]], [yy], marker="D", markersize=3.8,
                       markerfacecolor="black", markeredgecolor="black", markeredgewidth=0.6, zorder=3)
    ax_dp.set_xlim(-0.065, 0.095)
    ax_dp.set_xticks([-0.04, 0.0, 0.04, 0.08])
    ax_dp.set_xticklabels(["-0.04", "0", "+0.04", "+0.08"])
    ax_dp.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_dp.spines[spine].set_visible(False)
    ax_dp.spines["bottom"].set_position(("data", y_bottom))
    ax_dp.tick_params(axis="x", labelsize=7.0)
    ax_dp.set_title(r"$\Delta p_{\mathrm{edited}}$ (probability measure)" + "\nmean paired difference",
                     fontsize=7.6, pad=4)
    ax_dp.set_xlabel(r"$\Delta p$, 95% CI", fontsize=7.0, labelpad=2)

    # legend
    handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", markeredgecolor="black",
               lw=0.9, markersize=4.6, label="Holm-adjusted $p<0.05$ (choice)"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgecolor="black",
               lw=0.9, markersize=4.6, label="does not survive"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="black", markeredgecolor="black",
               lw=0.9, markersize=3.8, label=r"$\Delta p_{\mathrm{edited}}$ (uncorrected)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.5, columnspacing=1.2)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.20)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 2 data

# The one (item_id, model) pair Figure 2 illustrates, chosen for a short, ASCII-only question
# and profiles, and because R1's own sentence is visibly borrowed from a sibling option in the
# very same item (asserted below, not just eyeballed). Both fields belong in the figure's own
# provenance comment wherever it is captioned, exactly like every number in the paper.
EXAMPLE_ITEM_ID = "f8aa0d06086311ebbd5eac1f6bf848b6"
EXAMPLE_MODEL_DIR_TAG = "Qwen2_5-7B-Instruct"   # matches the stage1/2/3 file name suffix
EXAMPLE_PART_DIR = "exp3a"                      # Part A: 4 options, the original 3-model roster
EXAMPLE_STAGE1_SOURCE_DIR = "exp2"              # Part A replays stage 1 from here (main.tex Sec. 3)


def _read_jsonl(path: Path) -> list[dict]:
    import json
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_worked_example(results_dir: Path) -> dict:
    """Reconstruct one real item end to end, for Figure 2's worked example.

    See the module docstring for why this is the only part of this file that touches the
    network, and what it verifies. Every field returned traces to either a committed JSONL row
    (checked, never assumed) or a deterministic rebuild of the source dataset that is asserted
    to agree with that row before anything is returned. A mismatch raises; it does not fall back
    to a plausible-looking guess.
    """
    _load_harness()  # puts code/ on sys.path
    from harness.data import load_records, build_items
    from harness import repair as repair_mod
    from harness.extract import Rejection
    from harness import config as C

    stage1_path = (results_dir / EXAMPLE_STAGE1_SOURCE_DIR
                   / f"stage1___workspace___hf__models__{EXAMPLE_MODEL_DIR_TAG}.jsonl")
    stage2_path = (results_dir / EXAMPLE_PART_DIR
                   / f"stage2___workspace___hf__models__{EXAMPLE_MODEL_DIR_TAG}.jsonl")
    stage3_path = (results_dir / EXAMPLE_PART_DIR
                   / f"stage3___workspace___hf__models__{EXAMPLE_MODEL_DIR_TAG}.jsonl")

    s1_row = next(r for r in _read_jsonl(stage1_path) if r["item_id"] == EXAMPLE_ITEM_ID)
    s2_row = next(r for r in _read_jsonl(stage2_path) if r["item_id"] == EXAMPLE_ITEM_ID)
    s3_rows = {r["condition"]: r for r in _read_jsonl(stage3_path) if r["item_id"] == EXAMPLE_ITEM_ID}
    assert s2_row["built"], f"{EXAMPLE_ITEM_ID} is not a built item for {EXAMPLE_MODEL_DIR_TAG}"
    assert s1_row["choice"] is not None, "example item's stage-1 choice is unreadable"

    rej_match = next(
        rej for rej in s1_row["rejections"]
        if rej["title"] == s2_row["rival_title"] and rej["attribute"] == s2_row["attribute"]
    )
    rival_letter = rej_match["letter"]

    # The same fixed 400-item reference pool the real Part A run built once, up front
    # (run_experiment.log: "loading framolfese/2WikiMultihopQA [validation], scanning 16000
    # records" / "built 400 items from 12576 records"), and the same corpus_index built from
    # it -- which is what actually supplies R2's cross-item irrelevant sentence in that run,
    # regardless of which larger per-model item pool is being scored that run.
    records = load_records(C.DATASET_ID, C.DATASET_SPLIT, 16000)
    items = build_items(records, n_items=400, n_options=4, seed=C.SEED)
    by_id = {it.item_id: it for it in items}
    item = by_id[EXAMPLE_ITEM_ID]
    assert item.question == s1_row["question"], "reconstructed question does not match the run"
    assert item.gold_letter == s1_row["gold_letter"], "reconstructed gold letter does not match"
    assert [o.title for o in item.options] == s1_row["option_titles"], (
        "reconstructed option order does not match the committed run -- the dataset or the "
        "seeded shuffle has drifted since this example was picked"
    )
    corpus_index = repair_mod.build_corpus_index(items)

    # Part A's own run used PILOT_PREFER_ABOVE=3, not config.py's own default of 4 (main.tex
    # Sec. 3: "Preferring a candidate that already lacks the named attribute
    # (PILOT_PREFER_ABOVE=3)"). Reproducing the default here would silently pick a different
    # R3/R4 target than the run actually used.
    C.PREFER_LACKING_TARGET_ABOVE = 3

    rejection = Rejection(
        sentence=rej_match["sentence"], letter=rival_letter, title=s2_row["rival_title"],
        matched_by=rej_match.get("matched_by", "token"), attribute=s2_row["attribute"],
        cue=rej_match.get("cue"), negation_cue=rej_match.get("negation_cue"),
    )
    conditions, diag = repair_mod.build_conditions_with_diagnostics(
        item, rejection, corpus_index, choice_letter=s1_row["choice"], n_options=4,
    )

    # Prove it: four independent checks against the committed record, not a recomputation of
    # the same code path against itself.
    assert diag.r3_picked_letter == s2_row["r3_picked_letter"], (
        f"reconstructed R3 target {diag.r3_picked_letter!r} != "
        f"committed {s2_row['r3_picked_letter']!r}"
    )
    for cond in ("R1", "R2", "R3", "R4"):
        expected_letter = rival_letter if cond in ("R1", "R2") else diag.r3_picked_letter
        real_letter = s3_rows[cond]["edited_letter"]
        assert real_letter == expected_letter, f"{cond}: recomputed {expected_letter} != run's own {real_letter}"

    def added_sentence(cond: str, option_letter: str) -> str:
        idx = ord(option_letter) - ord("A")
        edited_profile = conditions[cond].options[idx].profile
        original_profile = item.options[idx].profile
        assert edited_profile.startswith(original_profile), f"{cond} {option_letter}: not a simple append"
        return edited_profile[len(original_profile):].strip()

    rival_idx = ord(rival_letter) - ord("A")
    third_letter = diag.r3_picked_letter
    third_idx = ord(third_letter) - ord("A")

    return {
        "item_id": EXAMPLE_ITEM_ID,
        "model": EXAMPLE_MODEL_DIR_TAG,
        "question": item.question,
        "gold_letter": item.gold_letter,
        "gold_title": item.gold_title,
        "attribute": s2_row["attribute"],
        "rejection_sentence": rej_match["sentence"],
        "rival_letter": rival_letter,
        "rival_title": item.options[rival_idx].title,
        "rival_profile": item.options[rival_idx].profile,
        "third_letter": third_letter,
        "third_title": item.options[third_idx].title,
        "third_profile": item.options[third_idx].profile,
        "r1_sentence": added_sentence("R1", rival_letter),
        "r2_sentence": added_sentence("R2", rival_letter),
        "r3_sentence": added_sentence("R3", third_letter),
        "r4_sentence": added_sentence("R4", third_letter),
        "r1_flipped": s3_rows["R1"]["chosen_is_edited"],
        "r3_flipped": s3_rows["R3"]["chosen_is_edited"],
    }


# --------------------------------------------------------------------------- Figure 2


def make_design_schematic(example: dict, out_path: Path):
    """The five conditions (R0-R4) as a 2x2 of edit content x edit location, plus R0.

    Draws no experimental number -- see module docstring -- but the worked example is a real
    item (``example``, from ``load_worked_example``), not an invented one: the question, the
    model's actual rejection, the rival it named, and the real R1/R2 sentences it caused to be
    inserted. No corpus wording is cleaned up; real formatting quirks in the source text (the
    stray space in "full- length" and "A- Day" below) are reproduced exactly as the corpus has
    them.

    Each edited-location column (the named rival, the unnamed third option) shows its option's
    profile once, followed by its two insertions (relevant-sentence, then irrelevant-sentence)
    stacked beneath it, each carrying its own R-label and content-type tag. An earlier version
    of this figure printed each column's profile twice -- once per row of the 2x2 -- which cost
    about a third of the figure's height in text the reader had already read; the 2x2 reading
    (content x location) survives here through the R1/R2 vs. R3/R4 column split plus the
    per-insertion tint and label, not through repeating the profile. R0 -- which edits nothing --
    is a short footnote-style strip below the grid rather than a full-height column, since it
    only ever holds one short sentence.

    Card and strip heights are computed from the actual wrapped line counts (below) rather than
    guessed and fixed, because a fixed guess overlapped body text with the inserted-sentence
    highlight whenever a column's real profile and its real inserted sentences were both long at
    once -- caught by rendering and looking, not by eyeballing the code.
    """
    import textwrap

    def wrap(text: str, width: int) -> str:
        return "\n".join(textwrap.wrap(text, width=width)) if text else ""

    BODY_FS = 6.6
    # The option letter/title is now just the first wrapped line of the profile paragraph
    # itself (below), not a separate bold heading above it, so the top gap only has to clear
    # the card border, not reserve room for a heading line the way the old per-condition cards
    # (each with its own bold "R1 (the repair)" title) needed.
    BODY_TOP_GAP = 0.30       # box top edge to first profile line
    RULE_GAP_ABOVE = 0.08     # gap from the end of a text block to the separating rule
    RULE_GAP_BELOW = 0.07     # gap from the rule to the next section's label
    LABEL_GAP = 0.09          # gap from a section's one-line label to its inserted sentence
    BOTTOM_PAD = 0.16         # clears the highlight bbox's own padding at the card's bottom edge
    PROFILE_WRAP = 46         # chars/line; wider than the old 4-cell layout since freeing R0's
    INSERT_WRAP = 42          # own column gives both remaining columns more width to wrap into

    def wrap_title_profile(letter: str, title: str, profile: str) -> str:
        # Title and profile are wrapped as two separate paragraphs, not one flowed string --
        # textwrap collapses embedded newlines along with all other whitespace, so wrapping
        # them together silently ran the option title into the first sentence of its own
        # profile whenever the title happened not to fall on a word boundary.
        return wrap(f"{letter}) {title}", PROFILE_WRAP) + "\n" + wrap(profile, PROFILE_WRAP)

    rival_body = wrap_title_profile(example["rival_letter"], example["rival_title"], example["rival_profile"])
    third_body = wrap_title_profile(example["third_letter"], example["third_title"], example["third_profile"])
    r1_added = "+ " + wrap(f"“{example['r1_sentence']}”", INSERT_WRAP)
    r2_added = "+ " + wrap(f"“{example['r2_sentence']}”", INSERT_WRAP)
    r3_added = "+ " + wrap(f"“{example['r3_sentence']}”", INSERT_WRAP)
    r4_added = "+ " + wrap(f"“{example['r4_sentence']}”", INSERT_WRAP)

    # Each column's two sections are drawn in the same order (relevant, then irrelevant), so the
    # relevant/irrelevant tag lands in the same relative slot in both columns even though the two
    # columns' profiles differ in length (the rival's runs three lines, the third option's one).
    riv_sections = [("R1  (the repair)", "relevant sentence", r1_added, "0.78"),
                     ("R2  (control)", "irrelevant sentence", r2_added, "0.92")]
    third_sections = [("R3", "relevant sentence", r3_added, "0.78"),
                       ("R4", "irrelevant sentence", r4_added, "0.92")]

    q_wrapped = wrap(f"“{example['question']}”", 92)
    rej_wrapped = wrap(
        f"model chose {example['gold_letter']}) {example['gold_title']}, and said of "
        f"{example['rival_letter']}: “{example['rejection_sentence']}”", 100
    )
    n_q_lines = q_wrapped.count("\n") + 1
    n_rej_lines = rej_wrapped.count("\n") + 1

    # ---- geometry, worked out from content, not guessed ----
    # Figure width is fixed (the paper's textwidth); figure height is derived below from how
    # much vertical room the real wrapped text actually needs, so this stays correct if a
    # future example item has a longer or shorter profile or inserted sentence.
    FIG_W_IN = TEXTWIDTH_IN
    FIG_H_IN = 4.6                      # provisional; only the axis-unit/inch ratio matters below
    Y_TOP = 10.0                        # provisional ylim top; rescaled to the real content below
    IN_PER_UNIT = FIG_H_IN / Y_TOP
    LINE_UNIT = (BODY_FS * 1.18 / 72.0) / IN_PER_UNIT
    # The R-label and its "relevant/irrelevant sentence" tag sit side by side on one line (see
    # column_card below), not stacked on two -- this reserves exactly the one line they use.
    # 7.4pt matches the old per-condition card title's own size, not shrunk from it.
    LABEL_LINE_UNIT = (7.4 * 1.15 / 72.0) / IN_PER_UNIT

    def column_height(body_text: str, sections) -> float:
        h = BODY_TOP_GAP + (body_text.count("\n") + 1) * LINE_UNIT
        for _, _, added_text, _ in sections:
            h += RULE_GAP_ABOVE + RULE_GAP_BELOW
            h += LABEL_LINE_UNIT + LABEL_GAP
            h += (added_text.count("\n") + 1) * LINE_UNIT
        return h + BOTTOM_PAD

    box_h = max(column_height(rival_body, riv_sections), column_height(third_body, third_sections))

    banner_line_unit = (6.6 * 1.15 / 72.0) / IN_PER_UNIT
    banner_h = 0.08 + n_q_lines * banner_line_unit + 0.05 + n_rej_lines * banner_line_unit
    header_h = 0.58
    r0_gap = 0.12
    r0_h = 0.40
    bottom_margin = 0.12
    top_margin = 0.10

    content_h = top_margin + banner_h + header_h + box_h + r0_gap + r0_h + bottom_margin
    # Rescale the provisional canvas so 1 axis-unit really does equal IN_PER_UNIT inches at the
    # figure height this content needs -- solved directly rather than iterated, since box_h
    # depends on IN_PER_UNIT and content_h depends on box_h, both linearly.
    FIG_H_IN = 4.35 * (content_h / 7.6)  # keeps the same inches-per-unit as the first pass
    Y_TOP = content_h

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, Y_TOP)
    ax.axis("off")

    def column_card(x, y, w, h, profile_text, sections):
        """One option's profile, drawn once, with its insertions stacked beneath it."""
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                        linewidth=0.9, edgecolor="black", facecolor="white", zorder=2)
        ax.add_patch(rect)
        cur_top = y + h - BODY_TOP_GAP
        ax.text(x + 0.12, cur_top, profile_text, ha="left", va="top", fontsize=BODY_FS, zorder=3,
                family="monospace", linespacing=1.18)
        cur_top -= (profile_text.count("\n") + 1) * LINE_UNIT
        for r_label, kind_label, added_text, tint in sections:
            cur_top -= RULE_GAP_ABOVE
            ax.plot([x + 0.10, x + w - 0.10], [cur_top, cur_top], color="0.75", lw=0.6, zorder=3)
            cur_top -= RULE_GAP_BELOW
            # r_label and kind_label sit side by side on this one line, not stacked -- the
            # cursor below advances by a single LABEL_LINE_UNIT, matching what is actually drawn.
            ax.text(x + 0.12, cur_top, r_label, ha="left", va="top", fontsize=7.4,
                    fontweight="bold", zorder=3)
            ax.text(x + w - 0.12, cur_top, kind_label, ha="right", va="top", fontsize=7.0,
                    style="italic", zorder=3)
            cur_top -= LABEL_LINE_UNIT + LABEL_GAP
            ax.text(x + 0.12, cur_top, added_text, ha="left", va="top",
                    fontsize=BODY_FS, zorder=3, family="monospace", fontweight="bold",
                    linespacing=1.18, bbox=dict(boxstyle="round,pad=0.20", fc=tint, ec="none"))
            cur_top -= (added_text.count("\n") + 1) * LINE_UNIT
        return rect

    left_margin, col_gap, right_margin = 0.35, 0.45, 0.35
    box_w = (10.0 - left_margin - right_margin - col_gap) / 2.0
    riv_x = left_margin
    third_x = left_margin + box_w + col_gap
    col_riv_center = riv_x + box_w / 2
    col_third_center = third_x + box_w / 2

    box_top = Y_TOP - top_margin - banner_h - header_h
    box_y = box_top - box_h

    y_banner_top = Y_TOP - top_margin
    ax.text(5.0, y_banner_top, q_wrapped, ha="center", va="top", fontsize=6.8, style="italic")
    ax.text(5.0, y_banner_top - banner_line_unit * n_q_lines - 0.05, rej_wrapped,
            ha="center", va="top", fontsize=6.6, style="italic", linespacing=1.15)

    header_y = box_top + 0.40
    subheader_y = box_top + 0.14
    ax.text(5.0, header_y, "edited profile (each shown once, insertions stacked beneath)",
            ha="center", va="center", fontsize=8.2, fontweight="bold")
    ax.text(col_riv_center, subheader_y, f"the named rival ({example['rival_letter']})",
            ha="center", va="center", fontsize=7.2, style="italic")
    ax.text(col_third_center, subheader_y, f"an unnamed third option ({example['third_letter']})",
            ha="center", va="center", fontsize=7.2, style="italic")

    column_card(riv_x, box_y, box_w, box_h, rival_body, riv_sections)
    column_card(third_x, box_y, box_w, box_h, third_body, third_sections)

    r0_y = box_y - r0_gap - r0_h
    rect = mpatches.FancyBboxPatch((riv_x, r0_y), box_w + col_gap + box_w, r0_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=0.8, edgecolor="0.4", facecolor="0.94", zorder=2, ls="--")
    ax.add_patch(rect)
    ax.text(riv_x + 0.14, r0_y + r0_h / 2, "R0", ha="left", va="center", fontsize=7.6,
            fontweight="bold", zorder=3)
    ax.text(riv_x + 0.62, r0_y + r0_h / 2,
            "nothing edited. integrity check: the model must repeat its original choice exactly.",
            ha="left", va="center", fontsize=6.6, style="italic", zorder=3)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    # here.parent is the directory holding harness/ and figures/ as siblings -- code/ in the
    # working tree, the repo root itself in the flat public release. here.parent.parent is only
    # meaningful when here.parent is actually named "code" (the working tree, where paper_full/
    # is a sibling of code/); the flat release has no paper_full/ at all, so falling back to
    # here.parent there keeps the default inside the cloned repository instead of writing above
    # it. See harness/config.py's CODE_ROOT/REPO_ROOT for the same layout-detection logic.
    code_root = here.parent
    repo_root = code_root.parent if code_root.name == "code" else code_root
    ap.add_argument("--results", type=Path, default=code_root / "results",
                     help="results tree holding exp3a/exp3b/exp3c (default: code/results)")
    ap.add_argument("--out", type=Path, default=repo_root / "paper_full" / "figures",
                     help="output directory for the rendered PDFs (default: paper_full/figures)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = compute_contrast_table(args.results)
    forest_path = args.out / "fig_contrasts_forest.pdf"
    make_forest_plot(rows, forest_path)
    print(f"wrote {forest_path}")

    example = load_worked_example(args.results)
    print(f"\nworked example for Figure 2: item {example['item_id']} / {example['model']} "
          f"(rival {example['rival_letter']}={example['rival_title']!r}, "
          f"third {example['third_letter']}={example['third_title']!r}, "
          f"attribute={example['attribute']!r}); "
          f"R1 flipped={example['r1_flipped']}, R3 flipped={example['r3_flipped']}")
    schem_path = args.out / "fig_design_schematic.pdf"
    make_design_schematic(example, schem_path)
    print(f"wrote {schem_path}")

    # a quick, human-readable dump of what got plotted, for sanity-checking against
    # the run-3 analysis and interaction-test reports by eye
    print("\ncontrast table used for Figure 1 (part, contrast, OR [CI], Holm p, dp [CI]):")
    for r in rows:
        print(f"  {r['part']} {r['contrast']:6s} OR={r['or']:.2f} [{r['or_lo']:.2f},{r['or_hi']:.2f}] "
              f"holm_p={r['p_holm']:.4f} survives={r['holm_survives']} "
              f"dp={r['dp']:+.4f} [{r['dp_lo']:+.4f},{r['dp_hi']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
