"""Offline bounds on the three unfixed confounds of the content contrasts.

The repair sentence (R1, R3) and its irrelevant control (R2, R4) differ on more than the named
attribute: the repair names a co-candidate far more often, the control states a parentage
relation far more often, and a few date repairs pass the presence check only through a cue
embedded in a longer word. None of this can be fixed without fresh generation, but each can be
bounded on the items whose inserted sentences can be rebuilt offline: recompute the content
contrasts R1-R2 and R3-R4 on the stratum where the two sentences of the pair agree on the
confound, and see whether the effect survives.

Strata, each over the rebuilt items only (so the "rebuilt" row is the comparison baseline):

- ``no_co_candidate``: neither sentence of the pair names another option
  (``audit_instrument_defects._mentions_co_candidate``, the rule behind the 28.0% vs 4.3%).
- ``template_matched``: both sentences state a parentage relation, or neither does
  (``states_parentage``, a whole-word rule; the harness's own cue matches by substring).
- ``date_cue_clean``: the relevant sentence does not pass as a date only through an
  embedded cue (``audit_date_correctness.cue_only_embedded``).
- ``all_three``: the intersection.

    python -m harness.audit_confound_bounds --data-file dump.jsonl [--json out.json]

No GPU, no network, no judge. Exact McNemar p is uncorrected: these are sensitivity checks on
cells whose Holm verdict is already reported, not new members of the test family.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config as C
from .analyze_run3 import PARTS, discrete_contrast, discrete_outcomes, load_part
from .audit_date_correctness import cue_only_embedded, reconstruct_all
from .audit_instrument_defects import _mentions_co_candidate

CONTENT_CONTRASTS = (("R1", "R2"), ("R3", "R4"))
STRATA = ("rebuilt", "no_co_candidate", "template_matched", "date_cue_clean", "all_three")

# ponytail: whole-word family terms only; "grandfather" and "step-" relations fall outside
# it on purpose, since the confound is the parentage template the control pool carries.
_PARENTAGE = re.compile(r"\b(son|daughter|child|children|father|mother|parent|parents)\b",
                        re.IGNORECASE)


def states_parentage(sentence: str) -> bool:
    return bool(_PARENTAGE.search(sentence or ""))


def _attributes(results: Path) -> dict[tuple[str, str, str], str]:
    out = {}
    for part, (directory, _) in PARTS.items():
        for path in sorted((results / directory).glob("stage2_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("built"):
                        out[(part, rec["model"], rec["item_id"])] = rec["attribute"]
    return out


def stratum_flags(entry: dict, attribute: str, a: str, b: str) -> dict[str, bool]:
    titles = entry["option_titles"]
    sa, sb = entry[f"{a}_sentence"], entry[f"{b}_sentence"]
    no_cc = not (_mentions_co_candidate(sa, titles, entry[f"{a}_edited_idx"])
                 or _mentions_co_candidate(sb, titles, entry[f"{b}_edited_idx"]))
    tmpl = states_parentage(sa) == states_parentage(sb)
    date_ok = True
    if attribute in ("date_of_birth", "date_of_death"):
        date_ok = cue_only_embedded(sa, attribute) is not True
    return {"rebuilt": True, "no_co_candidate": no_cc, "template_matched": tmpl,
            "date_cue_clean": date_ok, "all_three": no_cc and tmpl and date_ok}


def compute(results: Path, data_file: Path) -> dict:
    rebuilt = [e for e in reconstruct_all(results, data_file) if e.get("reconstructed")]
    attr = _attributes(results)
    out: dict = {"n_rebuilt": len(rebuilt), "parts": {}}
    for part in PARTS:
        models = load_part(results, part)
        outcomes = discrete_outcomes(
            {f"{m}::{i}": conds for m, its in models.items() for i, conds in its.items()})
        entries = {f"{e['model']}::{e['item_id']}": e for e in rebuilt if e["part"] == part}
        missing = [k for k in entries if k not in outcomes]
        assert not missing, f"part {part}: {len(missing)} rebuilt items have no stage-3 rows"
        part_out = {"n_rebuilt": len(entries), "states_parentage": {
            cond: sum(states_parentage(e[f"{cond}_sentence"]) for e in entries.values())
            for cond in ("R1", "R2", "R3", "R4")}}
        for a, b in CONTENT_CONTRASTS:
            ref = discrete_contrast(outcomes, a, b)
            full = {"b": ref["b_a_only"], "c": ref["c_b_only"]}
            cells = {"all_built": full}
            for s in STRATA:
                keep = {k: outcomes[k] for k, e in entries.items()
                        if stratum_flags(e, attr[(part, e["model"], e["item_id"])], a, b)[s]}
                r = discrete_contrast(keep, a, b)
                cells[s] = {"n_items": len(keep), "b": r["b_a_only"], "c": r["c_b_only"],
                            "or": r["or"]["or"], "haldane_anscombe": r["or"]["haldane_anscombe"],
                            "p_exact": r["p_exact_two_sided"]}
            part_out[f"{a}-{b}"] = cells
        out["parts"][part] = part_out
    return out


def self_check() -> int:
    assert states_parentage("He is the son of Anna.")
    assert states_parentage("Her parents were farmers.")
    assert not states_parentage("A person of comparison, Johnson sang a song.")
    assert not states_parentage("His grandfather was a judge.")
    print("self-check ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=C.CODE_ROOT / "results")
    ap.add_argument("--data-file", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()
    if args.data_file is None:
        ap.error("--data-file is required to rebuild the inserted sentences")
    res = compute(args.results, args.data_file)
    text = json.dumps(res, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
