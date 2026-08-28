"""The two prompts for the study's two arms, and the renderer that builds a chat from an item.

``spontaneous`` never asks the model to rule anything out. ``elicited`` asks directly which
candidate was ruled out and what is missing from its profile.
"""

from __future__ import annotations

from .data import Item

SYSTEM = (
    "You answer questions by choosing one of several candidate profiles. "
    "Use only the profiles given to you."
)

_SPONTANEOUS_TASK = (
    "Give the letter and name of the correct candidate, then explain your choice "
    "in two to four sentences."
)

_ELICITED_TASK = (
    "Give the letter and name of the correct candidate, then explain your choice "
    "in two to four sentences. Say which one of the other candidates you ruled out "
    "and what specifically is missing from that candidate's profile."
)

ARMS = {"spontaneous": _SPONTANEOUS_TASK, "elicited": _ELICITED_TASK}


def render(item: Item, arm: str) -> list[dict[str, str]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")

    blocks = []
    for i, opt in enumerate(item.options):
        letter = chr(ord("A") + i)
        blocks.append(f"{letter}) {opt.title}\n{opt.profile}")

    user = (
        f"Question: {item.question}\n\n"
        "Candidates:\n\n" + "\n\n".join(blocks) + "\n\n" + ARMS[arm]
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
