"""Model backends: vLLM first, plain transformers as a fallback, a stub for offline tests.

Gated repositories fall back automatically to an ungated mirror of the same weights.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import (
    MAX_MODEL_LEN_CAP,
    MAX_NEW_TOKENS,
    SEED,
    TEMPERATURE,
    SMALL_CARD_GIB,
    TRUST_REMOTE_CODE,
    UNGATED_MIRRORS,
    VLLM_ENFORCE_EAGER,
    VLLM_GPU_MEM_UTILIZATION,
)
from .watchdog import vram_fraction

log = logging.getLogger(__name__)

Chat = Sequence[dict]

# How many top logprobs vLLM is asked for on the one-token forced-choice probe. Large enough
# that a handful of candidate letters (A..D, occasionally more) are reliably inside the
# returned top-k even when the model is not confident about the letter itself -- see
# `LetterProbRead.complete`, which is how a run notices this was not large enough on some item
# rather than silently imputing a missing letter's probability.
LOGPROB_TOPK = 20

# The forced-choice probe's only instruction. Appended as one more user turn onto the same
# rendered context the free-text call already used -- nothing about that context changes.
LETTER_PROBE_INSTRUCTION = "Answer with a single letter and nothing else."


def append_letter_probe(chat: Chat) -> list[dict]:
    """The same rendered context, plus one instruction to answer with a single letter.

    A second, independent call on this is the forced-choice probability read (see the
    experiment design doc's continuous-measure addendum): the free-text call and its
    `choice`/`chosen_is_edited` fields are untouched by this -- this is additional, not a
    replacement.

    The instruction is appended to the *existing* final user turn rather than as a new user
    turn: `chat` always ends in a user message (see `prompts.render`), and a second consecutive
    user turn with no assistant reply between them breaks strict user/assistant alternation.
    Most chat templates silently tolerate it; Mistral-7B-Instruct-v0.3's does not and raises
    `jinja2.exceptions.TemplateError` at `apply_chat_template` -- caught only because it
    crashed instead of quietly rendering something unintended.
    """
    *head, last = chat
    if last["role"] != "user":
        raise ValueError(f"expected chat to end in a user turn, got {last['role']!r}")
    merged = {**last, "content": f"{last['content']}\n\n{LETTER_PROBE_INSTRUCTION}"}
    return [*head, merged]


def render_for_probe(tok, chat: Chat) -> tuple[str, bool]:
    """Render `chat` for a forced single-token read, choosing how by what `chat` ends in.

    A chat ending in a user turn (today's only probe, `append_letter_probe`'s output) renders
    exactly as `letter_probs` always has: `add_generation_prompt=True`, a fresh assistant turn
    opened for the model to fill in. A chat ending in an *assistant* turn -- a prefilled-stem
    probe variant (see `pilot.probe_variants`) -- is rendered as a continuation of that turn
    instead (`continue_final_message=True`), so the forced token is the next token *after* the
    stem, not a new reply to it.

    Returns `(rendered_text, continuation_confirmed)`. `continuation_confirmed` is always True
    for the user-turn case. For the assistant-turn case it is only True when the rendered text
    still ends with the stem exactly as given -- some chat templates accept
    `continue_final_message` without error but still splice in their own turn-boundary tokens
    after it, which would silently defeat the whole point of prefilling. That is reported here,
    not assumed away: a caller records it rather than trusting the kwarg was honoured just
    because it was accepted.
    """
    if not chat or chat[-1]["role"] != "assistant":
        return tok.apply_chat_template(list(chat), tokenize=False, add_generation_prompt=True), True

    stem = chat[-1]["content"]
    try:
        text = tok.apply_chat_template(list(chat), tokenize=False, continue_final_message=True)
    except TypeError as exc:
        raise ContinuationUnsupported(
            f"apply_chat_template does not accept continue_final_message on this "
            f"transformers version: {exc}"
        ) from exc
    return text, text.rstrip().endswith(stem.rstrip())


@dataclass
class LetterProbRead:
    """One forced-choice probability read, restricted to `candidates` and renormalised.

    `raw_logprobs` keeps every candidate's raw logprob (`None` for a candidate that never
    turned up), so the read can be redone offline without a re-run -- raw model output is
    always kept rather than only a derived summary, since re-running the model is expensive
    and the extraction logic is expected to change. `complete` is
    False the moment one candidate's logprob could not be found (vLLM's top-k did not contain
    it); such a read is excluded from the continuous analysis rather than imputed. `probs` is
    the softmax over whatever candidates *were* found -- informational even when incomplete,
    but callers doing the paired analysis must gate on `complete`, not just presence.
    """

    candidates: list[str]
    raw_logprobs: dict[str, float | None]
    probs: dict[str, float]
    complete: bool
    backend: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidates": list(self.candidates),
            "raw_logprobs": dict(self.raw_logprobs),
            "probs": dict(self.probs),
            "complete": self.complete,
            "backend": self.backend,
            "detail": dict(self.detail),
        }


@dataclass
class PromptLogprobs:
    """One prompt's own tokens, and -- for every position but the first -- the log-probability
    the model assigned to the token that is actually there, conditioned only on the tokens
    before it. This is the causal-LM reading a forced-choice generation call cannot give: a
    prompt-level (not next-token) logprob, used here to score how surprising an *inserted*
    sentence is in the context it was inserted into (see `pilot.surprisal`).

    `token_ids` and `tokens` are kept for every backend, including the stub, so a span of
    interest inside this prompt can be located positionally (see `find_inserted_span`) without
    ever re-tokenising by hand. `logprobs[0]` is always `None` -- there is no preceding context
    for the first token, the same convention vLLM's own `prompt_logprobs` output uses.
    `complete` is False the moment any position beyond the first has no logprob (a backend that
    could not or would not supply one there); such a read should be excluded downstream, not
    imputed, the same discipline `LetterProbRead.complete` already established.
    """

    token_ids: list
    tokens: list[str]
    logprobs: list[float | None]
    backend: str
    complete: bool
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "token_ids": list(self.token_ids),
            "tokens": list(self.tokens),
            "logprobs": list(self.logprobs),
            "backend": self.backend,
            "complete": self.complete,
            "detail": dict(self.detail),
        }


class PromptLogprobsUnsupported(RuntimeError):
    """Raised when a backend cannot supply prompt-level logprobs.

    Whether `vllm==0.6.3.post1`'s `SamplingParams` accepts `prompt_logprobs` at all is not
    verified anywhere in this repository -- there is no GPU available to check it against a
    real engine. Rather than assume support, `VLLMBackend.prompt_token_logprobs` probes for it
    at call time and raises this, with the underlying error attached, the first time it turns
    out not to work -- a caller (see `pilot.surprisal`) can catch this specifically and fall
    back to the transformers path (`HFBackend`) instead of crashing the whole run.
    """


class ContinuationUnsupported(RuntimeError):
    """Raised when a chat ending in an assistant turn cannot be rendered as a continuation.

    `transformers==4.46.3` (this project's pin) accepts `continue_final_message` on
    `apply_chat_template`, but *accepting the keyword* and *a given model's chat template
    actually treating it as "keep going from here" rather than restarting with a fresh
    assistant header* are two different things -- the second is exactly what
    `pilot.probe_variants` exists to measure, not assume, for a reasoning model whose template
    may or may not suppress the opening `<think>` block on a prefilled turn. This exception
    covers only the first, structural failure (an old `transformers` without the keyword at
    all); the second is reported as data (`PromptLogprobs`/`LetterProbRead` `detail`), never as
    a crash.
    """


def renormalize_letter_logprobs(raw_logprobs: dict[str, float | None]) -> dict[str, float]:
    """Softmax over whichever candidates have a logprob (`None` entries are skipped, never
    imputed). Returns {} if nothing was found at all."""
    present = {letter: lp for letter, lp in raw_logprobs.items() if lp is not None}
    if not present:
        return {}
    m = max(present.values())
    exps = {letter: math.exp(lp - m) for letter, lp in present.items()}
    total = sum(exps.values())
    return {letter: v / total for letter, v in exps.items()}


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _letter_match(token_text: str, letter: str) -> bool:
    return token_text.strip() == letter


def letter_read_from_token_logprobs(
    token_logprobs: dict[str, float],
    candidates: Sequence[str],
    *,
    backend: str,
    detail: dict | None = None,
) -> LetterProbRead:
    """Build a `LetterProbRead` from {decoded token text: logprob} for one generation step --
    e.g. vLLM's top-k at the single generated position.

    A letter can appear as more than one distinct token (with or without a leading space);
    every matching entry's probability mass is combined by log-sum-exp rather than picking one
    arbitrarily, because both are genuine ways the model could have emitted that letter. A
    candidate with no matching entry at all gets `None`: the top-k did not contain it.
    """
    raw: dict[str, float | None] = {}
    for letter in candidates:
        matches = [lp for text, lp in token_logprobs.items() if _letter_match(text, letter)]
        raw[letter] = _logsumexp(matches) if matches else None
    complete = all(v is not None for v in raw.values())
    probs = renormalize_letter_logprobs(raw)
    return LetterProbRead(
        candidates=list(candidates), raw_logprobs=raw, probs=probs, complete=complete,
        backend=backend, detail=detail or {},
    )


def _new_tokens_after(base_ids: Sequence[int], combo_ids: Sequence[int]) -> list[int]:
    """The tokens in `combo_ids` beyond the longest common prefix with `base_ids`.

    Appending a bare letter to the rendered prompt and re-tokenising the whole string is the
    only reliable way to learn which token the tokenizer would actually use for it in this
    exact context -- a plain lookup of "A" cannot tell whether the model's chat template ends
    on whitespace that BPE would merge into a leading-space variant of the letter's token.
    Comparing by common prefix (rather than assuming `combo_ids` simply extends `base_ids`)
    covers the case where that merge changes the *last* token of the base sequence too.
    """
    common = 0
    for a, b in zip(base_ids, combo_ids):
        if a != b:
            break
        common += 1
    return list(combo_ids[common:])


def find_inserted_span(base_ids: Sequence, edited_ids: Sequence) -> tuple[int, int]:
    """The `[start, end)` slice of `edited_ids` that `base_ids` does not have -- one contiguous
    insertion, located by the longest common prefix and the longest common suffix, never by
    searching for the inserted text itself.

    `_new_tokens_after` above handles the special case of an insertion at the very end (a bare
    letter appended to a prompt that otherwise stays identical): common prefix, and everything
    after it is new. An inserted sentence appended to one option's profile is not at the end of
    the *rendered prompt* unless that option happens to be presented last -- the remaining
    options and the task instructions that follow it are unchanged and must be recognised as
    such, which a prefix-only comparison cannot do (it would report everything from the
    insertion point to the end of the prompt as "new", when almost all of that is the
    untouched tail). Comparing from both ends at once handles that: the common suffix is
    computed after the common prefix and is capped so the two regions can never overlap, which
    is what keeps this correct even when the inserted sentence's own words recur earlier or
    later in the same profile -- the position of the divergence is what is being found here,
    never its content, so a recurring word elsewhere cannot be mistaken for part of the
    insertion.

    Returns `(len(base_ids), len(base_ids))` (an empty span at the point of first difference,
    which is the prompt's own length when the two sequences are identical) when `edited_ids`
    inserts nothing relative to `base_ids`.
    """
    n_base, n_edited = len(base_ids), len(edited_ids)
    limit = min(n_base, n_edited)

    prefix = 0
    while prefix < limit and base_ids[prefix] == edited_ids[prefix]:
        prefix += 1

    max_suffix = limit - prefix  # never let the suffix region eat into the prefix region
    suffix = 0
    while suffix < max_suffix and base_ids[n_base - 1 - suffix] == edited_ids[n_edited - 1 - suffix]:
        suffix += 1

    return prefix, n_edited - suffix


def _token_has_leading_space(piece: str) -> bool:
    return piece.startswith(("▁", "Ġ", " "))  # sentencepiece, GPT2-BPE, or literal


def resolve_letter_token_ids(
    tokenizer, prompt_text: str, letters: Sequence[str]
) -> tuple[dict[str, int], dict[str, str]]:
    """Which token id the tokenizer actually produces for each candidate letter, in this exact
    rendered prompt -- resolving the leading-space ambiguity ("A" and " A" are different token
    ids, and which one a given chat template invites is template-specific) empirically rather
    than by guessing. Returns (letter -> token id, letter -> "bare" | "leading_space")."""
    base_ids = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    token_ids: dict[str, int] = {}
    variant: dict[str, str] = {}
    for letter in letters:
        combo_ids = list(tokenizer(prompt_text + letter, add_special_tokens=False).input_ids)
        new_ids = _new_tokens_after(base_ids, combo_ids)
        if not new_ids:
            new_ids = list(tokenizer(letter, add_special_tokens=False).input_ids)
        tid = new_ids[0]
        token_ids[letter] = tid
        piece = tokenizer.convert_ids_to_tokens([tid])[0]
        variant[letter] = "leading_space" if _token_has_leading_space(piece) else "bare"
    return token_ids, variant


def device_capability() -> tuple[int, int] | None:
    """CUDA compute capability, or None when there is no device to ask."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def device_vram_gib() -> float | None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def preferred_dtype(capability: tuple[int, int] | None = None) -> str:
    """bfloat16 above compute capability 8.0, float16 below it.

    This is not a tuning choice. bfloat16 has no hardware support below 8.0 and vLLM refuses
    to load at all on such a card: a T4 is 7.5. Getting it wrong is an immediate hard failure,
    which is the good case -- the bad case would be silent emulation at unusable speed.
    """
    cap = capability if capability is not None else device_capability()
    if cap is None:
        return "float16"  # no device to ask: the conservative choice
    return "bfloat16" if cap[0] >= 8 else "float16"


def resolve_profile(name: str = "auto") -> str:
    """Which model roster fits this card."""
    if name != "auto":
        return name
    vram = device_vram_gib()
    if vram is None:
        return "3090ti"
    profile = "3090ti" if vram >= SMALL_CARD_GIB else "t4"
    log.info("detected %.1f GiB of VRAM, capability %s -> the %s roster",
             vram, device_capability(), profile)
    return profile


def _enforce_eager() -> bool:
    if VLLM_ENFORCE_EAGER in ("1", "true", "True"):
        return True
    if VLLM_ENFORCE_EAGER in ("0", "false", "False"):
        return False
    vram = device_vram_gib()  # "auto": eager on a small card, graphs on a large one
    return vram is not None and vram < SMALL_CARD_GIB


class ModelUnavailable(RuntimeError):
    """Raised when weights cannot be obtained -- gated repo, no token, network failure."""


class Backend:
    name = "backend"
    kind = "abstract"

    def generate(
        self, chats: Sequence[Chat], *, temperature: float | None = None, seed: int | None = None,
    ) -> list[str]:
        """Free-text generation, one reply per chat.

        `temperature`/`seed` default to `None`, meaning "this backend's own greedy
        configuration" (`config.TEMPERATURE`, `config.SEED`) -- every existing caller, and
        every committed result, was produced by that exact path, so passing neither must remain
        byte-identical to what this method did before these two parameters existed. Passing
        `temperature=0.0` explicitly still takes the greedy (argmax) path, never a sampling path
        merely parameterised at zero -- those are not the same operation on any backend below,
        and the difference matters: a sampler at temperature 0 can still branch on its RNG state
        in ways argmax decoding structurally cannot.
        """
        raise NotImplementedError

    def letter_probs(
        self, chats: Sequence[Chat], candidates: Sequence[Sequence[str]]
    ) -> list[LetterProbRead]:
        """The forced-choice probability read: one call per chat, `max_tokens=1`, restricted to
        that item's candidate letters. Additional to `generate`, never a replacement for it."""
        raise NotImplementedError

    def prompt_token_logprobs(self, chats: Sequence[Chat]) -> list[PromptLogprobs]:
        """Every prompt token's own logprob, conditioned on the tokens before it -- one call per
        chat, no generation. Used by `pilot.surprisal` to score how surprising an inserted
        sentence is in the context it was inserted into; not needed by, and never a replacement
        for, `generate` or `letter_probs`."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _accepts_kwargs(fn: Callable, names: Sequence[str]) -> bool:
    """Whether `fn` declares every name in `names` as a parameter, or takes `**kwargs` (which
    accepts anything). Used once, at `StubBackend` construction, to decide whether a
    caller-supplied responder wants to see `temperature`/`seed` without breaking on a responder
    that predates those parameters and takes only `(index, chat)`."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    declared = {p.name for p in params}
    return all(name in declared for name in names)


def _default_stub_letter_logprobs(
    index: int, chat: Chat, candidates: Sequence[str]
) -> dict[str, float]:
    """Canned raw logprobs used when a `StubBackend` is built without its own
    `letter_prob_responder` -- monotonically decreasing by candidate order, so the
    distribution is complete and believable, and depends on nothing but the candidate list
    itself (never wall-clock, randomness, or dict order), so two stub runs on the same input
    are byte-identical."""
    return {letter: -0.5 * rank for rank, letter in enumerate(candidates)}


_STUB_WORD = re.compile(r"\S+")


def _stub_render_text(chat: Chat) -> str:
    """A deterministic stand-in for `apply_chat_template` -- the stub has no tokenizer, so
    tokenising the actual rendered text is not available to it. Every turn's content, in order,
    is enough for `find_inserted_span` to locate an inserted span positionally, which is all
    the stub's own tests need it for."""
    return "\n".join(str(turn.get("content", "")) for turn in chat)


def _default_stub_prompt_logprobs(index: int, chat: Chat) -> tuple[list[str], list[float | None]]:
    """Canned prompt-token logprobs used when a `StubBackend` is built without its own
    `prompt_logprob_responder`: whitespace-split "tokens" from the rendered chat text (a word
    each -- crude compared to real subword tokenisation, but sufficient to exercise the span
    finder and the surprisal arithmetic with no GPU), and a logprob of `None` for the first
    token, then a constant -0.4 nats for every token after it -- deterministic, depends on
    nothing but `chat` itself."""
    tokens = _STUB_WORD.findall(_stub_render_text(chat))
    logprobs: list[float | None] = [None] * min(1, len(tokens)) + [-0.4] * max(0, len(tokens) - 1)
    return tokens, logprobs


class StubBackend(Backend):
    """Returns canned text. Lets the whole pipeline run in a test, with no GPU and no network."""

    kind = "stub"

    def __init__(
        self,
        responder: Callable[[int, Chat], str],
        name: str = "stub",
        letter_prob_responder: Callable[[int, Chat, Sequence[str]], dict[str, float]] | None = None,
        prompt_logprob_responder: Callable[[int, Chat], tuple[list, list[float | None]]] | None = None,
    ):
        self.name = name
        self._responder = responder
        self._letter_prob_responder = letter_prob_responder or _default_stub_letter_logprobs
        self._prompt_logprob_responder = prompt_logprob_responder or _default_stub_prompt_logprobs
        # Whether `responder` itself wants to see (temperature, seed) -- a sweep's own custom
        # responder can declare `temperature`/`seed` keyword parameters (or **kwargs) to make
        # its canned replies genuinely depend on them; every existing responder in this project
        # (e.g. `run_pilot._StubReplies`, taking only `(index, chat)`) does not, and must keep
        # working completely unexamined -- inspected once here, not on every call.
        self._responder_wants_sampling_kwargs = _accepts_kwargs(responder, ("temperature", "seed"))

    def generate(
        self, chats: Sequence[Chat], *, temperature: float | None = None, seed: int | None = None,
    ) -> list[str]:
        if temperature is None and seed is None:
            return [self._responder(i, chat) for i, chat in enumerate(chats)]
        out: list[str] = []
        for i, chat in enumerate(chats):
            if self._responder_wants_sampling_kwargs:
                out.append(self._responder(i, chat, temperature=temperature, seed=seed))
                continue
            # No sampling-aware responder was supplied: still make the reply visibly and
            # reproducibly depend on (temperature, seed) rather than silently ignore them, so a
            # `--dry-run` sweep against the default stub responder is a meaningful smoke test of
            # the plumbing (same call reproduces, a different seed does not) instead of a no-op.
            # Deterministic in (temperature, seed) alone -- never wall-clock, never call order.
            eff_temperature = TEMPERATURE if temperature is None else temperature
            eff_seed = SEED if seed is None else seed
            base = self._responder(i, chat)
            out.append(f"{base} [t={eff_temperature:g} seed={eff_seed}]")
        return out

    def letter_probs(
        self, chats: Sequence[Chat], candidates: Sequence[Sequence[str]]
    ) -> list[LetterProbRead]:
        reads = []
        for i, (chat, cands) in enumerate(zip(chats, candidates)):
            raw_by_letter = self._letter_prob_responder(i, chat, cands)
            raw = {letter: raw_by_letter.get(letter) for letter in cands}
            complete = all(v is not None for v in raw.values())
            probs = renormalize_letter_logprobs(raw)
            reads.append(LetterProbRead(
                candidates=list(cands), raw_logprobs=raw, probs=probs, complete=complete,
                backend="stub", detail={},
            ))
        return reads

    def prompt_token_logprobs(self, chats: Sequence[Chat]) -> list[PromptLogprobs]:
        reads = []
        for i, chat in enumerate(chats):
            token_ids, logprobs = self._prompt_logprob_responder(i, chat)
            tokens = [str(t) for t in token_ids]
            complete = all(lp is not None for lp in logprobs[1:])
            reads.append(PromptLogprobs(
                token_ids=list(token_ids), tokens=tokens, logprobs=list(logprobs),
                backend="stub", complete=complete, detail={},
            ))
        return reads


class VLLMBackend(Backend):
    kind = "vllm"

    def __init__(self, name: str, max_model_len: int = MAX_MODEL_LEN_CAP):
        self.name = name
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except Exception as exc:  # noqa: BLE001 - any import failure means "not this backend"
            raise ModelUnavailable(f"vllm/transformers unavailable: {exc}") from exc

        try:
            self._tok = AutoTokenizer.from_pretrained(
                name, trust_remote_code=TRUST_REMOTE_CODE
            )
            self._llm = LLM(
                model=name,
                dtype=preferred_dtype(),
                max_model_len=max_model_len,
                gpu_memory_utilization=VLLM_GPU_MEM_UTILIZATION,
                enforce_eager=_enforce_eager(),
                trust_remote_code=TRUST_REMOTE_CODE,
                seed=SEED,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(f"could not load {name} under vllm: {exc}") from exc

        self._params = SamplingParams(
            temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS, seed=SEED
        )
        frac = vram_fraction()
        log.info(
            "%s loaded: %s, max_model_len=%d, mem target %.2f, eager=%s, VRAM now %s",
            name, preferred_dtype(), max_model_len, VLLM_GPU_MEM_UTILIZATION,
            _enforce_eager(), f"{frac:.1%}" if frac is not None else "unknown",
        )

    def generate(
        self, chats: Sequence[Chat], *, temperature: float | None = None, seed: int | None = None,
    ) -> list[str]:
        texts = [
            self._tok.apply_chat_template(list(c), tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        if temperature is None and seed is None:
            params = self._params  # the exact pre-existing object: byte-identical to before
        else:
            from vllm import SamplingParams

            eff_temperature = TEMPERATURE if temperature is None else temperature
            eff_seed = SEED if seed is None else seed
            # vLLM's `SamplingParams.seed` seeds that one *request's* own sampler -- with the
            # same prompt, the same `seed` reproduces the same sample every time (vLLM's
            # documented per-request determinism), but it neither reads nor writes any
            # process-global RNG state and says nothing about any other request's seed. This is
            # a different seed from the one passed to `LLM(seed=SEED)` at construction above,
            # which governs weight-loading and CUDA-graph capture order, not sampling. At
            # `eff_temperature == 0.0` vLLM's sampler takes the greedy (argmax) path internally
            # regardless of `seed` -- matching this backend's existing zero-temperature
            # behaviour rather than a temperature-parameterised sampling path that merely
            # happens to have its randomness suppressed.
            params = SamplingParams(
                temperature=eff_temperature, max_tokens=MAX_NEW_TOKENS, seed=eff_seed
            )
        outs = self._llm.generate(texts, params)
        return [o.outputs[0].text.strip() for o in outs]

    def token_len(self, text: str) -> int:
        return len(self._tok(text).input_ids)

    def letter_probs(
        self, chats: Sequence[Chat], candidates: Sequence[Sequence[str]]
    ) -> list[LetterProbRead]:
        from vllm import SamplingParams

        rendered = [render_for_probe(self._tok, list(c)) for c in chats]
        texts = [t for t, _ok in rendered]
        confirmed = [ok for _t, ok in rendered]
        params = SamplingParams(
            max_tokens=1, logprobs=LOGPROB_TOPK, temperature=TEMPERATURE, seed=SEED
        )
        outs = self._llm.generate(texts, params)
        reads = []
        for out, cands, ok in zip(outs, candidates, confirmed):
            step_logprobs = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            token_texts = {
                lp.decoded_token: lp.logprob for lp in step_logprobs.values()
                if lp.decoded_token is not None
            }
            reads.append(letter_read_from_token_logprobs(
                token_texts, cands, backend="vllm",
                detail={"topk": LOGPROB_TOPK, "continuation_confirmed": ok},
            ))
        return reads

    def _probe_prompt_logprobs_support(self) -> None:
        """Try building a `SamplingParams(prompt_logprobs=...)` once, cache the outcome.

        Whether `vllm==0.6.3.post1` accepts this kwarg at all is not verified anywhere in this
        repository -- there is no GPU here to check it against a real engine build. This probes
        rather than assumes: a construction failure is captured once and raised as a clear,
        catchable `PromptLogprobsUnsupported` on every call, instead of an opaque `TypeError`
        surfacing from deep inside a batched `generate()` call.
        """
        if getattr(self, "_prompt_logprobs_supported", None) is not None:
            return
        from vllm import SamplingParams

        try:
            SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=TEMPERATURE, seed=SEED)
        except (TypeError, ValueError) as exc:
            self._prompt_logprobs_supported = False
            self._prompt_logprobs_error = str(exc)
        else:
            self._prompt_logprobs_supported = True
            self._prompt_logprobs_error = None

    def prompt_token_logprobs(self, chats: Sequence[Chat]) -> list[PromptLogprobs]:
        """Prompt-level logprobs via `SamplingParams(prompt_logprobs=0)`: no extra top-k over
        alternatives, just the logprob of the prompt token that is actually there at every
        position (`None` at position 0, matching vLLM's own convention). Raises
        `PromptLogprobsUnsupported` -- caught, not crashed on -- the first time either the
        keyword itself is rejected or an engine that accepted it still returns nothing.
        """
        from vllm import SamplingParams

        self._probe_prompt_logprobs_support()
        if not self._prompt_logprobs_supported:
            raise PromptLogprobsUnsupported(
                f"vllm {self.name}: SamplingParams(prompt_logprobs=...) is not accepted by "
                f"this vllm build: {self._prompt_logprobs_error}"
            )

        texts = [
            self._tok.apply_chat_template(list(c), tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        params = SamplingParams(
            max_tokens=1, prompt_logprobs=0, temperature=TEMPERATURE, seed=SEED
        )
        outs = self._llm.generate(texts, params)

        reads = []
        for out in outs:
            prompt_lps = out.prompt_logprobs
            if prompt_lps is None:
                raise PromptLogprobsUnsupported(
                    f"vllm {self.name}: SamplingParams accepted prompt_logprobs but the engine "
                    "returned none -- this build does not actually populate them"
                )
            token_ids = list(out.prompt_token_ids)
            tokens: list[str] = []
            logprobs: list[float | None] = []
            for pos, tid in enumerate(token_ids):
                entry_map = prompt_lps[pos]
                if entry_map is None:  # position 0: no preceding context, by convention
                    tokens.append(self._tok.convert_ids_to_tokens([tid])[0])
                    logprobs.append(None)
                    continue
                entry = entry_map.get(tid)
                if entry is None:
                    tokens.append(self._tok.convert_ids_to_tokens([tid])[0])
                    logprobs.append(None)
                else:
                    tokens.append(
                        entry.decoded_token if entry.decoded_token is not None
                        else self._tok.convert_ids_to_tokens([tid])[0]
                    )
                    logprobs.append(entry.logprob)
            complete = all(lp is not None for lp in logprobs[1:])
            reads.append(PromptLogprobs(
                token_ids=token_ids, tokens=tokens, logprobs=logprobs,
                backend="vllm", complete=complete, detail={"prompt_logprobs_topk": 0},
            ))
        return reads

    def close(self) -> None:
        # vLLM does not release its GPU allocation just because the LLM object goes out of
        # scope: the model runner keeps CUDA-graph pools and the distributed process group
        # alive. Without this, the next model's init OOMs against memory the previous model
        # never gave back -- measured, see the experiment ledger.
        import gc

        import torch

        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001 - best-effort cleanup, never block a shutdown on it
            log.warning("vllm distributed cleanup failed for %s", self.name, exc_info=True)
        del self._llm
        del self._tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFBackend(Backend):
    kind = "transformers"

    def __init__(self, name: str, max_model_len: int = MAX_MODEL_LEN_CAP):
        self.name = name
        self.max_model_len = max_model_len
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(f"transformers unavailable: {exc}") from exc

        self._torch = torch
        try:
            self._tok = AutoTokenizer.from_pretrained(
                name, trust_remote_code=TRUST_REMOTE_CODE
            )
            import transformers

            # `torch_dtype` up to transformers 4.x, renamed `dtype` in 5.x. Passing the wrong
            # one is not a clean error: an unknown kwarg can be swallowed into the config and
            # the model silently loads in float32, which will not fit.
            major = int(transformers.__version__.split(".")[0])
            dtype_key = "dtype" if major >= 5 else "torch_dtype"
            torch_dtype = getattr(torch, preferred_dtype())
            self._model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="auto",
                trust_remote_code=TRUST_REMOTE_CODE,
                **{dtype_key: torch_dtype},
            )
            log.info("%s loaded under transformers in %s", name, preferred_dtype())
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(f"could not load {name} under transformers: {exc}") from exc
        self._model.eval()

    def generate(
        self, chats: Sequence[Chat], *, temperature: float | None = None, seed: int | None = None,
    ) -> list[str]:
        eff_temperature = TEMPERATURE if temperature is None else temperature
        out: list[str] = []
        for chat in chats:
            ids = self._tok.apply_chat_template(
                list(chat), return_tensors="pt", add_generation_prompt=True
            ).to(self._model.device)
            gen_kwargs: dict = dict(
                max_new_tokens=MAX_NEW_TOKENS, pad_token_id=self._tok.eos_token_id,
            )
            if eff_temperature > 0.0:
                # Sampling: `temperature` is only meaningful together with `do_sample=True` --
                # passing it under greedy decoding is silently ignored by `generate()`, which
                # would make a bug here invisible rather than loud, so the two are always set
                # together. Seeded via a per-call `torch.Generator` (never the process-global
                # RNG), so two calls with the same `seed` reproduce and a different one does not,
                # without perturbing any other model's or any other call's random state.
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = eff_temperature
                eff_seed = SEED if seed is None else seed
                gen_kwargs["generator"] = self._torch.Generator(
                    device=ids.device
                ).manual_seed(eff_seed)
            else:
                # Greedy (argmax): `temperature==0.0` is not "sampling with the randomness
                # turned down to nothing" -- it is a structurally different decoding path, and
                # `do_sample=False` is how transformers spells it. No `temperature` kwarg is
                # passed here, exactly as before this method took one.
                gen_kwargs["do_sample"] = False
            with self._torch.no_grad():
                gen = self._model.generate(ids, **gen_kwargs)
            out.append(self._tok.decode(gen[0, ids.shape[-1]:], skip_special_tokens=True).strip())
        return out

    def letter_probs(
        self, chats: Sequence[Chat], candidates: Sequence[Sequence[str]]
    ) -> list[LetterProbRead]:
        """Logits at the final position, indexed directly at each candidate letter's resolved
        token id -- see `resolve_letter_token_ids` for how "A" vs " A" is decided per prompt.
        No sampling, no top-k: the full vocabulary softmax is available, so every candidate
        always has a value and `complete` is always True here (unlike the vLLM top-k path)."""
        reads = []
        for chat, cands in zip(chats, candidates):
            prompt_text, confirmed = render_for_probe(self._tok, list(chat))
            token_ids, variant = resolve_letter_token_ids(self._tok, prompt_text, cands)
            ids = self._tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            ids = ids.input_ids.to(self._model.device)
            with self._torch.no_grad():
                logits = self._model(ids).logits[0, -1, :]
            log_probs = self._torch.log_softmax(logits.float(), dim=-1)
            raw: dict[str, float | None] = {}
            for letter in cands:
                tid = token_ids.get(letter)
                raw[letter] = float(log_probs[tid].item()) if tid is not None else None
            complete = all(v is not None for v in raw.values())
            probs = renormalize_letter_logprobs(raw)
            reads.append(LetterProbRead(
                candidates=list(cands), raw_logprobs=raw, probs=probs, complete=complete,
                backend="transformers",
                detail={"token_variant": variant, "continuation_confirmed": confirmed},
            ))
        return reads

    def prompt_token_logprobs(self, chats: Sequence[Chat]) -> list[PromptLogprobs]:
        """Teacher-forced forward pass over each prompt: no sampling, no top-k. Position `i`'s
        logprob comes from the distribution the model assigns *after* seeing tokens `0..i-1`
        (`logits[i-1]`), the standard next-token-prediction reading, applied here to the
        prompt's own tokens instead of to a generated continuation -- modelled directly on the
        manual logit read `letter_probs` above already does for a single position.
        """
        reads = []
        for chat in chats:
            prompt_text = self._tok.apply_chat_template(
                list(chat), tokenize=False, add_generation_prompt=True
            )
            ids = self._tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            ids = ids.input_ids.to(self._model.device)
            with self._torch.no_grad():
                logits = self._model(ids).logits[0]  # (seq_len, vocab)
            log_probs = self._torch.log_softmax(logits.float(), dim=-1)

            token_ids = ids[0].tolist()
            tokens = self._tok.convert_ids_to_tokens(token_ids)
            logprobs: list[float | None] = [None] if token_ids else []
            for pos in range(1, len(token_ids)):
                logprobs.append(float(log_probs[pos - 1, token_ids[pos]].item()))

            reads.append(PromptLogprobs(
                token_ids=token_ids, tokens=tokens, logprobs=logprobs,
                backend="transformers", complete=True, detail={},
            ))

            # `logits`/`log_probs` are full-vocab float32 tensors, one pair per call. Left to
            # the allocator's own pace across hundreds of variable-length prompts they
            # fragment VRAM upward until the watchdog's 95% ceiling aborts the run (measured on
            # Part C's longer, 6-option prompts). Freeing them each item keeps peak usage flat.
            del ids, logits, log_probs
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        return reads

    def close(self) -> None:
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def get_backend(
    name: str, kind: str = "auto", max_model_len: int = MAX_MODEL_LEN_CAP,
    allow_mirror: bool = True,
) -> Backend:
    """Build a backend for `name`, falling back by backend and then by repository.

    Order: vllm on the named repo, transformers on the named repo, then the same two on an
    ungated mirror of the same weights if one is known. The mirror is recorded in the results,
    so a run never quietly claims to be the gated checkpoint it could not fetch.
    """
    candidates = [name]
    if allow_mirror and name in UNGATED_MIRRORS:
        candidates.append(UNGATED_MIRRORS[name])

    last: Exception | None = None
    for repo in candidates:
        if repo != name:
            log.warning("falling back to the ungated mirror %s for %s", repo, name)
        for backend_cls, backend_kind in ((VLLMBackend, "vllm"), (HFBackend, "transformers")):
            if kind != "auto" and kind != backend_kind:
                continue
            try:
                return backend_cls(repo, max_model_len=max_model_len)
            except ModelUnavailable as exc:
                log.warning("%s under %s: %s", repo, backend_kind, exc)
                last = exc

    raise ModelUnavailable(f"no backend could load {name}: {last}")
