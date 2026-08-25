"""Model backends: vLLM first, plain transformers as a fallback, a stub for offline tests.

Gated repositories fall back automatically to an ungated mirror of the same weights.
"""

from __future__ import annotations

import logging
import math
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


@dataclass
class LetterProbRead:
    """One forced-choice probability read, restricted to `candidates` and renormalised.

    `raw_logprobs` keeps every candidate's raw logprob (`None` for a candidate that never
    turned up), so the read can be redone offline without a re-run -- per CLAUDE.md's
    "raw model output is always kept" convention, extended to this measure. `complete` is
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

    def generate(self, chats: Sequence[Chat]) -> list[str]:
        raise NotImplementedError

    def letter_probs(
        self, chats: Sequence[Chat], candidates: Sequence[Sequence[str]]
    ) -> list[LetterProbRead]:
        """The forced-choice probability read: one call per chat, `max_tokens=1`, restricted to
        that item's candidate letters. Additional to `generate`, never a replacement for it."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _default_stub_letter_logprobs(
    index: int, chat: Chat, candidates: Sequence[str]
) -> dict[str, float]:
    """Canned raw logprobs used when a `StubBackend` is built without its own
    `letter_prob_responder` -- monotonically decreasing by candidate order, so the
    distribution is complete and believable, and depends on nothing but the candidate list
    itself (never wall-clock, randomness, or dict order), so two stub runs on the same input
    are byte-identical."""
    return {letter: -0.5 * rank for rank, letter in enumerate(candidates)}


class StubBackend(Backend):
    """Returns canned text. Lets the whole pipeline run in a test, with no GPU and no network."""

    kind = "stub"

    def __init__(
        self,
        responder: Callable[[int, Chat], str],
        name: str = "stub",
        letter_prob_responder: Callable[[int, Chat, Sequence[str]], dict[str, float]] | None = None,
    ):
        self.name = name
        self._responder = responder
        self._letter_prob_responder = letter_prob_responder or _default_stub_letter_logprobs

    def generate(self, chats: Sequence[Chat]) -> list[str]:
        return [self._responder(i, chat) for i, chat in enumerate(chats)]

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

    def generate(self, chats: Sequence[Chat]) -> list[str]:
        texts = [
            self._tok.apply_chat_template(list(c), tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        outs = self._llm.generate(texts, self._params)
        return [o.outputs[0].text.strip() for o in outs]

    def token_len(self, text: str) -> int:
        return len(self._tok(text).input_ids)

    def letter_probs(
        self, chats: Sequence[Chat], candidates: Sequence[Sequence[str]]
    ) -> list[LetterProbRead]:
        from vllm import SamplingParams

        texts = [
            self._tok.apply_chat_template(list(c), tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        params = SamplingParams(
            max_tokens=1, logprobs=LOGPROB_TOPK, temperature=TEMPERATURE, seed=SEED
        )
        outs = self._llm.generate(texts, params)
        reads = []
        for out, cands in zip(outs, candidates):
            step_logprobs = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            token_texts = {
                lp.decoded_token: lp.logprob for lp in step_logprobs.values()
                if lp.decoded_token is not None
            }
            reads.append(letter_read_from_token_logprobs(
                token_texts, cands, backend="vllm", detail={"topk": LOGPROB_TOPK},
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

    def generate(self, chats: Sequence[Chat]) -> list[str]:
        out: list[str] = []
        for chat in chats:
            ids = self._tok.apply_chat_template(
                list(chat), return_tensors="pt", add_generation_prompt=True
            ).to(self._model.device)
            with self._torch.no_grad():
                gen = self._model.generate(
                    ids,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=self._tok.eos_token_id,
                )
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
            prompt_text = self._tok.apply_chat_template(
                list(chat), tokenize=False, add_generation_prompt=True
            )
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
                backend="transformers", detail={"token_variant": variant},
            ))
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
