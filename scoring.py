"""
Multiple-choice log-likelihood scoring.

The starter notebook generates with `max_new_tokens=20` and parses the output
free-form, but that's brittle: the model might output "A. The answer is...",
"(A)", " A", or just hallucinate a letter not in the choice set. Worse,
free-form generation gives no graceful behavior when num_choices != 4.

The right approach for multiple-choice eval is log-likelihood scoring:
  for each choice index j in {0..num_choices-1}:
      letter = "ABCDEFGHIJ"[j]
      score[j] = log P(letter | image, prompt)
  predict argmax(score)

For a *single-token* answer (which is the case here — A, B, C, D are all
single tokens in SmolLM2's tokenizer), this is one forward pass per example
that returns the next-token logits. Then we just gather logits at the
letter positions.

This is much cheaper than running generate(), and is deterministic.
"""
from __future__ import annotations
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from config import CFG
from data import build_prompt_from_fields


@torch.inference_mode()
def _score_batch_letter_logits(
    model,
    processor,
    images: Sequence,
    prompts: Sequence[str],
    num_choices_per: Sequence[int],
) -> torch.Tensor:
    """Compute per-letter logits at the answer position.

    Returns a [B, num_letters] tensor on the model's device, with positions
    >= num_choices_per[i] masked to -inf for each row i.
    """
    assert len(images) == len(prompts) == len(num_choices_per)

    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    # Pre-compute the token id for each candidate answer letter, exactly as
    # the model would see it after "Answer:" → " A". The leading space
    # matters for BPE tokenizers — " A" and "A" are different tokens.
    letter_token_ids = []
    for letter in CFG.choice_letters:
        ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        # Sanity: each letter should encode to a single token. If a tokenizer
        # surprises us, fall back to the last token (the actual letter token).
        letter_token_ids.append(ids[-1])
    letter_token_ids = torch.tensor(letter_token_ids, device=device)

    inputs = processor(
        text=list(prompts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        # No truncation — see TrainCollator for rationale.
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    outputs = model(**inputs)
    # outputs.logits shape: [B, T, V]. We want the logits at the position of
    # the LAST non-pad token of each row, since that's the position whose
    # next-token prediction is the answer letter.
    logits = outputs.logits
    attn = inputs["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    batch_idx = torch.arange(logits.size(0), device=device)
    last_logits = logits[batch_idx, last_idx, :]

    letter_logits = last_logits[:, letter_token_ids]  # [B, num_letters]

    max_choices = letter_logits.size(1)
    n = torch.tensor(num_choices_per, device=device)
    arange = torch.arange(max_choices, device=device).unsqueeze(0)
    mask = arange >= n.unsqueeze(1)
    letter_logits = letter_logits.masked_fill(mask, float("-inf"))

    return letter_logits


@torch.inference_mode()
def score_batch(
    model,
    processor,
    images: Sequence,
    prompts: Sequence[str],
    num_choices_per: Sequence[int],
) -> list[int]:
    """Returns predicted answer index (0-indexed) for each example in the batch."""
    letter_logits = _score_batch_letter_logits(
        model, processor, images, prompts, num_choices_per
    )
    return letter_logits.argmax(dim=1).cpu().tolist()


@torch.inference_mode()
def score_dataset(
    model,
    processor,
    dataset,
    batch_size: int = 4,
    tta_k: int = 1,
    tta_skip_below: int = 0,
    show_progress: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Score every example in `dataset` (which must be in eval/test mode).

    Args:
        tta_k: if >1, run K=tta_k forward passes per example with random
            permutations of the choices, then sum log-probs back in original-
            choice space and argmax. Pass 0 (k=0) always uses the identity
            permutation so tta_k=1 reduces exactly to the no-TTA behavior.
            Recommended: 4 for ~+1-2 points, with diminishing returns past 8.
        tta_skip_below: if >0, examples with num_choices < tta_skip_below
            fall back to k=1 (no TTA) even when tta_k > 1. Useful for
            "smart TTA" where TTA hurts on small choice sets — e.g. set
            tta_skip_below=3 to apply TTA only to 3+ choice questions.
        seed: RNG seed for choice permutations (reproducibility).

    Returns:
        list of {id, pred, gt (or -1), num_choices}.
    """
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(0, len(dataset), batch_size), disable=not show_progress)
    except ImportError:
        iterator = range(0, len(dataset), batch_size)

    rng = random.Random(seed)
    max_letters = len(CFG.choice_letters)
    device = next(model.parameters()).device

    results: list[dict] = []
    for start in iterator:
        end = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, end)]
        B = len(batch)

        # If smart TTA is on and ALL examples in this batch are below the
        # threshold, skip TTA entirely for this batch (no-TTA fast path).
        # If only SOME are below threshold, the per-example merge below
        # handles the mix.
        all_below = (
            tta_skip_below > 0
            and all(b["num_choices"] < tta_skip_below for b in batch)
        )

        if tta_k <= 1 or all_below:
            preds = score_batch(
                model, processor,
                images=[b["image"] for b in batch],
                prompts=[b["prompt"] for b in batch],
                num_choices_per=[b["num_choices"] for b in batch],
            )
        else:
            # Accumulate per-original-choice log-probs across K permutations.
            # Note: argmax is invariant to scaling, so summing is fine — no
            # need to divide by tta_k.
            accum = torch.zeros((B, max_letters), device=device, dtype=torch.float32)

            for k in range(tta_k):
                # Pass 0 = identity (so tta_k=1 matches the no-TTA path).
                # Subsequent passes use random permutations of [0..n-1].
                perms = []
                for b in batch:
                    n = b["num_choices"]
                    if k == 0:
                        perms.append(list(range(n)))
                    else:
                        p = list(range(n))
                        rng.shuffle(p)
                        perms.append(p)

                shuffled_prompts = [
                    build_prompt_from_fields(
                        question=b["question"],
                        choices=[b["choices"][i] for i in p],
                        lecture=b.get("lecture"),
                        hint=b.get("hint"),
                    )
                    for b, p in zip(batch, perms)
                ]

                letter_logits = _score_batch_letter_logits(
                    model, processor,
                    images=[b["image"] for b in batch],
                    prompts=shuffled_prompts,
                    num_choices_per=[b["num_choices"] for b in batch],
                )

                # Map shuffled-position logits back to original-choice slots.
                # If perm = [2, 0, 1], then letter_logits[bi, 0] is the logit
                # for "letter A in shuffled prompt", which IS original choice 2.
                # So accum[bi, 2] += letter_logits[bi, 0]. Etc.
                lf = letter_logits.float()
                for bi, p in enumerate(perms):
                    for shuf_pos, orig_choice in enumerate(p):
                        accum[bi, orig_choice] += lf[bi, shuf_pos]

            # Set positions beyond num_choices to -inf so they can't win argmax.
            # (They're zero-valued because we never wrote to them above.)
            for bi, b in enumerate(batch):
                accum[bi, b["num_choices"]:] = float("-inf")

            preds = accum.argmax(dim=1).cpu().tolist()

            # Smart TTA: for examples below the threshold, replace the TTA
            # prediction with the no-TTA (identity) prediction. Pass 0 in the
            # K loop above already used identity, so we can recover its
            # argmax by re-running just the identity pass on those examples.
            if tta_skip_below > 0:
                below_idx = [bi for bi, b in enumerate(batch)
                             if b["num_choices"] < tta_skip_below]
                if below_idx:
                    sub_batch = [batch[bi] for bi in below_idx]
                    sub_preds = score_batch(
                        model, processor,
                        images=[b["image"] for b in sub_batch],
                        prompts=[b["prompt"] for b in sub_batch],
                        num_choices_per=[b["num_choices"] for b in sub_batch],
                    )
                    for bi, sp in zip(below_idx, sub_preds):
                        preds[bi] = sp

        for b, p in zip(batch, preds):
            results.append({
                "id": b["id"],
                "pred": int(p),
                "gt": b.get("answer", -1),
                "num_choices": b["num_choices"],
            })
    return results


def compute_accuracy(results: list[dict]) -> float:
    """Compute overall accuracy on results that have ground truth."""
    n = sum(1 for r in results if r["gt"] != -1)
    if n == 0:
        return float("nan")
    correct = sum(1 for r in results if r["gt"] != -1 and r["pred"] == r["gt"])
    return correct / n


def compute_accuracy_by_num_choices(results: list[dict]) -> dict:
    """Per-num_choices breakdown: {nc: {n, correct, acc}} for each num_choices.

    Useful for spotting where the model struggles (typically 5-choice in
    ScienceQA, where the base model rarely saw 'E' as an answer choice in
    pretraining).
    """
    buckets: dict[int, dict] = {}
    for r in results:
        if r["gt"] == -1:
            continue
        nc = r.get("num_choices")
        if nc is None:
            continue
        b = buckets.setdefault(nc, {"n": 0, "correct": 0})
        b["n"] += 1
        if r["pred"] == r["gt"]:
            b["correct"] += 1
    return {
        nc: {"n": v["n"], "correct": v["correct"],
             "acc": (v["correct"] / v["n"]) if v["n"] else float("nan")}
        for nc, v in sorted(buckets.items())
    }
