"""
Dataset and prompt construction for the ScienceQA visual MC task.

Key design decisions:
  1. Prompt format matches the starter notebook exactly so the model sees the
     same template at train and test time.
  2. For training we use SFT with loss masked to the answer-letter tokens
     only. Without masking, ~99% of the loss signal would go into modeling
     the lecture/hint/question text — wasted under a 5M-param budget.
  3. We don't pre-resize images here — let the processor handle it. The
     starter's manual 224x224 bicubic resize destroys detail in diagrams.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CFG


# ── Kaggle competition-data staging ──────────────────────────────────────────

def stage_competition_data(
    raw_data_dir: Path | str,
    staged_dir: Path | str = "/kaggle/working/staged_data",
) -> Path:
    """Resolve the duplicated images/ directory present in some Kaggle uploads.

    Kaggle competition zips can extract with a nested ``images/images/<split>/``
    layout (zip-of-already-zipped quirk). ScienceQADataset constructs paths as
    ``data_dir / row['image_path']`` where ``image_path`` already starts with
    ``images/...`` — so the nested layout fails with FileNotFoundError.

    This helper detects the duplication and symlinks both CSVs and the inner
    images/ directory into a unified staged layout under /kaggle/working/.
    If no duplication exists (e.g., local or Colab), it returns ``raw_data_dir``
    unchanged — making this a safe no-op outside Kaggle.

    Args:
        raw_data_dir: where the competition CSVs live (e.g.,
            ``/kaggle/input/competitions/<comp>``).
        staged_dir: target dir under /kaggle/working/ to stage symlinks into.
            Defaults to ``/kaggle/working/staged_data``.

    Returns:
        The Path to use as ``data_dir`` for ``load_splits`` and
        ``ScienceQADataset``: either ``raw_data_dir`` unchanged, or the
        staged dir.
    """
    raw_data_dir = Path(raw_data_dir)
    inner_images = raw_data_dir / "images" / "images"

    if not inner_images.exists():
        return raw_data_dir

    staged_dir = Path(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    for csv_name in ("test.csv", "val.csv", "train.csv"):
        src = raw_data_dir / csv_name
        dst = staged_dir / csv_name
        if src.exists() and not dst.exists():
            os.symlink(src, dst)

    if not (staged_dir / "images").exists():
        os.symlink(inner_images, staged_dir / "images")

    return staged_dir


# ── Prompt construction ──────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def build_prompt_from_fields(
    question: str,
    choices: list,
    lecture=None,
    hint=None,
    answer_letter: str | None = None,
) -> str:
    """Build the prompt from explicit fields. Used by TTA-aware scoring,
    which needs to rebuild the prompt with permuted choices.
    """
    context_parts = []
    if lecture is not None and pd.notna(lecture) and str(lecture).strip():
        context_parts.append(_truncate(str(lecture), CFG.truncate_lecture_chars))
    if hint is not None and pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices_str = "\n".join(
        f"  {CFG.choice_letters[i]}. {c}" for i, c in enumerate(choices)
    )

    prompt = "<image>\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {question}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += "Answer:"
    if answer_letter is not None:
        prompt += f" {answer_letter}"
    return prompt


def build_prompt_text(row: pd.Series, include_answer: bool = False) -> str:
    """Identical structure to starter, but with optional lecture truncation.

    Note: the <image> token marks where the processor will splice visual
    tokens. Don't rename it.
    """
    answer_letter = None
    if include_answer:
        answer_letter = CFG.choice_letters[int(row["answer"])]
    return build_prompt_from_fields(
        question=row["question"],
        choices=row["choices"],
        lecture=row.get("lecture"),
        hint=row.get("hint"),
        answer_letter=answer_letter,
    )


# ── Dataset classes ──────────────────────────────────────────────────────────

class ScienceQADataset(Dataset):
    """Returns dicts with PIL image + raw text. The collator does the
    processor call and (for training) the loss-mask construction.

    Two modes:
      - train: includes the answer letter in the prompt (for SFT)
      - eval/test: no answer in prompt (for log-likelihood scoring)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        mode: str = "train",  # "train" | "eval" | "test"
    ):
        assert mode in ("train", "eval", "test")
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img = Image.open(self.data_dir / row["image_path"]).convert("RGB")

        if self.mode == "train":
            return {
                "image": img,
                "text": build_prompt_text(row, include_answer=True),
                "answer_letter": CFG.choice_letters[int(row["answer"])],
                "id": str(row["id"]),
            }
        else:
            return {
                "image": img,
                "prompt": build_prompt_text(row, include_answer=False),
                "choices": list(row["choices"]),
                # Raw fields exposed so TTA scoring can rebuild the prompt with
                # permuted choices without re-reading the CSV.
                "question": str(row["question"]),
                "lecture": row.get("lecture"),
                "hint": row.get("hint"),
                "num_choices": int(row["num_choices"]) if "num_choices" in row else len(row["choices"]),
                "answer": int(row["answer"]) if "answer" in row and pd.notna(row.get("answer")) else -1,
                "id": str(row["id"]),
            }


# ── Loss-masked training collator ────────────────────────────────────────────

class TrainCollator:
    """Builds (input_ids, attention_mask, pixel_values, labels) where labels
    are -100 everywhere except the final answer letter token.

    The trick: the prompt ends with `Answer: X`. We tokenize the full text,
    then set labels[i] = input_ids[i] only for tokens that fall within the
    answer-letter span at the end. Everything else gets -100 (ignore_index).
    """

    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]

        enc = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            # NO truncation: VLM prompts can't be safely truncated mid-stream
            # because cutting inside the visual-token block desyncs the visual
            # features from their placeholder positions and the processor
            # raises. Length control happens upstream via
            # CFG.truncate_lecture_chars in build_prompt_text().
        )

        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        if CFG.mask_prompt_loss:
            # Find the answer-letter token at the end of each sequence and
            # zero out everything before it.
            #
            # Strategy: locate the substring " {LETTER}" at end of each text.
            # Tokenize just that suffix to know how many tokens to keep.
            for i, b in enumerate(batch):
                letter = b["answer_letter"]
                # The answer is at the very end of the prompt; we want labels
                # to cover only the trailing letter token(s).
                # Tokenize " A", " B", etc. without special tokens to count.
                suffix_ids = self.tokenizer(
                    f" {letter}", add_special_tokens=False
                )["input_ids"]
                n_suffix = len(suffix_ids)

                # The non-pad length of this row (right-padded by default in HF):
                row_attn = enc["attention_mask"][i]
                nonpad_len = int(row_attn.sum().item())

                # Mask everything except the last n_suffix non-pad tokens.
                mask = torch.full_like(labels[i], -100)
                start = nonpad_len - n_suffix
                end = nonpad_len
                mask[start:end] = labels[i, start:end]
                labels[i] = mask

        enc["labels"] = labels
        return dict(enc)


# ── Helpers to load/parse the CSVs ───────────────────────────────────────────

def load_splits(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    for df in (train_df, val_df, test_df):
        df["choices"] = df["choices"].apply(json.loads)

    return train_df, val_df, test_df
