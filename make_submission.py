"""
Generate submission.csv from a trained LoRA adapter.

This is the script that runs in the OFFLINE Kaggle eval environment, so it
must:
  1. Not hit the network. Load the base model from a Kaggle dataset path,
     not from HuggingFace Hub.
  2. Load the LoRA adapter from a Kaggle dataset path.
  3. Run log-likelihood scoring on test.csv.
  4. Write submission.csv with id,answer columns.

Usage:
    python make_submission.py \
        --base_model_path /kaggle/input/smolvlm-500m-instruct \
        --adapter_path    /kaggle/input/p2p-adapter/final_adapter \
        --data_dir        /kaggle/input/pixels-to-predictions/data \
        --out             submission.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor
# transformers v5+ removed AutoModelForVision2Seq in favor of AutoModelForImageTextToText.
# Submission runs offline with whatever transformers ships on Kaggle, which differs from
# the version we pinned during training — so handle both.
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
from peft import PeftModel

from config import CFG
from data import load_splits, ScienceQADataset
from scoring import score_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, default=CFG.model_id,
                   help="HF id (online) or local path (offline) to the base model")
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=str(CFG.data_dir))
    p.add_argument("--out", type=str, default="submission.csv")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--use_4bit", action="store_true",
                   help="Load base in 4-bit (saves VRAM but slower at inference)")
    p.add_argument("--tta_k", type=int, default=1,
                   help="Choice-permutation TTA passes (1=no TTA, 4=recommended).")
    p.add_argument("--tta_skip_below", type=int, default=0,
                   help="Skip TTA on examples with num_choices below this "
                        "threshold. Useful: tta_skip_below=3 skips 2-choice "
                        "where TTA empirically hurts.")
    return p.parse_args()


def load_inference_model(base_path: str, adapter_path: str, use_4bit: bool):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            base_path,
            quantization_config=bnb,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        base = AutoModelForImageTextToText.from_pretrained(
            base_path,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(adapter_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    return model, processor


def main():
    args = parse_args()

    # Load test split
    _, _, test_df = load_splits(args.data_dir)
    test_ds = ScienceQADataset(test_df, Path(args.data_dir), mode="test")
    print(f"Test: {len(test_ds):,}")

    # Load model + adapter
    model, processor = load_inference_model(
        args.base_model_path, args.adapter_path, args.use_4bit
    )

    # Score every example
    results = score_dataset(
        model, processor, test_ds,
        batch_size=args.batch_size,
        tta_k=args.tta_k,
        tta_skip_below=args.tta_skip_below,
    )

    # Write submission CSV
    sub = pd.DataFrame([{"id": r["id"], "answer": int(r["pred"])} for r in results])

    # Sanity: order matches test_df, all integers, all ids present
    assert set(sub["id"]) == set(test_df["id"].astype(str)), "id mismatch with test.csv"
    assert sub["answer"].dtype.kind == "i", "answer column must be integer"

    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(sub)} rows")
    print(sub.head())


if __name__ == "__main__":
    main()
