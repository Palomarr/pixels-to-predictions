"""
Generate submission.csv from one or more trained LoRA adapters.

This is the script that runs in the OFFLINE Kaggle eval environment, so it
must:
  1. Not hit the network. Load the base model from a Kaggle dataset path,
     not from HuggingFace Hub.
  2. Load the LoRA adapter(s) from Kaggle dataset path(s).
  3. Run log-likelihood scoring on test.csv (with optional smart-TTA).
  4. Write submission.csv with id,answer columns.

Single adapter:
    python make_submission.py \
        --base_model_path /kaggle/input/smolvlm-500m-instruct \
        --adapter_path    /kaggle/input/p2p-adapter/final_adapter \
        --data_dir        /kaggle/input/pixels-to-predictions/data \
        --out             submission.csv \
        --tta_k 4 --tta_skip_below 3

Ensemble of multiple adapters (logit-averaged):
    python make_submission.py \
        --adapter_path    /path/to/v3_adapter /path/to/v6_adapter \
        ...   # other args identical
"""
from __future__ import annotations
import argparse
import gc
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
    p.add_argument("--adapter_path", type=str, nargs="+", required=True,
                   help="One or more adapter paths. Multiple paths trigger "
                        "logit-averaged ensembling across adapters.")
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

    adapter_paths = args.adapter_path  # list of strings (1 or more)

    if len(adapter_paths) == 1:
        # ── Single-adapter path (existing behavior) ──────────────────────────
        model, processor = load_inference_model(
            args.base_model_path, adapter_paths[0], args.use_4bit
        )
        results = score_dataset(
            model, processor, test_ds,
            batch_size=args.batch_size,
            tta_k=args.tta_k,
            tta_skip_below=args.tta_skip_below,
        )
    else:
        # ── Ensemble path: load each adapter in turn, accumulate logits ──────
        # Per-example logits (post-TTA, post-masking) sum across adapters; the
        # final argmax over the sum gives the ensemble prediction. This is
        # equivalent to averaging log-probs in original-choice space.
        print(f"\n[ensemble] Logit-averaging across {len(adapter_paths)} adapters")
        for i, p in enumerate(adapter_paths):
            print(f"  [{i+1}] {p}")

        # Accumulators keyed by example id (test set has no duplicates)
        ensemble_logits: dict = {}
        gt_by_id: dict = {}
        nc_by_id: dict = {}

        for i, adapter_path in enumerate(adapter_paths):
            print(f"\n[ensemble {i+1}/{len(adapter_paths)}] Loading {adapter_path}")
            model, processor = load_inference_model(
                args.base_model_path, adapter_path, args.use_4bit
            )

            adapter_results = score_dataset(
                model, processor, test_ds,
                batch_size=args.batch_size,
                tta_k=args.tta_k,
                tta_skip_below=args.tta_skip_below,
                return_logits=True,
            )

            for r in adapter_results:
                eid = r["id"]
                if eid not in ensemble_logits:
                    ensemble_logits[eid] = r["logits"].clone()
                    gt_by_id[eid] = r["gt"]
                    nc_by_id[eid] = r["num_choices"]
                else:
                    ensemble_logits[eid] = ensemble_logits[eid] + r["logits"]

            # Free GPU memory before loading the next adapter
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Build results from accumulated logits — argmax once at the end
        results = []
        for eid, logits in ensemble_logits.items():
            results.append({
                "id": eid,
                "pred": int(logits.argmax().item()),
                "gt": gt_by_id[eid],
                "num_choices": nc_by_id[eid],
            })

    # Write submission CSV
    sub = pd.DataFrame([{"id": r["id"], "answer": int(r["pred"])} for r in results])

    # Sanity: order matches test_df, all integers, all ids present
    assert set(sub["id"]) == set(test_df["id"].astype(str)), "id mismatch with test.csv"
    assert sub["answer"].dtype.kind == "i", "answer column must be integer"

    sub.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(sub)} rows")
    print(sub.head())


if __name__ == "__main__":
    main()
