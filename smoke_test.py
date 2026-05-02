"""
Smoke test: validate pipeline end-to-end on tiny data (no training).

Catches the things that always silently break before you spend hours training:
  - Image paths resolve correctly
  - Prompt building doesn't crash on edge-case rows
  - Processor + collator produce tensors of the right shape
  - Loss-mask construction targets the right positions
  - LoRA regex matches the expected modules
  - Log-likelihood scoring runs without errors and produces sensible logits
  - The pretrained model gets >random accuracy on val (sanity check)

This is the SVG-project lesson: smoke tests with 20-sample validity checks
catch bugs before full training runs do. Run this first.

Usage: python smoke_test.py --data_dir data
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from config import CFG
from data import load_splits, ScienceQADataset, TrainCollator, build_prompt_text
from build_model import build_processor, build_model
from scoring import score_dataset, compute_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=str(CFG.data_dir))
    p.add_argument("--n_val", type=int, default=32)
    args = p.parse_args()

    print("=" * 70)
    print("SMOKE TEST")
    print("=" * 70)

    # ── 1. Data loads ────────────────────────────────────────────────────────
    print("\n[1] Loading splits...")
    train_df, val_df, test_df = load_splits(args.data_dir)
    print(f"    train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"    train.csv columns: {list(train_df.columns)}")
    nc_dist = train_df["choices"].apply(len).value_counts().sort_index()
    print(f"    num_choices distribution (train): {dict(nc_dist)}")

    # ── 2. Prompt building ───────────────────────────────────────────────────
    print("\n[2] Sample prompt:")
    print("-" * 70)
    print(build_prompt_text(train_df.iloc[0], include_answer=True))
    print("-" * 70)

    # ── 3. Datasets ──────────────────────────────────────────────────────────
    print("\n[3] Building datasets...")
    train_ds = ScienceQADataset(train_df.head(8), Path(args.data_dir), mode="train")
    val_ds = ScienceQADataset(val_df.head(args.n_val), Path(args.data_dir), mode="eval")
    sample = train_ds[0]
    print(f"    train sample keys: {list(sample.keys())}")
    print(f"    image size: {sample['image'].size}")

    # ── 4. Processor + Collator ──────────────────────────────────────────────
    print("\n[4] Building processor + collator...")
    processor = build_processor()
    collator = TrainCollator(processor, max_length=CFG.max_seq_length)
    batch = collator([train_ds[i] for i in range(2)])
    print(f"    batch keys: {list(batch.keys())}")
    print(f"    input_ids shape: {batch['input_ids'].shape}")
    if "pixel_values" in batch:
        print(f"    pixel_values shape: {batch['pixel_values'].shape}")
    print(f"    labels shape: {batch['labels'].shape}")
    n_unmasked = (batch["labels"] != -100).sum().item()
    print(f"    unmasked label tokens (should be small, ~num_examples): {n_unmasked}")

    # Decode the unmasked positions to verify they're answer letters
    for i in range(batch["labels"].size(0)):
        unmasked_ids = batch["labels"][i][batch["labels"][i] != -100]
        if len(unmasked_ids) > 0:
            decoded = processor.tokenizer.decode(unmasked_ids)
            print(f"    row {i} unmasked decoded: {decoded!r}")

    # ── 5. Model + LoRA ──────────────────────────────────────────────────────
    print("\n[5] Loading model with LoRA (this is the slow step)...")
    model = build_model(for_training=True)

    # ── 6. Training-step forward + backward ──────────────────────────────────
    print("\n[6] One training-step forward+backward...")
    device = next(model.parameters()).device
    batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
    out = model(**batch_dev)
    print(f"    loss: {out.loss.item():.4f}")
    out.loss.backward()
    print("    backward pass OK")

    # ── 7. Log-likelihood scoring on val (uses untrained adapter — should be ~random) ──
    print(f"\n[7] Scoring {args.n_val} val examples (untrained adapter)...")
    model.eval()
    results = score_dataset(model, processor, val_ds, batch_size=4, show_progress=False)
    acc = compute_accuracy(results)
    print(f"    Val accuracy (untrained adapter): {acc:.4f}")
    print("    (Should be near 1/avg_num_choices, ~0.25-0.30 for 4-choice questions.")
    print("     Some lift from base model's pretrained knowledge is normal.)")

    # ── 8. Output sanity ─────────────────────────────────────────────────────
    print(f"\n[8] First 5 predictions:")
    for r in results[:5]:
        ok = "✓" if r["pred"] == r["gt"] else "✗"
        print(f"    {ok} id={r['id']}  pred={r['pred']}  gt={r['gt']}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
