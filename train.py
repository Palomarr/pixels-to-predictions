"""
Training entry point. Uses HF Trainer for the boilerplate (gradient accum,
mixed precision, checkpointing, logging) and our custom collator for the
loss-masked SFT.

Usage:
    # Reproduce one of the named ablations from config.ABLATIONS:
    python train.py --ablation v3_epochs5

    # Or specify hyperparameters directly:
    python train.py --epochs 5 --lora_r 8 --run_name my_run

    # CLI flags also override an ablation profile:
    python train.py --ablation v3_epochs5 --epochs 6  # v3 settings, but 6 epochs

Designed to run on a single T4 (Kaggle/Colab free tier) or your RTX 4090.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import CFG, ABLATIONS, apply_ablation
from data import load_splits, ScienceQADataset, TrainCollator
from build_model import build_model, build_processor
from scoring import score_dataset, compute_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    # Named ablation profile (applied first, then CLI flags can override fields).
    p.add_argument("--ablation", type=str, default=None,
                   choices=sorted(ABLATIONS.keys()),
                   help="Named ablation profile from config.ABLATIONS.")
    # Individual overrides — default None means "don't touch CFG for this field".
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--smoke_test", action="store_true",
                   help="Use 64 train / 32 val examples for fast pipeline check.")
    return p.parse_args()


def main():
    args = parse_args()

    # Step 1: apply named ablation profile (if any). Sets multiple CFG fields.
    if args.ablation:
        applied = apply_ablation(args.ablation)
        print(f"[ablation] Applied profile '{args.ablation}': {applied}")

    # Step 2: apply individual CLI overrides on top of the ablation profile.
    # These are guarded by `is not None` so omitted flags don't clobber CFG.
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.lr is not None:
        CFG.learning_rate = args.lr
    if args.lora_r is not None:
        CFG.lora_r = args.lora_r
    if args.img_size is not None:
        CFG.img_size = args.img_size
    if args.data_dir is not None:
        CFG.data_dir = Path(args.data_dir)
    if args.output_dir is not None:
        CFG.output_dir = Path(args.output_dir)
    if args.run_name is not None:
        CFG.run_name = args.run_name

    # Recover from any corrupted CUDA state in the notebook before our first
    # cuda op. Kaggle kernels share CUDA state across cells, so a prior failed
    # run (e.g. NaN/inf weights from the loss explosion) can leave the device
    # in a state where set_seed below raises "illegal memory access". If this
    # still fails after the cleanup, restart the kernel.
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"[CUDA] {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    set_seed(CFG.seed)
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df, val_df, _ = load_splits(CFG.data_dir)
    if args.smoke_test:
        train_df = train_df.head(64)
        val_df = val_df.head(32)
        print(f"[smoke_test] Using {len(train_df)} train / {len(val_df)} val")

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Model + processor ────────────────────────────────────────────────────
    processor = build_processor()
    model = build_model(for_training=True)

    # ── Datasets and collator ────────────────────────────────────────────────
    train_ds = ScienceQADataset(train_df, CFG.data_dir, mode="train")
    val_ds = ScienceQADataset(val_df, CFG.data_dir, mode="eval")
    collator = TrainCollator(processor, max_length=CFG.max_seq_length)

    # ── TrainingArguments ────────────────────────────────────────────────────
    # Note: we don't use Trainer's eval — we run our own log-likelihood eval
    # after training (free-form generation eval would not match how we score).
    targs = TrainingArguments(
        output_dir=str(CFG.output_dir / CFG.run_name),
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.per_device_batch_size,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler,
        logging_steps=20,
        save_strategy="steps",
        save_steps=CFG.save_every_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,  # collator needs raw dict fields
        dataloader_num_workers=2,
        dataloader_pin_memory=False,  # PIL images can't be pinned anyway
        seed=CFG.seed,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save the LoRA adapter (small file, ~17MB) ────────────────────────────
    final_dir = CFG.output_dir / CFG.run_name / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved adapter to: {final_dir}")

    # ── Evaluate with log-likelihood scoring ─────────────────────────────────
    print("Running log-likelihood eval on validation set...")
    trainer.model.eval()
    results = score_dataset(trainer.model, processor, val_ds, batch_size=4)
    val_acc = compute_accuracy(results)
    print(f"Val accuracy: {val_acc:.4f}")

    # Save val results for the report
    eval_path = CFG.output_dir / CFG.run_name / "val_results.json"
    eval_path.write_text(json.dumps({
        "run_name": CFG.run_name,
        "ablation": args.ablation,  # which named profile (if any) was applied
        "val_accuracy": val_acc,
        "n_val": len(results),
        "config": {
            "lora_r": CFG.lora_r,
            "lora_alpha": CFG.lora_alpha,
            "lora_target_regex": CFG.lora_target_regex,
            "epochs": CFG.epochs,
            "learning_rate": CFG.learning_rate,
            "img_size": CFG.img_size,
            "batch_size_effective": CFG.per_device_batch_size * CFG.grad_accum_steps,
            "warmup_ratio": CFG.warmup_ratio,
            "max_grad_norm": 0.3,  # set in TrainingArguments above
        },
    }, indent=2))
    print(f"Saved eval results to: {eval_path}")


if __name__ == "__main__":
    main()
