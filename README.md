# Pixels-to-Predictions: SmolVLM-500M QLoRA Fine-Tuning for ScienceQA

LoRA-fine-tuned `HuggingFaceTB/SmolVLM-500M-Instruct` for visual multiple-choice
science QA. Designed for the 5M trainable-parameter cap and offline Kaggle
evaluation.

**Final submission**: `v3 + v6` adapter ensemble + smart-TTA → public LB **0.81690**.
Full methodology, ablations, and discussion are in `report/report.pdf`.

## Quickstart for graders

To reproduce the final submission's adapters from scratch (~10 hours T4):

```bash
# 1. Install dependencies (skip on Kaggle; see notebooks for pinned versions)
pip install -r requirements.txt

# 2. Validate the pipeline on tiny data (~3 min on T4)
python smoke_test.py --data_dir data

# 3. Train both adapters used in the final ensemble
python train.py --ablation v3_epochs5 --data_dir data    # ~5h, train-val 0.8015
python train.py --ablation v6_dora --data_dir data        # ~6h, train-val 0.7948

# 4. Generate the final ensemble submission
python make_submission.py \
    --base_model_path /path/to/smolvlm-500m-instruct \
    --adapter_path    outputs/smolvlm_qlora_r8_all7_v3_epochs5/final_adapter \
                      outputs/smolvlm_qlora_dora_r8_all7_v6_epochs5/final_adapter \
    --data_dir        data \
    --out             submission.csv \
    --tta_k           4 \
    --tta_skip_below  3
```

To reproduce any single ablation row in the report's @ablation-table, pick a
profile from `config.ABLATIONS` and run `python train.py --ablation <name>`.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one CFG dataclass + the `ABLATIONS` registry of named profiles + `apply_ablation()` helper. |
| `data.py` | Dataset, prompt building, loss-masked SFT collator, `stage_competition_data()` Kaggle-staging helper. |
| `build_model.py` | Loads SmolVLM in 4-bit + applies LoRA (or DoRA via `lora_use_dora=True`). Asserts <5M trainable. |
| `scoring.py` | Log-likelihood scoring with smart-TTA. Supports `return_logits=True` for adapter ensembling. |
| `train.py` | Training entry (HF Trainer + custom collator). Supports `--ablation NAME` + per-field overrides. |
| `make_submission.py` | Loads adapter(s), scores test set with smart-TTA, writes `submission.csv`. Supports multi-adapter logit-averaged ensembling via `--adapter_path A B C`. |
| `smoke_test.py` | End-to-end pipeline check on tiny data — run this first when changing config. |
| `p2p_train.ipynb` | Kaggle training notebook (thin wrapper around `train.py`). |
| `p2p_submit.ipynb` | Kaggle offline submission notebook (thin wrapper around `make_submission.py`). |
| `requirements.txt` | Pinned package versions for non-Kaggle reproduction. |
| `report/report.typ` / `.pdf` | Final report (Typst source + compiled). |

## Ablation registry

All seven measured runs are registered in `config.ABLATIONS`:

| Profile | Description | Train-val (4-bit) | LB (bf16 + smart-TTA) |
|---|---|---:|---:|
| `v1_baseline` | r=8, all-7 projections, 384px, 3 epochs | 0.7634 | 0.77665 |
| `v2_img512` | + img_size=512 | 0.7586 | 0.77867 |
| `v3_epochs5` | + epochs=5 | 0.8015 | 0.81287 |
| `v4_epochs7` | + epochs=7 (overfitting control) | 0.7996 | 0.80281 |
| `v5_r4` | r=4, alpha=8, 5 epochs (capacity-halving control) | 0.7729 | 0.79476 |
| `v6_dora` | DoRA, r=8, 5 epochs | 0.7948 | 0.81488 |
| `v7_choice_aug` | + train-time choice-order augmentation | 0.7634 | not submitted |
| **(ensemble)** | **`v3 + v6` logit-averaged + smart-TTA** | — | **0.81690** |

CLI flags override individual fields of an ablation:

```bash
python train.py --ablation v3_epochs5 --epochs 6 --run_name my_test
```

Each run writes a `val_results.json` to `outputs/<run_name>/` with the resolved
config and val accuracy — pairing recipe (registry) with measurement (json).

## Generating submissions

### Single adapter

```bash
python make_submission.py \
    --base_model_path /kaggle/input/smolvlm-500m-instruct \
    --adapter_path    /kaggle/input/p2p-adapter-v3/final_adapter \
    --data_dir        /kaggle/input/pixels-to-predictions/data \
    --out             submission.csv \
    --tta_k           4 \
    --tta_skip_below  3
```

### Adapter ensemble (logit-averaged)

```bash
python make_submission.py \
    --base_model_path /kaggle/input/smolvlm-500m-instruct \
    --adapter_path    /kaggle/input/p2p-adapter-v3/final_adapter \
                      /kaggle/input/p2p-adapter-v6/final_adapter \
    --data_dir        /kaggle/input/pixels-to-predictions/data \
    --out             submission.csv \
    --tta_k           4 \
    --tta_skip_below  3
```

`--tta_skip_below=3` is smart-TTA: K=4 choice-permutation TTA on 3+ choice
questions, but bypasses TTA on 2-choice questions where empirical val
measurements showed it hurts (-1.23 pp on val). See `report/report.typ` §2.5
for the mechanism, §5.2 for per-bucket measurements.

## Key design decisions (full discussion in `report/`)

- **QLoRA, all 7 projections, r=8** — 4,341,760 trainable params (87% of 5M cap).
  Targets `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` in the text
  decoder only, via a regex that excludes the vision encoder's same-named layers.
- **Loss masked to the answer-letter token** — concentrates 100% of gradient
  signal on "predict the right letter" rather than language-modeling the
  lecture/hint context.
- **Log-likelihood scoring at inference** — single forward pass per example,
  computing `log P(letter_j | image, prompt)` for each valid choice; argmax
  predicts the answer. Avoids the brittleness of generation parsing.
- **Smart-TTA** — choice-permutation TTA at K=4, but bypassed for 2-choice
  questions where it empirically hurts. +1.21 pp LB on v1.
- **QLoRA stability fixes** — `paged_adamw_8bit` + `max_grad_norm=0.3` +
  `warmup_ratio=0.1` are necessary to prevent fp16/bf16 loss explosion on T4.

## Known caveats / offline-eval notes

- **4-bit-vs-bf16 inference gap (~3.4 pp on v1)**: training uses 4-bit NF4
  via `bitsandbytes`, but Kaggle's offline-evaluation environment lacks
  `bitsandbytes`, so submission inference uses bf16. The LoRA adapter is
  optimized for 4-bit-dequantized weights and is then applied to original bf16
  weights — introducing a precision mismatch. We treat this as forced by the
  competition's rule #5 (no internet at evaluation time).
- **bf16 fallback on T4 (Turing/sm_75)**: PyTorch 2.10 reports
  `is_bf16_supported=True` on T4 but T4 has no native bf16 tensor cores; ops
  fall back to emulated kernels. Throughput penalty was not separately measured.
- **`images/images/` duplication in the competition zip**: handled by
  `data.stage_competition_data()`, which symlinks the inner `images/` directory
  to the staged location. No-op outside Kaggle.
- **`transformers v5` removed `AutoModelForVision2Seq`**: handled by a
  `try/except` import fallback to `AutoModelForImageTextToText` in
  `build_model.py` and `make_submission.py`.
- **CUDA state corruption between Kaggle cells**: handled by a defensive
  `torch.cuda.empty_cache() + synchronize()` before `set_seed()` in `train.py`.

## Adding a new ablation

1. Add a new entry to `config.ABLATIONS` with descriptive key + CFG overrides.
2. (If needed) add a CFG field for new behavior and wire it through the
   relevant module (`build_model.py`, `data.py`, etc.).
3. Run: `python train.py --ablation <new_profile_name>`
4. Each completed run writes `outputs/<run_name>/val_results.json` with the
   resolved config and val accuracy, pairing recipe and measurement.

## License & attribution

Course final project for NYU Deep Learning. Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`.
Dataset: ScienceQA-style multimodal multiple-choice QA (provided by the competition).
