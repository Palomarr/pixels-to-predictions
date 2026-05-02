# Pixels-to-Predictions: SmolVLM-500M QLoRA Fine-Tuning for ScienceQA

LoRA-fine-tuned `HuggingFaceTB/SmolVLM-500M-Instruct` for visual multiple-choice
science QA. Designed for the 5M trainable-parameter cap and offline Kaggle
evaluation.

**Best submission**: v3 adapter + smart-TTA → public LB **0.81287**.
Full methodology, ablations, and discussion are in `report/report.pdf`.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one CFG dataclass + the `ABLATIONS` registry of named profiles. |
| `data.py` | Dataset, prompt building, loss-masked SFT collator, Kaggle-data staging helper. |
| `build_model.py` | Loads SmolVLM in 4-bit + applies LoRA. Verifies <5M trainable. |
| `scoring.py` | Log-likelihood scoring with optional smart-TTA. |
| `train.py` | Training entry point (HF Trainer + custom collator). Supports `--ablation NAME`. |
| `make_submission.py` | Loads adapter, scores test set, writes `submission.csv`. Supports `--tta_k` / `--tta_skip_below`. |
| `smoke_test.py` | End-to-end pipeline check on tiny data — run this first. |
| `p2p_train.ipynb` | Kaggle training notebook (thin wrapper around `train.py`). |
| `p2p_submit.ipynb` | Kaggle offline submission notebook (thin wrapper around `make_submission.py`). |
| `report/report.typ`, `report/report.pdf` | Final report. |

## Workflow

### 1. Smoke test (run first; catches data/path/regex issues)

```bash
python smoke_test.py --data_dir data
```

Validates: data loads, prompts build, collator masks correctly, LoRA regex
matches the expected modules, model forward+backward works, log-likelihood
scoring produces sensible predictions. ~2-5 min on a T4.

### 2. Train

Reproduce a registered ablation profile:

```bash
python train.py --ablation v3_epochs5 --data_dir data
```

Available profiles in `config.ABLATIONS`:

| Profile | Config |
|---|---|
| `v1_baseline` | r=8, all-7 projections, 384px, 3 epochs |
| `v2_img512` | as v1 + img_size=512 |
| **`v3_epochs5`** | **as v1 + 5 epochs (best submission)** |
| `v4_epochs7` | as v1 + 7 epochs (overfitting control) |
| `v5_r4` | r=4, alpha=8, 5 epochs (capacity-halving control) |

Each profile saves a ~17 MB LoRA adapter to `outputs/<run_name>/final_adapter/`
and a config-resolved `val_results.json` alongside.

CLI flags override individual fields of an ablation:
```bash
python train.py --ablation v3_epochs5 --epochs 6 --run_name my_test
```

For a fast pipeline test on tiny data:
```bash
python train.py --smoke_test
```

### 3. Generate submission (offline-safe on Kaggle)

Bundle two Kaggle Datasets: the base SmolVLM model files and your trained
adapter (`final_adapter/` from step 2). Then in the offline submission notebook:

```bash
python make_submission.py \
    --base_model_path /kaggle/input/smolvlm-500m-instruct \
    --adapter_path    /kaggle/input/p2p-adapter/final_adapter \
    --data_dir        /kaggle/input/pixels-to-predictions/data \
    --out             submission.csv \
    --tta_k           4 \
    --tta_skip_below  3
```

`--tta_skip_below=3` is the smart-TTA setting: K=4 choice-permutation TTA on
3+ choice questions, but bypasses TTA on 2-choice questions where empirical
val measurements showed it hurts. See `report/report.typ` §2.5 for the
mechanism and §5.2 for the per-bucket measurements.

## Key design decisions (full discussion in `report/`)

- **QLoRA, all 7 projections, r=8** — 4,341,760 trainable params (87% of 5M cap).
  Targets `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` in the text
  decoder only, via a regex that excludes the vision encoder's same-named layers.
- **Loss masked to the answer letter token** — concentrates 100% of gradient
  signal on "predict the right letter" rather than language-modeling the
  lecture/hint context.
- **Log-likelihood scoring at inference** — single forward pass per example,
  computing `log P(letter_j | image, prompt)` for each valid choice; argmax
  predicts the answer. Avoids the brittleness of generation parsing.
- **Smart-TTA** — choice-permutation TTA at K=4, but bypassed for 2-choice
  questions where it empirically hurts (see report §5.2). +1.21 pp LB.
- **QLoRA stability fixes** — `paged_adamw_8bit` + `max_grad_norm=0.3` +
  `warmup_ratio=0.1` are necessary to prevent fp16 loss explosion on T4.

## Adding a new ablation

1. Add a new entry to `config.ABLATIONS` with descriptive key + CFG overrides.
2. (If needed) add a CFG flag for new behavior and wire it through.
3. Run: `python train.py --ablation <new_profile_name>`
4. Each completed run writes `outputs/<run_name>/val_results.json` with the
   resolved config and val accuracy, pairing recipe and measurement.
