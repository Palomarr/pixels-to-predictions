"""
Configuration for Pixels-to-Predictions (SmolVLM-500M ScienceQA).

Centralizing config means you can change one value and have it propagate to
training, scoring, and submission. Mirrors the structure used in the SVG
competition pipeline (TECH_LOG-style: every constant has a justification).

Each named ablation we ran is registered in ABLATIONS below. Use
`apply_ablation("v3_epochs5")` (or `python train.py --ablation v3_epochs5`)
to set CFG to that exact configuration — the recipe registry that pairs
with each run's `val_results.json` for full reproducibility.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    # On Kaggle: data lives at /kaggle/input/<dataset>/data
    # On Colab/local: adjust DATA_DIR to wherever you mounted/extracted the data.
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # ── Model ────────────────────────────────────────────────────────────────
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"

    # ── LoRA (per the budget calc: 4.34M params, 87% of the 5M cap) ──────────
    lora_r: int = 8
    lora_alpha: int = 16            # 2*r heuristic
    lora_dropout: float = 0.05
    # DoRA (decomposed LoRA) replaces the additive update with a magnitude-direction
    # decomposition. ~+0.28M params (one magnitude scalar per output channel) but
    # zero inference-time cost once weights are merged. Slower training (~10-20%).
    lora_use_dora: bool = False
    # IMPORTANT: target only LM-side projections. We use a regex below.
    # SmolVLM uses Idefics3 architecture: text decoder is at .model.text_model
    lora_target_regex: str = (
        r".*model\.text_model\.layers\.\d+\.(self_attn|mlp)"
        r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )

    # ── Quantization (QLoRA, 4-bit NF4) ──────────────────────────────────────
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # compute_dtype is set in build_model.py based on GPU capability

    # ── Image preprocessing ──────────────────────────────────────────────────
    # SmolVLM's processor handles resizing internally. The starter resizes to
    # 224x224 before passing to processor; we keep that default but it's worth
    # ablating (longer_edge=512 might help for diagrams).
    img_size: int = 384            # bumped from starter's 224; SmolVLM is happy with bigger inputs and ScienceQA images often have small text/diagrams

    # ── Training ─────────────────────────────────────────────────────────────
    epochs: int = 2                 # start small; can extend if val acc still climbing
    learning_rate: float = 2e-4     # standard QLoRA LR
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"

    per_device_batch_size: int = 1  # T4 (16GB) is tight for VLM; stay small
    grad_accum_steps: int = 16      # effective batch = 16

    max_seq_length: int = 1024      # truncate prompt+answer to this; lecture field can be very long
    truncate_lecture_chars: int = 800  # pre-truncate the lecture text before formatting

    # ── Loss masking ─────────────────────────────────────────────────────────
    # We mask out everything except the answer letter token, so all gradient
    # signal goes into "predict the right letter". This matches how we'll
    # score at test time (log-likelihood of A/B/C/...).
    mask_prompt_loss: bool = True

    # ── Evaluation / scoring ─────────────────────────────────────────────────
    eval_every_steps: int = 200
    save_every_steps: int = 200
    seed: int = 42

    # Choice letter set (covers max num_choices we'd see in ScienceQA-style data)
    choice_letters: str = "ABCDEFGHIJ"

    # ── Reproducibility / logging ────────────────────────────────────────────
    log_dir: Path = Path("logs")
    run_name: str = "smolvlm_qlora_r8_all7_v1"


CFG = Config()


# ── Ablation registry ────────────────────────────────────────────────────────
# Each key is a stable, descriptive name for an ablation we ran. The value is
# the set of CFG overrides that exactly reproduces that run. Rerunning is then
# `python train.py --ablation v3_epochs5` (see train.py for wiring).
#
# The corresponding LB scores (with smart-TTA at K=4, skip num_choices<3) are
# documented in RESULTS.md at the repo root.
ABLATIONS: dict[str, dict] = {
    "v1_baseline": {
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 3,
        "img_size": 384,
        "run_name": "smolvlm_qlora_r8_all7_v1",
    },
    "v2_img512": {
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 3,
        "img_size": 512,
        "run_name": "smolvlm_qlora_r8_all7_v2_img512",
    },
    # v3 is our final / best submission (LB 0.81287 with smart-TTA).
    "v3_epochs5": {
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 5,
        "img_size": 384,
        "run_name": "smolvlm_qlora_r8_all7_v3_epochs5",
    },
    "v4_epochs7": {
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 7,
        "img_size": 384,
        "run_name": "smolvlm_qlora_r8_all7_v4_epochs7",
    },
    "v5_r4": {
        "lora_r": 4,
        "lora_alpha": 8,  # alpha = 2r heuristic preserved
        "epochs": 5,
        "img_size": 384,
        "run_name": "smolvlm_qlora_r4_all7_v5_epochs5",
    },
    # v6: same as v3 (r=8, all-7, 384px, 5 epochs) but with DoRA enabled.
    # Tests whether the magnitude-direction decomposition lifts accuracy at
    # equal training duration and equivalent param budget (~4.62M trainable).
    "v6_dora": {
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 5,
        "img_size": 384,
        "lora_use_dora": True,
        "run_name": "smolvlm_qlora_dora_r8_all7_v6_epochs5",
    },
}


def apply_ablation(name: str, cfg: Config = CFG) -> dict:
    """Apply a named ablation profile to the given Config object in-place.

    Args:
        name: profile key from ABLATIONS (e.g., "v3_epochs5").
        cfg: Config object to patch. Defaults to the global CFG singleton.

    Returns:
        The dict of overrides that were applied.

    Raises:
        KeyError: if `name` is not in ABLATIONS.
        AttributeError: if a profile references a CFG field that doesn't exist
            (shouldn't happen with the maintained registry, but catches typos).
    """
    if name not in ABLATIONS:
        raise KeyError(
            f"Unknown ablation '{name}'. "
            f"Registered: {sorted(ABLATIONS.keys())}"
        )
    overrides = ABLATIONS[name]
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(
                f"CFG has no attribute '{key}' (profile '{name}' references it)"
            )
        setattr(cfg, key, value)
    return overrides
