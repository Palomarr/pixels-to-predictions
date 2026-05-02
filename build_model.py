"""
Model loading and LoRA setup.

This is the place where the 5M-trainable-parameter constraint actually gets
enforced. We do three things:
  1. Load SmolVLM-500M-Instruct in 4-bit (NF4) using bitsandbytes.
  2. Prepare it for k-bit training (handles input embed gradient flow,
     output dtypes, etc.).
  3. Wrap with PEFT LoraConfig that targets ONLY the LM-side projections
     (regex-restricted so the vision encoder stays frozen).

Critical: ALWAYS call print_trainable_parameters() and verify the count is
~4.34M before launching training. If it's wildly different, something is
matching/missing — fix it here, not 2 hours into training.
"""
from __future__ import annotations
import re

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)
# transformers v5+ removed AutoModelForVision2Seq in favor of AutoModelForImageTextToText.
# We pin 4.57.6 during training (where both names exist) but the offline submission
# environment may have v5+ where only the new name is available.
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import CFG


def get_compute_dtype() -> torch.dtype:
    """Use bf16 if the GPU supports it (Ampere+), else fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_processor():
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Right-padding is required for causal LMs in training; processor sometimes
    # defaults differently for VLMs, so set it explicitly.
    processor.tokenizer.padding_side = "right"

    # IMPORTANT: by default SmolVLM's processor splits images into multiple
    # 384x384 tiles (longest_edge=4*384=1536 → up to ~1088 visual tokens per
    # image). For ScienceQA this is overkill — most images are <512px — and
    # it makes prompts blow past max_length and trigger truncation that
    # corrupts the visual-token block.
    #
    # Force a single tile per image. Tile size is CFG.img_size — bumping
    # this from 384 to 512 for the ablation roughly doubles visual tokens
    # (81 → ~144) and helps for diagram/small-text questions.
    processor.image_processor.size = {"longest_edge": CFG.img_size}
    if hasattr(processor.image_processor, "do_image_splitting"):
        processor.image_processor.do_image_splitting = False
    return processor


def build_model(for_training: bool = True):
    """Load model in 4-bit and wrap with LoRA."""
    compute_dtype = get_compute_dtype()

    bnb_config = None
    if CFG.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=CFG.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=CFG.bnb_4bit_use_double_quant,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        CFG.model_id,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if not for_training:
        model.eval()
        return model

    # Prepare for k-bit training: enables input embed gradient, casts norms
    # to fp32, etc. Required for QLoRA to actually train.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # Discover the actual matched module names BEFORE building LoraConfig so
    # we can sanity-check. This is the single biggest gotcha — see HW2 lesson
    # about replacing `cls_logits` instead of the whole head.
    pattern = re.compile(CFG.lora_target_regex)
    matched = [name for name, _ in model.named_modules() if pattern.match(name)]
    print(f"[LoRA] regex matched {len(matched)} modules. Sample:")
    for n in matched[:5]:
        print(f"    {n}")
    if len(matched) == 0:
        raise RuntimeError(
            "LoRA regex matched no modules. Inspect model.named_modules() and "
            "fix CFG.lora_target_regex. The text decoder might be at a different path."
        )

    lora_config = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CFG.lora_target_regex,  # PEFT accepts regex strings
    )
    model = get_peft_model(model, lora_config)

    # CRITICAL: verify trainable count
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LoRA] trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)")
    if n_trainable > 5_000_000:
        raise RuntimeError(
            f"Trainable param count {n_trainable:,} exceeds the 5M cap. "
            f"Reduce lora_r or shrink target_modules."
        )

    return model


def discover_text_module_path(model_id: str = CFG.model_id) -> None:
    """Diagnostic helper: dump all linear-ish module names so you can confirm
    the path to the LM and adjust the regex if needed. Run this once when
    setting up a new model.

    Usage:
        python -c "from build_model import discover_text_module_path; discover_text_module_path()"
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )
    print("All Linear modules:")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f"  {name}  shape=({mod.in_features},{mod.out_features})")
