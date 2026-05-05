#import "acl-template.typ": acl-paper

#show: acl-paper.with(
  title: [Pixels-to-Predictions: SmolVLM-500M QLoRA Fine-Tuning for ScienceQA],
  authors: (
    (
      name: "Yikai Sun",
      affiliation: "NYU Tandon School of Engineering",
      email: "ys4035@nyu.edu",
    ),
  ),
  abstract: [
    We fine-tune `HuggingFaceTB/SmolVLM-500M-Instruct` for the _Pixels-to-Predictions_ multimodal multiple-choice science QA competition under a strict 5,000,000-trainable-parameter budget and an offline-evaluation rule. We apply QLoRA (4-bit NF4 + double quantization) with rank-8 LoRA across all seven text-decoder linear projections, yielding 4,341,760 trainable parameters (87% of the cap). We score by single-token log-likelihood at the answer-letter position and apply a position-bias-aware test-time augmentation (smart-TTA) that averages logits over $K=4$ random choice permutations while bypassing TTA on 2-choice questions where it empirically degrades accuracy. Our final submission --- a logit-averaged ensemble of two adapters (standard LoRA and DoRA, both 5-epoch r=8) --- achieves *0.81690* on the public leaderboard, $+5.23$ pp over the same model with standard scoring (0.76458). Six single-adapter ablations and one ensemble isolate which axes of capacity matter at this scale: training duration is the dominant lever ($+3.6$ pp from 3 to 5 epochs), but extending to 7 epochs _reduces_ LB by 1.0 pp (overfitting), establishing 5 epochs as near-optimal; doubling LoRA rank from 4 to 8 contributes $+1.8$ pp at matched training duration; higher visual resolution, DoRA, and training-time choice-order augmentation are each statistically tied with baseline.
  ],
)

= Introduction

The competition task is multimodal multiple-choice science QA: each example pairs an image (diagrams, maps, charts, photos) with a question, 2--5 textual answers, and optional `hint` / `lecture` context. The dataset matches the ScienceQA schema.

The dataset's `num_choices` distribution is approximately 21 / 50 / 25 / 4% over $\{2, 3, 4, 5\}$, giving a random-guess baseline of #sym.tilde 32.5%, _not_ 25%. The base SmolVLM-500M, used zero-shot, achieves *41.5%* on a 200-example val sample (binomial SE $approx 3.5$ pp), with a strong "A" bias (71.5% of predictions choose the first option) and 5-choice accuracy below random (18.6%) --- we hypothesize because the letter "E" rarely appears as a multiple-choice option in pretraining text. These two patterns motivate the loss-masked SFT and choice-permutation TTA below.

*Contributions.* (i) A QLoRA fine-tuning pipeline targeting all seven text-decoder projections at rank 8 (4.34M trainable, 87% of cap). (ii) A position-bias-aware smart-TTA that averages over $K=4$ random choice permutations on 3+ choice questions, bypassing TTA on 2-choice. (iii) A data-curve-driven ablation methodology with the headline finding that training duration peaks at 5 epochs and overfits at 7, while five other axes (rank, image res, DoRA, choice-aug, ensemble lift) are statistically tied with baseline. (iv) Diagnosis and resolution of a QLoRA loss-explosion failure on T4 via paged AdamW 8-bit, grad clipping at 0.3, and warmup ratio 0.1.

Code: #link("https://github.com/Palomarr/pixels-to-predictions").

= Methodology

== Architecture and Parameter Budget

`SmolVLM-500M-Instruct` consists of a frozen SigLIP-Base vision encoder (#sym.tilde 93M params), a frozen modality projector, and a SmolLM2-360M text decoder (32 layers, hidden 960, 15 attention heads, 5 KV heads, MLP intermediate 2,560). We adapt only the text decoder via LoRA on all seven linear projections per layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) at $r=8$, $alpha=16$, yielding *4,341,760 trainable parameters* (87% of the 5M cap). Considered alternatives within the cap (attention-only at $r=16$: 3.28M; MLP-only at $r=8$: 2.70M) all sacrifice either MLP or attention adaptation; the QLoRA paper finds low-rank adaptation across all linear layers beats higher-rank attention-only at matched budgets. A regex `target_modules` constrains LoRA to text-decoder layers only --- naive name-based matching would also hit the vision encoder's same-named projections, wasting #sym.tilde 30% of the budget on frozen vision features.

== Quantization: QLoRA

We load the base in 4-bit NF4 with double quantization (#sym.tilde 250 MB VRAM vs #sym.tilde 1.0 GB at bf16); compute dtype is nominally `bf16` but on T4 (sm_75, no native bf16 tensor cores) falls back to emulated kernels. The submission notebook runs offline without `bitsandbytes`, so it loads the base in full `bf16` --- introducing a precision-mismatch gap discussed in @discussion-quantization.

== Prompt Format and Loss Masking

We follow the competition starter notebook's template:

```
<image>
Context:
{lecture, truncated to 800 chars}
{hint}

Question: {question}
Choices:
  A. {choice_0}
  B. {choice_1}
  ...
Answer: {letter — only at training}
```

The `<image>` token marks where the processor splices in visual tokens. We force `do_image_splitting=False` and `longest_edge=384` for exactly 81 visual tokens per image; the processor's default multi-tile splitting up to 1,536 px (#sym.tilde 1,088 visual tokens) would blow past our 1,024-token sequence limit and waste compute on most ScienceQA images, which are $<512$ px natively.

*Loss masking.* During SFT we set `labels` to $-100$ at every token position _except_ the trailing answer-letter token, concentrating 100% of the gradient signal on "predict the right letter" rather than on language-modeling the lecture/question/hint context. Without this, our 4.34M-parameter budget would be largely consumed by a training objective that has no test-time analogue.

== Inference: Log-Likelihood Scoring

For each choice $j in {0, …, "num_choices"-1}$, we compute $"score"[j] = log P("letter"_j | "image", "prompt")$ from the next-token logits at the answer position, where $"letter"_j$ is the single token for the $j$-th letter (`" A"`, `" B"`, ...; the leading space matters under SmolLM2's BPE). Letters beyond `num_choices` are masked to $-infinity$ before argmax. One forward pass per example, fully deterministic --- avoiding the brittleness of generation parsing (variable formats, hallucinated letters).

== Test-Time Augmentation: Smart Choice-Permutation <smart-tta>

Pretrained-baseline analysis revealed strong position bias (A predicted 71% of the time); after fine-tuning, residual bias scales with the number of letter positions involved. For each test example we run $K=4$ forward passes (pass 0 = identity, passes 1--3 = random shuffles), map each pass's letter-position logit back to its original-choice slot via `accum[i, original_choice] += letter_logits[i, shuffled_position]`, and argmax the accumulated logits. Position bias cancels because each original choice appears in every position roughly evenly across passes.

*Smart TTA: skip 2-choice.* $K=4$ TTA hurts 2-choice questions by $-1.23$ pp on val while helping 3-, 4-, 5-choice by $+1.18$, $+2.78$, $+4.55$ pp. With only two unique permutations available at $n=2$, random sampling adds variance without canceling bias. A `tta_skip_below=3` parameter bypasses TTA below threshold, falling back to identity-pass. Smart-TTA yields $+1.91$ pp val and $+1.21$ pp LB on v1 --- both larger than naive always-TTA.

== Training Stability Fixes

Our first training run failed catastrophically: at step 20 the loss spiked to $2.66 times 10^16$, and from steps 40--100 reported `0.000000` (gradient underflow). The pattern --- 16 steps of stable forward+backward (one gradient-accumulation cycle) followed by a single divergent optimizer update --- pointed to three independent fp16 issues that compound under QLoRA.

#figure(
  table(
    columns: 2,
    align: (left, left),
    table.header([*Fix*], [*What it addresses*]),
    [`optim="paged_adamw_8bit"`], [Default `adamw_torch` keeps optimizer states in fp32; under QLoRA + bnb mixed-precision dequant, scale mismatches amplify. The 8-bit paged optimizer is the reference QLoRA recipe.],
    [`max_grad_norm=0.3` (was 1.0)], [Default 1.0 is calibrated for full fine-tuning. With only 4.34M params receiving gradients flowing through 511M frozen-quantized params, per-parameter step sizes are effectively much higher; aggressive clipping is the standard QLoRA practice.],
    [`warmup_ratio=0.1` (was 0.03)], [At 0.03, only #sym.tilde 12 warmup steps preceded peak LR=2e-4; the first optimizer update fired at step 16 with LR essentially at peak. 0.1 gives #sym.tilde 58 warmup steps, so by step 16 LR is #sym.tilde 40% of peak.],
  ),
  caption: [QLoRA stability fixes that turned a divergent first run into reproducible training.],
) <stability-fixes>

After applying all three fixes, training was stable across all subsequent runs.

= Experimental Setup <experimental-setup>

Train 3,109 / val 1,048 / test 1,008 (labels held out). At effective batch 16 and 3 epochs, only #sym.tilde 580 optimizer steps. Paged AdamW 8-bit at LR $2 times 10^(-4)$, cosine decay with 10% warmup, max grad norm 0.3, weight decay 0. Effective batch 16 (per-device 1 $times$ gradient accumulation 16); nominal `bf16` (T4 fallback). `max_seq_length=1024`, lectures pre-truncated to 800 characters, images forced to a single 384$times$384 tile (#sym.tilde 81 visual tokens). Single random seed (42). NVIDIA T4 (16 GB) on Kaggle Free Tier; 3-epoch run #sym.tilde 2:44, 5-epoch #sym.tilde 4:50, 7-epoch #sym.tilde 6:30.

= Main Result <results-main>

Our best submission applies the v3 adapter ($r=8$, 384 px, 5 epochs) with smart-TTA at $K=4$:

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([*Configuration*], [*Public LB*]),
    [Random baseline (analytic)], [#sym.tilde 0.325],
    [Pretrained, no fine-tune (200-ex val proxy)], [#sym.tilde 0.415],
    [v1 fine-tuned, no TTA], [0.76458],
    [v1 fine-tuned + smart-TTA], [0.77665],
    [v2 (img 512) + smart-TTA], [0.77867],
    [v5 ($r=4$, 5 epochs) + smart-TTA], [0.79476],
    [v4 (7 epochs) + smart-TTA], [0.80281],
    [v3 ($r=8$, 5 epochs) + smart-TTA], [0.81287],
    [v6 (DoRA, 5 epochs) + smart-TTA], [0.81488],
    [*v3 + v6 ensemble + smart-TTA*], [*0.81690*],
  ),
  caption: [Public leaderboard scores. The pretrained-baseline row is a 200-example val proxy and not directly comparable to the LB rows (1,008-example hidden test); it anchors the qualitative gap from no-fine-tune to fine-tuned.],
) <main-result>

Net lift from training-side improvements (v1 #sym.arrow v6): $+5.03$ pp. Net lift from inference-only smart-TTA on v1: $+1.21$ pp. The v3+v6 ensemble adds another $+0.20$ pp (within one standard error of v6 alone) and is selected as the final submission as the highest measured LB.

= Ablations

We vary training duration, LoRA rank, image resolution, adapter variant (LoRA vs DoRA), training-time choice-order augmentation, TTA strategy, and adapter ensembling. All runs use a single random seed; with $n=1{,}008$ test examples and accuracies in 0.77--0.81, the binomial standard error of any single-run LB estimate is $approx 1.3$ pp, so we treat deltas $<2$ pp as suggestive rather than significant.

== Headline Ablations Table

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    table.header([*ID*], [*Config*], [*Train-val*], [*LB*]),
    [v1], [$r=8$, all-7, 384 px, 3 epochs (baseline)], [0.7634], [0.77665],
    [v2], [+ img_size=512], [0.7586], [0.77867],
    [v3], [+ epochs=5], [0.8015], [0.81287],
    [v4], [+ epochs=7], [0.7996], [0.80281],
    [v5], [$r=4$, $alpha=8$, 5 epochs], [0.7729], [0.79476],
    [v6], [DoRA, $r=8$, 5 epochs], [0.7948], [0.81488],
    [v7], [+ choice-order training aug], [0.7634], [n/s],
    [*ens.*], [*v3 + v6 logit-averaged*], [*---*], [*0.81690*],
  ),
  caption: [Train-val is 4-bit eval (training-time); LB uses bf16 + smart-TTA. v1 LB without TTA: 0.76458. v6 trainable params: 4.62M (vs 4.34M for v1--v5, v7). v7 not submitted ("n/s") because train-val identity to v1 predicts LB $approx$ 0.78, well below the v3+v6 ensemble.],
) <ablation-table>

== TTA Strategy (No Retraining)

On v1's val ($n=1048$): no TTA = 0.7290, always-TTA ($K=4$) = 0.7405 ($+1.15$ pp), *smart-TTA* ($K=4$, skip 2-choice) = *0.7481* ($+1.91$ pp). Per-bucket smart-TTA: 2-choice 0.7992 (bypassed, identical to no-TTA), 3-choice 0.7106 ($+1.18$ pp), 4-choice 0.8492 ($+2.78$ pp), 5-choice 0.3182 ($+4.55$ pp). Position bias is largest where invalid letter slots are most numerous --- TTA's logit averaging cancels exactly that bias. Smart-TTA dominates, used for all submissions from v1+sTTA onward.

== Image Resolution (v2)

The hypothesis was that ScienceQA's diagram-heavy questions would benefit from higher visual token resolution. Empirically, train-val _fell_ by 0.48 pp ($0.7634 #sym.arrow 0.7586$) while LB _rose_ by 0.20 pp ($0.77665 #sym.arrow 0.77867$); both deltas within one standard error. We report this as *statistically tied with baseline*. Plausible reasons: ScienceQA images are mostly $<512$ px natively (upsampling adds no information); SmolVLM was pretrained at specific image scales (512-px tiles are mildly OOD); at our parameter budget, visual fidelity is not the binding constraint.

== LoRA Rank (v5) and Decomposition

The capacity-halving control (v5: $r=4$, $alpha=8$, 5 epochs, #sym.tilde 2.17M trainable params) yields *0.79476* LB. Crossing v1, v3, v5 isolates two axes at fixed everything-else: $r=4 #sym.arrow r=8$ at epochs=5 contributes $+1.81$ pp (v5#sym.arrow v3); 3 #sym.arrow 5 epochs at $r=8$ contributes $+3.62$ pp (v1#sym.arrow v3). Training duration is roughly $2 times$ the lever of rank, indicating fitting time is a stronger constraint than fitting capacity at our parameter budget and dataset size.

== Training Duration: The Convergence Curve <training-duration>

The single most impactful axis. Three measurements at epochs $in {3, 5, 7}$ trace the generalization curve cleanly:

#figure(
  table(
    columns: 4,
    align: (right, left, right, right),
    table.header([*Epochs*], [*Run*], [*Train-val*], [*LB*]),
    [3], [v1], [0.7634], [0.77665],
    [*5*], [*v3 (best)*], [*0.8015*], [*0.81287*],
    [7], [v4], [0.7996], [0.80281],
  ),
  caption: [Convergence curve at $r=8$. Both metrics peak at 5 epochs and decline at 7.],
) <convergence>

The v3 ablation was motivated by v1's training loss --- _still descending_ at step 580 (final epoch-3 step, loss 0.2535; descent: 1.386 #sym.arrow 0.674 #sym.arrow 0.519 #sym.arrow 0.323 #sym.arrow 0.254) with no plateau --- which we interpreted as under-training (9,300 example-views was insufficient). v3's $+3.62$ pp LB lift confirmed this. The v4 result (5 #sym.arrow 7 epochs) is a textbook overfitting signature: both metrics decline, asymmetrically ($-0.19$ pp on train-val vs $-1.01$ pp on LB), consistent with fitting training-specific patterns that transfer to similarly-distributed val but not held-out test. Across the explored range, training duration affects LB by #sym.tilde 5 pp --- larger than rank (#sym.tilde 1.8 pp) or visual resolution (#sym.tilde 0 pp).

== DoRA: Adapter-Variant Control (v6)

DoRA (Liu et al., 2024) replaces LoRA's additive update with a magnitude-direction decomposition $W' = m dot.c (W + Delta W) \/ ||W + Delta W||$, adding one learned magnitude scalar per output channel ($+277{,}480$ params; $4.62$M total trainable). v6 (DoRA on the v3 config) yields *0.7948* train-val and *0.81488* LB --- statistically tied with v3 (Δ within one SE). DoRA produced no measurable lift over standard LoRA at this dataset and parameter scale; the val-LB inversion (train-val below v3, LB above) mirrors v1#sym.arrow v2 and is discussed in @val-lb-flips.

== Choice-Order Training Augmentation (v7)

v7 (v3 config + per-`__getitem__` random permutation of choices, with answer-letter remapped) yields train-val *0.7634* --- numerically identical to v1's 0.7634, despite v7 training for 67% more steps. The augmentation transformed the SFT objective from position-conditioned ("for this image+question, predict letter B") to content-conditioned ("predict the letter whose choice text matches the question"); at our 4.34M-parameter budget on 3,109 training examples, the harder content-conditioned objective regressed to v1's plateau. *Position bias is therefore better addressed at test time via smart-TTA than at training time via augmentation*, at this scale. We did not submit v7 to the leaderboard.

== Adapter Ensembling

A logit-averaged ensemble of v3 and v6 yields *0.81690* LB --- $+0.20$ pp over v6's 0.81488. The two adapters share architecture, training data, and seed, differing only by adapter variant (standard LoRA vs DoRA), so error correlation is high; the ensemble's modest lift is consistent with that prior. The ensemble is our final submission.

= Discussion

== The 4-bit-vs-bf16 Inference Gap <discussion-quantization>

Training-time val uses the 4-bit base; inference val uses bf16 (no `bitsandbytes` available offline). On v1 we measured this gap explicitly: train-val (4-bit) 0.7634 vs inference val (bf16, no TTA) 0.7290 --- a *3.44 pp drop* from precision mismatch, larger than the $<1$ pp typically reported in the QLoRA paper. Our LoRA-on-all-7-projections config (including high-magnitude MLP weights) is plausibly more sensitive to dequantization noise than attention-only. Bundling `bitsandbytes` wheels to enable 4-bit inference was disproportionate engineering risk for the expected $1$--$3$ pp lift.

== Val-vs-Leaderboard Ranking Flips <val-lb-flips>

Two val-LB ranking inversions of similar magnitude were observed: v1#sym.arrow v2 (train-val $-0.48$ pp, LB $+0.20$ pp) and v3#sym.arrow v6 (train-val $-0.67$ pp, LB $+0.20$ pp). Each individual delta is within one SE, but the pattern repeating across two independently-trained pairs suggests either (i) val and test are not perfectly i.i.d. by per-example difficulty, so configs handling hard examples differently rank-flip; or (ii) the 4-bit#sym.arrow bf16 transition is non-uniform, with adapter variants (especially DoRA's magnitude scalars) absorbing dequantization noise differently. We cannot disambiguate from single-seed measurements.

= Reproducibility

Each of the seven reported runs (v1--v7) is registered as a named profile in `config.ABLATIONS`; one command reproduces any row of @ablation-table: `python train.py --ablation v3_epochs5`. Each completed run writes `val_results.json` with resolved config and val accuracy, pairing recipe with measurement. Submission scoring at `make_submission.py --tta_k 4 --tta_skip_below 3 --adapter_path A B` reproduces the smart-TTA + ensemble pipeline of our final submission. The repo's README documents the competition-data staging, transformers-v5 import fallback, CUDA-state cleanup, and QLoRA stability fixes (@stability-fixes) handled in code.

= Conclusion

Under a fixed model, a 5M-trainable-parameter cap, and offline evaluation, we achieved *0.81690* on the public leaderboard via a logit-averaged ensemble of two QLoRA adapters trained at rank 8 across all seven text-decoder linear projections for 5 epochs (one with standard LoRA, one with DoRA), augmented with smart choice-permutation TTA. Training duration was the dominant lever by a wide margin (3 #sym.arrow 5: $+3.62$ pp; 5 #sym.arrow 7: $-1.01$ pp), bounding the optimum near 5 epochs. Higher visual resolution was statistically tied with baseline; halving the rank cost 1.81 pp; DoRA contributed $+0.20$ pp; choice-order training augmentation regressed to baseline. Across all five non-training axes we tested, no single architectural or augmentation lever moved LB by more than 2 pp, indicating that fitting time on the small training set --- not capacity, adapter sophistication, or position-bias mitigation at training time --- was the binding constraint. The 4-bit-vs-`bf16` precision gap forced by the offline-eval rule accounts for #sym.tilde 3.4 pp of headroom otherwise available. All measurements use a single random seed (SE $approx 1.3$ pp).