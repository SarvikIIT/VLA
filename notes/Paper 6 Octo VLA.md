___
## The Problem

Prior robot policies train on one robot, one task — no generalization. The goal of Octo is a **generalist robot policy**: pretrain on massive diverse robot data, then finetune cheaply to any new robot.

Challenge: robots differ in sensors (cameras, force-torque, joint encoders), action spaces (end-effector vs joint control), and tasks. A fixed-input architecture can't handle this.

Octo's solution: **modular token-based design** — everything becomes tokens, the transformer is input/output agnostic, and new sensors/action spaces are added by plugging in new tokenizers or heads without touching pretrained weights.

---
## Architecture: Three Components

### 1. Input Tokenizers

Convert all inputs into token sequences before the transformer.

**Language instructions** → T5-base (111M param pretrained LM) → sequence of language embedding tokens → **task tokens** $\mathcal{T}_l$

**Images** → shallow CNN (a few conv layers, not ResNet-scale) → flattened patches → **observation tokens** $\mathcal{T}_o$

The model supports either language OR goal image conditioning. During training, one is randomly zeroed out per example, so the model learns both modalities: Tg is goal output token

$$[\mathcal{T}_l,\, \mathcal{T}_g,\, \mathcal{T}_o]$$

For datasets without language, goal images are always used. For new sensors (e.g., force-torque), add a new tokenizer — transformer weights stay frozen.

---
### 2. Transformer Backbone + Readout Tokens

All tokens are fed into a standard transformer. Attention is **block-wise causally masked**:

- Observation tokens at time $t$ attend to: all task tokens + observation tokens from $t$ or earlier (never future)
- Missing modalities (e.g., dataset has no language) are fully masked out — model handles heterogeneous datasets natively

The transformer produces embeddings:

$$e_l, e_g, e_o = T(\mathcal{T}_l, \mathcal{T}_g, \mathcal{T}_o)$$

**Readout tokens** $\mathcal{T}_{R,t}$ — learned vectors inserted into the sequence:

- They attend to all observation and task tokens before them
- **Nothing attends to them** — they passively accumulate context without affecting other tokens
- Like the `[CLS]` token in BERT — compact vector summary of the full context up to time $t$

The readout token embeddings $e = R(\mathcal{T}_R)$ are what get passed to the action head.

Why readout tokens instead of the last observation token? Flexibility — you can add multiple readout tokens for action chunking, swap action heads, or add new output modalities without retraining the backbone.

---

### 3. Diffusion Action Head

A lightweight 3-layer MLP decodes the readout embeddings into actions using **DDPM** (Denoising Diffusion Probabilistic Models).

Sample Gaussian noise $x^K \sim \mathcal{N}(0, I)$, then denoise $K$ steps:

$$x^{k-1} = \alpha\!\left(x^k - \gamma\,\epsilon_\theta(x^k, e, k) + \mathcal{N}(0, \sigma^2 I)\right)$$

- $\epsilon_\theta$ = denoising MLP conditioned on noisy action $x^k$, readout embedding $e$, step index $k$
- $\alpha, \gamma, \sigma$ from a cosine noise schedule
- $K = 20$ denoising steps during inference

The head predicts an **action chunk** multiple consecutive actions at once same as Paper 2 (action chunking). Octo uses 2 frames of observation history.

**Why diffusion over MSE?**
Robot actions are multimodal multiple valid ways to solve a task. MSE averages them, producing blurry averaged actions. Diffusion represents the full distribution → more decisive, precise actions.

---
## Training Data — Open X-Embodiment

800k robot trajectories from **25 curated datasets** (filtered from ~1.5M total in OXE):

- Removed: datasets without images, without delta end-effector control, too low resolution, too repetitive
- **Double-weight more diverse datasets** — prevents large repetitive datasets from dominating
- **Hindsight goal relabeling** — for datasets without language labels: randomly pick a future frame from the same trajectory as the goal image → all trajectories contribute to goal-conditioned training

Dataset weights proportional to number of samples, with manual adjustments for diversity balance (see pie chart in paper — Kuka, Fractal, Bridge dominate by volume).

---

## Finetuning — Modular Adaptation

This is Octo's main differentiator from prior work.

When adapting to a **new robot** with new sensors or action space:

1. Keep transformer weights **frozen**
2. Add new positional embedding for the new observation type
3. Add new tokenizer (lightweight encoder for new sensor)
4. Add new action head for new action space
5. Finetune with ~100 demonstrations for $<5$ hours on a single A5000 GPU

Because the transformer treats everything as tokens, new inputs/outputs slot in without re-initializing the backbone. Prior architectures fuse encoders into the transformer — changing input means re-initializing half the model.

Finetuning recipe: 50k steps, cosine decay LR schedule with linear warmup, batch size 256, same hyperparameters across all setups.

---

## Model Sizes

| Model | Params | Backbone size |
|---|---|---|
| Octo-Tiny | 10M | — |
| Octo-Small | 27M | ViT-S scale |
| Octo-Base | 93M | ViT-B scale |

Performance scales monotonically with model size on zero-shot tasks.

---

## Results

### Zero-Shot Control (Table in paper, Fig. 5)

Language-conditioned, no finetuning, on WidowX / UR5 / RT-1 robots:

- Octo outperforms **RT-1-X** (35M) by **29% on average**
- Competitive with **RT-2-X** (55 billion params) on WidowX and RT-1 tasks

Goal image conditioning adds +25% vs language on WidowX — goal images give more precise spatial information.

### Finetuning (Table I)

6 unseen setups, ~100 demonstrations each:

| Method | Average Success |
|---|---|
| ResNet+Transformer (scratch) | 20% |
| VC-1 (pretrained visual repr.) | 15% |
| **Octo (ours)** | **72%** |

Octo beats scratch by 52% and beats pretrained visual features by 57%. The pretrained transformer initialization is what matters — it already knows how robots move.

---

## Ablations (Table II) — What Matters Most

All evaluated on WidowX zero-shot, aggregated over 40 trials:

| Configuration | Aggregate Performance |
|---|---|
| **Octo-Small (full system)** | **83%** |
| RT-X data mix (11 datasets) | 60% |
| Single dataset (Bridge only) | 43% |
| Discretized action head | 18% |
| MSE action head | 35% |
| ResNet-50 + Transformer (prior arch) | 70% |

**Three levers that matter:**

1. **Data breadth** — 25 datasets beats 11 beats 1. More diverse pretraining = better generalization
2. **Diffusion head** — crushes MSE (35%) and discretized (18%). Multimodal distributions need diffusion
3. **Architecture** — shallow CNN + large transformer beats ResNet + small transformer

---

## Summary Table

| Component | What it does | Why it's needed |
|---|---|---|
| T5-base tokenizer | Encodes language → task tokens | Off-the-shelf LM, no training needed |
| Shallow CNN tokenizer | Encodes images → observation patches | Keeps params in transformer, not encoder |
| Block-wise causal masking | Obs tokens attend to past + task; missing modalities masked | Handles heterogeneous datasets natively |
| Readout tokens | CLS-like: compress full context, not attended to by others | Flexible output interface without retraining backbone |
| DDPM action head | Denoising diffusion over action chunk | Models multimodal distributions; more decisive than MSE |
| Hindsight goal relabeling | Future frame as goal image for unlabeled data | Unlabeled datasets still contribute to goal conditioning |
| Modular finetuning | New tokenizer + head, frozen transformer | 100 demos sufficient; no backbone re-initialization needed |

---

## Key Takeaways

1. **Generalist policy = token-based modularity** — inputs and outputs are just tokens; swap tokenizers without touching the transformer
2. **Scale of data is the main driver** — 25 datasets beats 11 beats 1; more diversity = better generalization
3. **Readout tokens = flexible output interface** — attend to everything, nothing attends back; like CLS in BERT
4. **Diffusion head is essential** — robot actions are multimodal; MSE produces averaged/blurry actions
5. **Finetuning works because the backbone is frozen** — 100 demos + 5 hours is enough when the transformer already knows robot dynamics
6. **Goal images > language** for precise tasks — more spatial information than natural language descriptions
