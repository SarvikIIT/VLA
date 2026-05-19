___
# Paper 8 — π₀ (Pi-Zero): A Vision-Language-Action Flow Model

> **Directly relevant to my project (blocking gate).** See the "Relation to My Project" section at the bottom — both verification questions resolved. Project survives intact; Pi-0 is the reference implementation for Phase 1.

---

## The Problem

Generalist robot foundation models need three things, all hard at once:
1. **Scale** — benefits of large pretraining only appear at scale
2. **Architecture** — must represent intricate, high-frequency dexterous behaviors
3. **Training recipe** — pretrain/post-train data curation, like LLMs

Prior VLAs (OpenVLA, RT-2) use **autoregressive discretization** — actions as discrete text-like tokens. This blocks high-frequency control and action chunking. π₀'s answer: a pretrained VLM backbone + a **flow-matching action expert** for continuous, high-frequency (up to 50 Hz) action chunks.

---

## Architecture — Single Transformer, Two Experts (Mixture of Experts)

$$\pi_0 = \underbrace{\text{PaliGemma VLM}}_{\text{from internet pretrain}} \;+\; \underbrace{\text{Action Expert}}_{\text{from scratch, }\sim300\text{M}}$$

One transformer, **two sets of weights** ("experts"). Tokens are routed to an expert; experts interact **only through the shared self-attention layers**.

| Expert                     | Handles                                               | Source         | Width | mlp_dim |
| -------------------------- | ----------------------------------------------------- | -------------- | ----- | ------- |
| VLM (Gemma 2B / PaliGemma) | images $\mathbf I_t^{1..n}$, language $\ell_t$        | PaliGemma init | 2048  | 16384   |
| Action expert              | state $\mathbf q_t$, noisy actions $\mathbf A_t^\tau$ | scratch        | 1024  | 4096    |

- PaliGemma = SigLIP (400M vision) + Gemma (2.6B language). Late-fusion VLM: image tokens embedded into the language token space.
- Action expert deliberately **downsized** (width 1024) for real-time inference → ~300M params. Total ≈ 3.3B.
- Design inspired by **Transfusion** — one transformer, multiple objectives (flow-matching loss on continuous action tokens; cross-entropy on discrete tokens).

**Inputs:** $\mathbf o_t = [\mathbf I_t^1, \ldots, \mathbf I_t^n, \ell_t, \mathbf q_t]$ — multiple RGB images (2–3 per robot), language tokens, joint-angle state. Action chunk $\mathbf A_t = [\mathbf a_t, \ldots, \mathbf a_{t+H-1}]$, **H = 50**.

---

## Flow Matching — Training & Inference

### Training (Conditional Flow Matching)
- Sample noise $\boldsymbol\epsilon \sim \mathcal N(0, I)$
- Noisy action (linear-Gaussian / optimal-transport path): $\mathbf A_t^\tau = \tau \mathbf A_t + (1-\tau)\boldsymbol\epsilon$
- Target vector field: $u(\mathbf A_t^\tau \mid \mathbf A_t) = \mathbf A_t - \boldsymbol\epsilon$ — velocity points from noise toward clean action (same convention as KAN-We-Flow: target = action − noise)
- Loss:

$$L^\tau(\theta) = \mathbb E_{p(\mathbf A_t \mid \mathbf o_t),\, q(\mathbf A_t^\tau \mid \mathbf A_t)}\;\big\lVert\, v_\theta(\mathbf A_t^\tau, \mathbf o_t) - (\mathbf A_t - \boldsymbol\epsilon)\,\big\rVert^2$$

- $\tau$ sampled from a **shifted Beta distribution** $p(\tau)=\text{Beta}\!\big(\tfrac{s-\tau}{s};\,1.5,\,1\big)$, $s=0.999$ — emphasizes **low (noisier)** timesteps. Rationale: predicting an action from a very informative observation $\mathbf o_t$ is a *harder* problem than denoising at low noise, so weight the hard (high-noise) regime more. (Image synthesis papers do the opposite.)

### Inference
- Start from pure noise $\mathbf A_t^0 \sim \mathcal N(0, I)$
- Forward Euler integrate: $\mathbf A_t^{\tau+\delta} = \mathbf A_t^\tau + \delta\, v_\theta(\mathbf A_t^\tau, \mathbf o_t)$
- **10 integration steps** ($\delta = 0.1$)

### ⚠️ Vanilla flow matching ≠ one step
| Method | Action decoding | Steps |
|---|---|---|
| Octo | DDPM diffusion | 20 |
| **π₀** | **vanilla flow matching** | **10** |
| KAN-We-Flow | consistency flow matching (+MFM, +ACR) | **1** |

The "one Euler step" property comes from **consistency** flow matching specifically, *not* vanilla flow matching. A vanilla learned velocity field isn't straight enough for a single step. This spectrum is a real design axis for any "put flow matching on X" project.

---

## Attention Mask — Block-wise Causal, 3 Blocks

Blocks: $[\mathbf I_t^{1..n}, \ell_t]\;\to\;[\mathbf q_t]\;\to\;[\mathbf a_t^\tau, \ldots, \mathbf a_{t+H-1}^\tau]$

- **Full bidirectional** attention *within* each block
- A block **cannot attend to future blocks**
- Block 1 (VLM inputs) can't attend forward → minimizes distribution shift from PaliGemma pretraining
- **Robot state $\mathbf q_t$ is its own block** — it does *not* change across the 10 flow integration steps, so its keys/values are **cached** once and reused. Only the action block is recomputed each integration step. Major inference saving.

**τ conditioning:** noisy action mapped to embedding via MLP $W_3\cdot\text{swish}(W_2\cdot\text{concat}(W_1 \mathbf a_t^\tau,\, \phi(\tau)))$, where $\phi$ = sinusoidal positional encoding of the flow timestep.

---

## Inference Time (RTX 4090, 3 cameras)

| Part | Time |
|---|---|
| Image encoders | 14 ms |
| Observation forward pass | 32 ms |
| 10× action forward pass (flow) | 27 ms |
| Network latency (if off-board) | 13 ms |
| **Total on-board** | **73 ms** |
| Total off-board | 86 ms |

Runs inference every 0.5–0.8 s (after executing 16–25 actions of the chunk) → up to 50 Hz control.

---

## Training Recipe — Pretrain / Post-train (the paper's real thesis)

| Phase | Data | Purpose |
|---|---|---|
| Pre-training | huge diverse mixture (own dexterous data + all OXE), lower quality | broad capability + **learn to recover from mistakes** (mistakes rarely in clean data) |
| Post-training | small high-quality curated, task-specific | fluent, confident execution |

Pretrain-only → brittle, no recovery. High-quality-only → doesn't know recoveries. Need both. Directly analogous to LLM pretrain → instruction-tune.

---

## Data

- Own dexterous dataset: **7 robot configurations, 68 tasks** (903M timesteps; 106M single-arm, 797M dual-arm) + entire **OXE** (22 robots). ~10,000 hours. 9.1% open-source, rest in-house.
- "Task" = combination of objects/behaviors (broader than the noun-verb "tasks" of prior work).
- Task-robot combo re-weighted by $n^{0.43}$ ($n$ = #samples) to avoid over-represented combos dominating.
- Config/action vectors zero-padded to 18 dims (largest robot); missing camera slots masked.
- Robots: UR5e, Bimanual UR5e, Franka, Bimanual Trossen (ALOHA), Bimanual ARX/AgileX, Mobile Trossen/ARX, Mobile Fibocom — single, dual-arm, and mobile manipulators.

---

## Results

- **Zero-shot** (shirt folding, bussing easy/hard, grocery bagging, toast): π₀ beats OpenVLA and Octo by a large margin. A compute-matched "parity" π₀ (160k steps) still beats all baselines. Even π₀-small (no VLM) beats OpenVLA and Octo.
- OpenVLA weak here — autoregressive discretization, **no action chunking**. Octo has chunks but limited representational capacity.
- **Language following:** π₀ ≫ π₀-small → VLM pretraining is what drives instruction following; also benefits from a high-level VLM policy decomposing tasks (SayCan-style).
- **Finetuning new dexterous tasks** (stack bowls, towel fold, tupperware-in-microwave, paper-towel replace, items-in-drawer): π₀ generally best; pretraining → up to **2×** over from-scratch. Strongest *prior* baselines are from-scratch ACT/Diffusion Policy — earlier methods underuse pretraining.
- **Complex multi-stage** (laundry folding, mobile laundry, box building, packing eggs/food): 5–20 min tasks, full pretrain+posttrain wins across the board. Longest dexterous end-to-end tasks in the literature.

---

## Limitations (paper's own)

- No principled theory of *what* pretraining data to include / how to weight it — open problem.
- Reliability varies by task; unclear how much/what data is needed for near-perfect performance.
- Positive transfer across very distinct domains (navigation, driving, legged) untested.

---

## Key Takeaways

1. **VLM backbone + separate flow-matching action expert** = high-frequency (50 Hz) dexterous continuous control — beats autoregressive-discretization VLAs decisively.
2. **Mixture-of-experts via shared attention** — big pretrained expert + small from-scratch action expert; same "freeze big, train small" philosophy as Octo/HAMLET.
3. **Vanilla flow matching is multi-step (10 here)** — one-step needs *consistency* FM, not vanilla. A real design axis.
4. **State-as-its-own-block enables KV caching** across integration steps — key inference trick.
5. **Recipe is the thesis** — diverse pretrain (recovery behaviors) + curated post-train (fluent execution), exactly mirroring LLMs.

---

## ★ Relation to My Project (Phase 1 / Phase 2)

**Q1 — Does Pi-0 do a controlled head-swap ablation (diffusion vs flow, same backbone)?**
**NO.** Pi-0 compares *whole systems* (π₀ vs Octo vs OpenVLA — different models). Its only architectural ablation is π₀ vs π₀-small (VLM-init vs no-VLM-init = *pretraining*, not head type). → **Phase 1's controlled "hold Octo fixed, swap only the head" comparison is still novel.**

**Q2 — Vanilla or adaptive flow matching?**
**VANILLA.** Standard CFM + a custom Beta τ-sampling. No MFM segmentation, no consistency loss, no ACR, no adaptive $K$, no adaptive loss weight. Fixed 10 steps. → **Phase 2 (adaptive MFM / ACR) is still novel.**

**Consequences:**
- **Framing tweak:** drop any "flow matching is new for VLAs" claim — Pi-0 owns that. Sharpen to: *isolated controlled diffusion-vs-flow comparison on a small open generalist policy* (P1) + *adaptive flow matching* (P2). Pi-0 *strengthens* the premise (strong evidence FM works on VLAs) while leaving these specific questions open.
- **Pi-0 = reference implementation.** Appendix B (action-expert arch, τ-conditioning MLP, 3-block mask, KV-cache trick) and the inference-time table are the blueprint for building the flow head on Octo.
- **New explicit design decision for Phase 1:** vanilla flow matching (Pi-0-style, ~10 steps, simpler) vs consistency flow matching (KAN-We-Flow-style, 1 step, harder training). This choice is itself part of the contribution and should be stated up front.
