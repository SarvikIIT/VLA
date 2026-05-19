___
# Phase 1 — Study & Execution Roadmap

**Goal of Phase 1:** Replace Octo's 20-step DDPM diffusion action head with a 1-step flow-matching head; benchmark latency + success rate vs the diffusion baseline.

This file is the *how-to-prepare* guide. The project decision itself lives in `Research Project Recommendations.md`.

---

## 1. What You Already Have (do NOT re-study)

- **Octo architecture** — tokenizers, block-wise causal masking, readout tokens, diffusion head. See `Paper 6 Octo VLA.md`.
- **Flow matching (inference side)** — straight-line path, one Euler step $\hat a = a_t + (1-t)\,v_\theta$. See `Paper 5 KAN We Flow.md`.
- **Why diffusion vs flow matching** — multimodality, 20 steps vs 1 step.

Conceptually you're ~60% there. The remaining 40% is **engineering + the flow-matching training loss**.

---

## 2. Knowledge Gaps to Close (ordered by priority)

### G1 — JAX + Flax basics  ⭐ biggest practical gap
Octo is written in **JAX/Flax**, not PyTorch. You cannot touch the code without this.

Study:
- JAX core: `jit`, `grad`, `vmap`, pure functions, why no in-place mutation, PRNG keys (explicit randomness — matters for noise sampling in your head)
- Flax: `nn.Module`, `setup` vs `@nn.compact`, how parameters are initialized and passed, `TrainState`
- Optax: how the optimizer/LR schedule is defined (Octo uses AdamW + cosine decay + warmup)

**Ready when:** you can write a tiny Flax MLP, train it on toy data with an Optax optimizer, and explain how PRNG keys produce the Gaussian noise sample.

### G2 — Flow-matching TRAINING objective  ⭐ the thing you implement
You know the *inference* step. You need the *loss*. Conditional Flow Matching:

- Sample $t \sim U[0,1]$, source noise $a_{\text{src}} \sim \mathcal N(0,I)$, target $a_{\text{tar}}$ = expert action
- Interpolate: $a_t = (1-t)\,a_{\text{src}} + t\,a_{\text{tar}}$
- Target velocity for the straight-line path: $u = a_{\text{tar}} - a_{\text{src}}$
- Loss: $\mathcal L = \mathbb E\big[\,\lVert v_\theta(a_t, t, e) - u \rVert_2^2\,\big]$, where $e$ = Octo readout embedding (the conditioning)

**Ready when:** you can derive on paper why regressing $v_\theta$ onto $a_{\text{tar}}-a_{\text{src}}$ makes one Euler step land on the action, and you've coded this loss for a 2D toy distribution.

### G3 — The DDPM head you're replacing
Read Diffusion Policy (Chi et al., 2023) — Octo's diffusion head is built on it. Understand: noise prediction $\epsilon_\theta$, the cosine schedule, the 20-step sampling loop, conditioning on the readout embedding. You need to know exactly what the *input/output interface* of the head is so the flow-matching head is a drop-in replacement with the same interface.

**Ready when:** you can point to the exact tensor in/out shapes of Octo's action head module.

### G4 — The Octo codebase
Official project page (from the paper): **octo-models.github.io** → GitHub repo (JAX).

Map these before changing anything:
- Where the **action head module** is defined (the diffusion head class)
- How it's **registered/plugged** into the transformer output (readout embedding → head)
- The **finetuning script** and its config (the ~100-demo, 50k-step recipe)
- How **pretrained checkpoints** (Octo-Small 27M) are loaded
- The **loss** wiring — where the DDPM loss is computed, so you know where to substitute the CFM loss

**Ready when:** you can run the provided finetuning example unmodified on Octo-Small and get it to step.

### G5 — Evaluation harness  ⭐ the hidden time-sink
No real robots → simulation is your only benchmark. The eval setup is usually the hardest part, not the model.

Decide early:
- Which sim benchmark you can actually install and run (identify the one Octo's repo supports out-of-the-box; do NOT pick one requiring hardware)
- What "success rate" means there, how many trials, how it's scored
- How to measure **inference latency** cleanly (per-action wall-clock, warm cache, exclude env stepping)

**Ready when:** you can run the diffusion baseline through the sim eval end-to-end and get a number.

---

## 3. Study Sequence (ordered stages, not a calendar)

| Stage | Focus | Output |
|---|---|---|
| S1 | JAX + Flax + Optax (G1) | toy Flax MLP trained |
| S2 | Flow-matching loss (G2) | CFM loss on 2D toy, plots match target |
| S3 | Diffusion Policy paper + Octo head code (G3, part G4) | written note: head I/O interface |
| S4 | Octo repo run unmodified (G4) | finetune example steps on Octo-Small |
| S5 | Eval harness (G5) | diffusion baseline number reproduced |

Do S1 and S2 in parallel with re-reading your Paper 5/6 notes. S4–S5 are the real gate — most time goes here.

---

## 4. Phase 1 Execution Milestones (after study)

- **M0 — Baseline:** reproduce one Octo-Small benchmark number with the stock diffusion head. *(Nothing changed yet — this is the control.)*
- **M1 — Head module:** implement a Flax flow-matching head with the **same input/output interface** as the diffusion head (in: readout embedding $e$, noisy action, $t$; out: velocity $v$).
- **M2 — Train:** swap the loss to CFM, finetune only the new head (~100 demos, transformer frozen), confirm it converges.
- **M3 — Benchmark:** diffusion vs flow-matching head, same backbone, same eval — table of **latency per action** + **success rate**.
- **M4 — Robustness:** if 1-step accuracy drops, sweep 1→2→4 Euler steps; report the steps-vs-accuracy curve.

**Phase 1 is DONE at M3.** M4 strengthens it. Phase 2 (adaptive) only starts after M3/M4.

---

## 5. "Ready to Start Coding" Gate

Tick all before writing project code:
- [ ] Can write & train a small Flax model (G1)
- [ ] Can implement CFM loss on a toy problem (G2)
- [ ] Can state Octo's action-head I/O shapes exactly (G3/G4)
- [ ] Octo finetune example runs unmodified on Octo-Small (G4)
- [ ] Diffusion baseline produces a sim eval number (G5)
- [x] **Pi-0 read & gate resolved** — confirmed NOT a controlled head-swap, confirmed vanilla (non-adaptive) FM. Phase 1 + Phase 2 both still novel. Framing tweaked (drop "FM new for VLAs"). See `Paper 8 Pi-0 VLA.md`.
- [ ] Decide Phase 1 FM variant: **vanilla** (Pi-0-style, ~10 steps) vs **consistency** (KAN-We-Flow-style, 1 step). This choice is part of the contribution — fix it before M1.
- [ ] OpenVLA (7) + VLA-R1 (9) — non-blocking, read after Phase 1

---

## 6. Timeline Estimate (calibrated to CF ~1800 background)

Strong algorithmic/coding background → study & implementation collapse toward the optimistic end. Infra (S4/S5) and convergence (M2) do **not** collapse — they're not algorithmic-skill-bound. Assumes ~full-time focus.

| Stage | Estimate | Risk |
|---|---|---|
| S1 JAX/Flax | 2–4 d | low (fast learner) |
| S2 flow-matching loss (toy) | 1–3 d | low (math is the strength) |
| S3 Diffusion Policy + Octo head code | ~2 d | low |
| S4 Octo repo running unmodified | 2–4 d | **high — infra/CUDA luck, not skill** |
| S5 sim eval → baseline number | 3–7 d | **high — the wildcard** |
| M1 implement flow head (Pi-0 App. B blueprint) | 3–4 d | low |
| M2 train / convergence | 4–8 d | **high — empirical ML intuition, least CF-helped** |
| M3 benchmark diffusion vs flow | 2–3 d | low |
| M4 robustness sweep (optional) | 2–3 d | low |

- **Phase 1 total:** ~3–6 weeks, midpoint ~4–4.5.
- **Phase 2 (if Phase 1 lands well):** +3–5 weeks, reusing infra. Plausible within a summer, not guaranteed.

**Leverage move:** front-load S4 + S5. They are the highest-variance risk and independent of coding strength — kill that uncertainty *first*, before polishing anything. Everything else is downhill for this profile.

---

## 7. Resources (find by name — verify before relying)

- **Octo** — official project page in the paper: octo-models.github.io → linked GitHub (JAX). Primary source of truth for code.
- **Flow Matching for Generative Modeling** — Lipman et al., 2022. The training objective and theory behind G2.
- **Diffusion Policy** — Chi et al., 2023. The head you're replacing (G3); cited directly by Octo.
- **KAN-We-Flow / FlowPolicy / MP1** — prior flow-matching action heads. Reference implementations for the head; also relevant to the Phase 1 novelty caveat.
- **JAX & Flax docs** — official documentation for G1 (the largest gap if your background is PyTorch).

> Note: confirm the Octo repo's currently-supported sim eval before committing to a benchmark — repos drift, and a memory/notes pointer is not a guarantee it still works. Verify live.
