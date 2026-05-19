___
# Summer Research Project — Chosen Plan

**Decision:** A single staged project, not 6 separate ideas.

- **Phase 1 (primary, guaranteed deliverable):** Replace Octo's diffusion action head with a flow-matching head.
- **Phase 2 (stretch, only if time permits → the paper):** Make that flow matching *adaptive* (adaptive MFM segments / ACR weight).

Phase 2 is a *modification of Phase 1's codebase*, not a new project. Phase 1 finishes and stands alone even if Phase 2 never starts.

Methodology bucket: **Option 2** (make an existing VLA work better) for Phase 1, shading into **Option 3** (better training) for Phase 2.

**Caveat (must resolve before coding):** Based on Papers 1–6 only. Papers 7–9 (OpenVLA, Pi-0, VLA-R1) not yet read. **Pi-0 is a flow-matching VLA** — read it first; it may overlap with Phase 1 and either weaken the novelty or give a strong baseline to build on.

---

## Phase 1 — Octo + Flow-Matching Head (Primary)

**Risk:** Low · **Quant relevance:** Medium · **Shippability:** High · **Status:** guaranteed deliverable

### The idea
Octo decodes actions with a 20-step DDPM diffusion head. KAN-We-Flow showed flow matching achieves the same accuracy in **one** Euler step. Swap Octo's diffusion head for a flow-matching head; measure the speed/accuracy tradeoff.

### Why this is a solid deliverable
- Concrete, falsifiable hypothesis: *flow-matching head matches Octo accuracy at ~20× lower inference cost*
- Builds directly on two papers already understood deeply (Octo architecture + KAN-We-Flow flow matching)
- Octo's **modular design** makes this a *head-swap, not a rebuild* — the action head is decoupled from the frozen transformer via readout tokens
- Octo is **fully open source** with finetuning scripts and pretrained checkpoints (27M / 93M)
- Unambiguous metric: inference latency (ms) + success rate — easy to defend to a professor
- **Genuine utility even though the technique isn't new:** Octo is widely used; a drop-in ~20× faster head is practically valuable, and "flow matching transfers to the *generalist multi-robot* setting" is a slightly new data point

### Honest scope note
Phase 1 *alone* is a strong engineering / empirical result and a complete internship deliverable — but the novelty is thin (flow-matching heads already exist). The research-paper contribution is Phase 2. Phase 1 is what makes Phase 2 credible and achievable.

### How to approach
1. Get Octo-Small running; reproduce one zero-shot benchmark number (sanity check)
2. Implement a flow-matching head: input = readout embedding $e$, predict velocity $v_\theta$; one Euler step $\hat a = a_t + (1-t)\,v_\theta$
3. Finetune ONLY the new head on a small in-domain dataset (~100 demos), transformer frozen — same recipe Octo uses
4. Benchmark diffusion head vs flow-matching head on the same backbone: latency per action + success rate
5. If accuracy holds at 1 step → strong result. If it degrades, sweep 1→2→4 Euler steps and report the curve.

**What to measure:** inference latency, success rate, steps-vs-accuracy curve.

**Done criterion:** a table comparing diffusion vs flow-matching head on ≥1 benchmark, with latency and success rate. That alone is a complete result.

---

## Phase 2 — Make the Flow Matching Adaptive (Stretch → Paper)

**Risk:** Medium · **Quant relevance:** High · **Shippability:** Medium · **Status:** only if Phase 1 finishes with time to spare

### The idea
KAN-We-Flow fixes two training constants by hand: $K=2$ MFM segments and $\lambda_{\text{ACR}}=1$. Make them **adaptive** — let the model choose segment count / loss weight from trajectory complexity or horizon, instead of a global constant. Apply this on top of the Phase 1 flow-matching head.

### Why this is the paper
- Genuinely novel — nobody has made flow-segment count or consistency-loss weighting adaptive to trajectory complexity. This is an open question, not a reproduction.
- Most **quant-aligned** of everything considered: consistency losses, EMA targets, adaptive estimation, distributional consistency — directly the kind of math quant research uses
- Reuses Phase 1 infrastructure → cheap to attempt once Phase 1 exists
- Combined framing for the paper: *"fast, adaptive flow matching for generalist VLAs"* — Phase 1 proves the head works, Phase 2 is the contribution

### How to approach (only after Phase 1)
1. Start from the working Phase 1 flow-matching head
2. Replace fixed $K=2$ with a schedule or a small predictor that picks segment count from horizon length
3. Alternatively / additionally: make $\lambda_{\text{ACR}}$ a function of measured drift instead of constant 1
4. Ablate adaptive vs fixed across difficulty tiers; show where adaptivity helps most (expect: long-horizon / very-hard tasks)

**What to measure:** success rate per difficulty tier, drift on long horizons, training stability, adaptive vs fixed ablation.

---

## Project Arc Summary

| | Phase 1 | Phase 2 |
|---|---|---|
| Role | Guaranteed deliverable | Stretch / paper |
| Contribution type | Engineering / empirical | Novel method |
| Depends on | Octo + KAN-We-Flow (understood) | Phase 1 codebase |
| Risk | Low | Medium |
| If it's the only thing that finishes | Still a complete result | N/A (won't start without Phase 1) |

---

## Alternatives Considered (Not Chosen — kept for reference)

These were ranked lower and are *not* part of the plan, but recorded so the reasoning isn't lost:

- **Adaptive HAMLET memory** (Option 2) — wrapper over any VLA, lowest risk, but more engineering than math; weakest quant alignment of the strong options.
- **GroupKAN into Octo's MLP layers** (Option 2) — clean cross-paper synthesis, but modifies the backbone → needs re-pretraining/heavy finetuning, much harder than a head-swap.
- **Extend BitVLA ternary quantization to another VLA** (Option 2/3) — quant-relevant, but quantization training is notoriously finicky; high risk of fighting instability all summer with nothing to show.
- **Octo data-mixture optimization** (Option 3) — highest scientific value, but each mixture needs a full Octo pretraining run (~14h on a TPU v4-128 pod) → infeasible on an intern compute budget.

---

## Before Coding — Resolve These

1. ~~Read Pi-0 (the blocking gate)~~ ✅ **RESOLVED.** Pi-0 read (see `Paper 8 Pi-0 VLA.md`). It does **not** do a controlled head-swap ablation (compares whole systems only) and uses **vanilla** flow matching (no MFM/ACR/adaptivity). Phase 1 *and* Phase 2 survive intact. Framing tweak required: drop "flow matching is new for VLAs" (Pi-0 owns that); reframe as *isolated controlled comparison on a small open generalist policy* + *adaptive flow matching*. Pi-0 is now the **reference implementation** (Appendix B arch, τ-conditioning MLP, 3-block mask, KV-cache trick).
2. **New design decision for Phase 1:** vanilla flow matching (Pi-0-style, ~10 steps, simpler) vs consistency flow matching (KAN-We-Flow-style, 1 step, harder training). State this choice up front — it is part of the contribution.
3. Confirm compute budget — Phase 1 runs on a single GPU; Phase 2 retraining needs more.
4. Confirm professor's expected deliverable — Phase 1 satisfies "working result / contribution to lab pipeline"; Phase 1 + Phase 2 is the publishable arc.
5. OpenVLA (7) and VLA-R1 (9) remain non-blocking — read post–Phase 1 for breadth.
