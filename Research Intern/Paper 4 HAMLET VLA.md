
___
## The Problem

Robotic manipulation tasks are **non-Markovian** — the right action depends on history, not just the current frame. Example: cube is hidden under a cup. You can't know where it is from what you see right now. Most VLAs ignore this and predict actions solely from the current observation.

**Why not just feed the last N frames directly?**
The paper tests this ("Multi-frame baseline") — it actually **hurts** on generic benchmarks:
- −3.3% on RoboCasa, −8.8% on LIBERO
- 3.6× more GPU memory, 35% slower inference
- Model learns **spurious correlations** between consecutive frames ("causal confusion") — memorizes the order of frames, not the meaning
## HAMLET IS A WRAPPER OVER THE VLA MODEL NOTHING NEW

---

## HAMLET's Solution: Two Components

1. **Moment Tokens** — compress each timestep into a small vector summary
2. **Memory Module** — Transformer that selectively reads the history of those summaries

---

## Component 1: Moment Tokens

At each timestep t, append 4 learnable vectors $\mathbf{m}_t$ to the VLM's input sequence. Feed everything through the VLM:

$$[\mathbf{h}_t;\, \mathbf{m}'_t] = \mathcal{F}_\theta([\mathbf{o}_t, \mathbf{c};\, \mathbf{m}_t])$$

- $\mathbf{o}_t$ = current camera images
- $\mathbf{c}$ = language instruction ("cover the cube with the nearest cup")
- $\mathbf{m}_t$ = learnable moment token vectors (4 vectors, trained)
- $\mathbf{m}'_t$ = output — a **compressed summary** of the scene at timestep $t$

Because the VLM uses causal attention internally, $\mathbf{m}'_t$ has attended to both the images and the instruction. It's a cheap 4-vector description of "what matters at this moment." This is what gets stored across time instead of raw frames.

---

## Component 2: Time-Contrastive Learning (TCL) — making moment tokens useful

Without TCL, moment tokens might collapse and encode useless static background. TCL is a **pre-training step** that forces them to be temporally discriminative.

**Core idea:**
- Same frame + augmentation → should look SIMILAR (positive pair)
- Different timestep → should look DIFFERENT (negative pair)

| Symbol | Meaning |
|---|---|
| $z_t = g(\mathbf{m}'_t)$ | Anchor — projected moment token for current frame |
| $z_t^+= g(\mathbf{m}'_{t,\text{aug}})$ | Positive — same frame, blurred/jittered/occluded |
| $z_t^- = g(\mathbf{m}'_{t'})$ | Negative — different timestep $t' \neq t$ (>16 steps away) |

**The loss (InfoNCE / contrastive loss):**

$$\mathcal{L}_{\text{TCL}}(z_t, z_t^+) = -\sum_{t=1}^{B} \log \frac{\exp\!\left(\text{sim}(z_t, z_t^+)/\tau\right)}{\exp\!\left(\text{sim}(z_t, z_t^+)/\tau\right) + \exp\!\left(\text{sim}(z_t, z_t^-)/\tau\right)}$$

- $\text{sim}(\mathbf{a}, \mathbf{b})$ = cosine similarity
- $\tau = 0.07$ (temperature — controls how sharp the separation is)
- Summed over all $B$ anchors in the minibatch

**Result:** moment tokens learn to attend to things that CHANGE over time (gripper, objects being manipulated) and ignore static things (walls, table background). Figure 4(a) shows this visually — attention concentrates on gripper and task-relevant objects after TCL.

During TCL training, the VLM backbone is **frozen** — only the moment token embeddings and projection head g are trained.

---

## Component 3: Memory Module (the Transformer)

Stack the last $T=4$ timesteps of moment tokens into one history matrix:

$$\mathbf{M}' = [\mathbf{m}'_{t-k(T-1)};\, \ldots;\, \mathbf{m}'_{t-k};\, \mathbf{m}'_t] \in \mathbb{R}^{L \times d}$$

- $k$ = action chunk length (how many actions predicted at once) (in chunks for action chuncking)
- $L = T \times n_m = 4 \times 4 = 16$ total token rows
- $d$ = embedding dimension

Run standard Transformer self-attention (Equation 6 in paper):

$$\mathbf{Q} = \mathbf{M}'\mathbf{W}_q, \quad \mathbf{K} = \mathbf{M}'\mathbf{W}_k, \quad \mathbf{V} = \mathbf{M}'\mathbf{W}_v$$

$$\mathbf{H} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{C}\right)\mathbf{V}$$

**Breaking this down:**
- $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ = learned weight matrices that project $\mathbf{M}'$ into three different "views" (comes from **Attention is all you need wala paper**)
- $\mathbf{Q}\mathbf{K}^\top$ = every token's query dot-producted with every token's key → relevance scores ($L \times L$ matrix)
- $/ \sqrt{d}$ = scaling to prevent vanishing gradients when $d$ is large
- $+ \mathbf{C}$ = **causal mask** — adds $-\infty$ where token $i$ would look at future position $j > i$, making those weights $\to 0$ after softmax. Token at step $t$ cannot cheat by looking at step $t+1$.
- $\text{softmax}(\cdot)$ = converts raw scores to probability weights (each row sums to 1)
- $\times \mathbf{V}$ = weighted sum of value vectors — each token's output is a blend of past tokens, weighted by relevance

Output $\tilde{\mathbf{M}}' \in \mathbb{R}^{L \times d}$ — same shape as input, but now every token is informed by relevant history.

**Take only the last $n_m = 4$ rows of $\tilde{\mathbf{M}}'$** → this is $\tilde{\mathbf{m}}'_t$, the **history-augmented moment token**.

It started as "what's happening at timestep $t$" and after the Transformer it's "what's happening at $t$, informed by everything relevant from the past 4 timesteps."

### Why Transformer beats simple concatenation

| Approach                           | What happens                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| $$\mathrm{Concat}(M_1, M_2, M_3)$$ | Treat all equally → smear everything together, noise dilutes signal            |
| Transformer                        | Weight by relevance → M2 suppressed to ~7%, M1 at ~91% when M1 is what matters |

The Transformer is a **learned, content-based lookup** into episode history. Not recency-based — relevance-based.

---

## Integration into Action Prediction

$$[\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+k-1}] = \mathcal{A}_\psi([\mathbf{h}_t;\, \tilde{\mathbf{m}}'_t],\, \mathbf{s}_t)$$

- $\mathbf{h}_t$ = VLM's representation of the current frame
- $\tilde{\mathbf{m}}'_t$ = history-augmented memory from the Memory Module
- $\mathbf{s}_t$ = proprioceptive state (joint angles, gripper state)
- $\mathcal{A}_\psi$ = action expert (diffusion/flow-matching head)

The whole pipeline is trained end-to-end with standard action prediction loss. VLM backbone is frozen during fine-tuning for GR00T N1.5; both VLM and moment tokens stay trainable for CogACT.

---

## Results

### Real-world tasks (Table 1) — the main result

| Method | Has History? | Avg Success |
|---|---|---|
| GR00T N1.5 baseline | No | 29.2% |
| + Multi-frame (naive) | Yes | 45.8% |
| + HAMLET | Yes | **76.4%** |

**+47.2% improvement over baseline.** Multi-frame helps but HAMLET dominates.

Real tasks tested: Pick-and-Place Twice, Cover-and-Stack, Swap Cubes — all require remembering what happened earlier in the episode.

### Simulation (Table 2) — Multi-frame hurts, HAMLET helps

On RoboCasa and LIBERO (tasks that don't specifically require history):
- Multi-frame: −3.3% (RoboCasa), −8.8% (LIBERO)  ← causal confusion
- HAMLET: +2.8% (RoboCasa), +2.0% (LIBERO)  ← still helps without hurting

Key insight: because HAMLET feeds single-frame inputs externally via the memory module, it preserves the single-frame VLA's generalizability.

### Efficiency (Table 4)

| Method | History Length | Latency | Peak Memory |
|---|---|---|---|
| GR00T N1.5 | 1 | 80.5ms (1.00×) | 289MB (1.00×) |
| + Multi-frame | 4 | 108.5ms (1.35×) | 1051MB (3.64×) |
| + HAMLET | 4 | **82.4ms (1.02×)** | **566MB (1.96×)** |
| + Multi-frame | 8 | 193.0ms (2.40×) | 2023MB (7.00×) |
| + HAMLET | 8 | **85.8ms (1.07×)** | **578MB (2.00×)** |

HAMLET adds basically no latency (1.02×) compared to multi-frame's 1.35×. Near-free.

---

## Ablations — which parts actually matter (Table 5)

### Component analysis (Table 5a)

| Moment Token | TCL | Memory Module | Score |
|---|---|---|---|
| ✗ | ✗ | ✗ | 62.6 |
| ✓ | ✗ | ✗ | 63.1 |
| ✓ | ✓ | ✗ | 63.4 |
| ✓ | ✗ | ✓ | 64.8 |
| ✓ | ✓ | ✓ | **65.4** |

**Memory module is the most critical component.** Removing it causes the biggest drop. TCL consistently helps but isn't the main driver.

### Memory architecture (Table 5c)

| Method | Score |
|---|---|
| No Memory | 62.6 |
| Moment Concat (just paste tokens together) | 62.7 |
| RNN | 64.5 |
| LSTM | 65.0 |
| GRU | 64.3 |
| **Transformer** | **65.4** |

**Moment Concat ≈ No Memory.** The Transformer's selective attention is what makes history useful — just concatenating tokens blindly doesn't work.

### Moment token length (Table 5b)

1 → 64.3, 4 → 65.4, 8 → **66.4**, 16 → 65.9, 32 → 62.7, 64 → 62.5

Sweet spot is 4–8. Too many tokens → redundancy hurts performance.

---

## Why it generalizes beyond its window size

HAMLET uses T=4 history length by default, but on "Pick-and-Place Three Times" the robot needs to remember 14+ steps. HAMLET still improves dramatically (37.5% vs 8.3% baseline).

Reason: Transformer KV-cache propagation. Each token's key-value pair influences all later tokens during attention. Even after old tokens drop from the explicit window, their information has already been baked into the newer tokens' representations through the attention mechanism.

---

## Summary Table

| Component | What it does | Why it's needed |
|---|---|---|
| Moment Tokens | 4 learnable vectors per timestep, compressed scene summary via VLM | Cheap to store, captures task-relevant info |
| TCL | Pre-trains tokens to distinguish different timesteps | Stops tokens encoding static/useless background |
| Memory Module (Transformer) | Self-attention over last 4 timesteps of moment tokens | Selectively retrieves relevant past, ignores irrelevant |
| Causal Mask | Prevents attending to future tokens | Ensures proper temporal ordering |
| History-augmented feature `m̃'_t` | Last n_m rows of Transformer output | Combined current + history signal fed to action expert |

---

## Key Takeaways

1. **History matters but raw frames are expensive and cause causal confusion** — you need compression
2. **Moment tokens = cheap compressed summaries** of each timestep (4 vectors instead of full images)
3. **TCL forces tokens to be temporally discriminative** — ignore static, capture change
4. **The Transformer memory selectively attends** — not all past timesteps matter equally
5. **Just concatenating past tokens doesn't help** — the selective attention of the Transformer is the key mechanism
6. **Near-zero computational overhead** (1.02× latency at history length 4) vs multi-frame's 1.35×
