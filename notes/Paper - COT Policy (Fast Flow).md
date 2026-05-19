	___
# COT Policy — Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings

> **Paper:** Sochopoulos et al., University of Edinburgh + Honda Research Institute Europe, May 2025
> **Core claim:** By using **Conditional Optimal Transport (COT)** couplings during flow matching training, robot policies can generate high-quality diverse actions in **1–2 integration steps** instead of 20+, with **zero extra training cost** (no distillation needed).

---

## The Problem

Diffusion / flow matching policies are the best action decoders for robots — they capture multimodal action distributions. But they're **too slow for real-time control** because sampling requires numerically integrating an ODE over many steps.

| Method | Quality | Speed | Training Cost |
|---|---|---|---|
| Diffusion Policy (DDPM/DDIM) | ✓ | ✗ (20+ steps) | ✓ single phase |
| Vanilla Flow Matching (CFM) | ✓ | ✗ (curved paths → need many steps) | ✓ single phase |
| Consistency Distillation | ✓ | ✓ (1–2 steps) | ✗ needs teacher + extra phases |
| **COT Policy (this paper)** | **✓** | **✓ (1–2 steps)** | **✓ single phase** |

The real bottleneck: vanilla flow matching learns **curved** integration paths (noise → action). When you try to integrate in 1–2 Euler steps, you overshoot the curve → bad actions. To get straight paths, prior work uses **distillation** (train a teacher, then distill into a student) — expensive and adds training phases.

**This paper's question:** Can we make the paths straight *during training itself*, without any distillation, by simply choosing **which noise sample pairs with which action sample** more intelligently?

---

## Background: Flow Matching (CFM) — the baseline

> **📖 See Paper 5 (KAN-We-Flow) and Paper 8 (π₀) notes for deeper flow matching background.**

The ODE that maps noise $x_0 \sim \mathcal{N}(0, I)$ to actions $x_1 \sim p_1$:

$$\frac{dx}{dt} = v_\theta(t, x), \quad t \in [0, 1]$$

Train the velocity field $v_\theta$ with the **Conditional Flow Matching** loss (linear interpolant path):

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1],\; (x_0, x_1) \sim q(x_0, x_1)} \left\| v_\theta(t,\; tx_1 + (1-t)x_0) - (x_1 - x_0) \right\|^2$$

- $x_0$ = noise sample, $x_1$ = target action
- The interpolant $x_t = tx_1 + (1-t)x_0$ moves along a **straight line** from $x_0$ to $x_1$
- Target velocity is $x_1 - x_0$ (constant along this line)

**The key question:** how do you pair $x_0$ with $x_1$? This is the **coupling** $q(x_0, x_1)$.

---

## The Coupling Problem — Why Naive OT Fails

> **📖 See Figure 1 in the paper** — this is the most important figure. Study it carefully.

### Three coupling strategies:

**1. Independent Coupling (I-CFM / vanilla CFM):**
$$q(x_0, x_1) = p_0(x_0) \cdot p_1(x_1)$$

Each noise sample pairs with a random target. Paths cross each other wildly → the learned velocity field becomes **curved** (averaging over conflicting directions) → needs many integration steps for accuracy.

**2. OT Coupling (OT-CFM):**
Compute the **Optimal Transport** plan — pair each noise $x_0$ with target $x_1$ that minimizes total squared distance $\|x_0 - x_1\|^2$. This gives straighter paths.

**BUT for conditional generation, this is BIASED.** Why? Because OT ignores the conditioning variable $c$ (the observation). Example from Figure 1:
- Noise from top Gaussians get paired almost exclusively with top-moon targets (because they're closest in Euclidean distance)
- At inference, if you sample noise from the top AND condition on $c = 1$ (bottom moon), the flow has no training signal for this pairing → **out-of-distribution**

**3. Conditional OT Coupling (COT-CFM — this paper):**
Pair noise and targets using a cost that accounts for BOTH spatial distance AND observation similarity.

---

## COT Policy: The Core Method

### Step 1: The COT Cost Function

Given a batch of target samples $B_1 = \{(x_1^{(i)}, c^{(i)})\}$ and noise samples $B_0 = \{(x_0^{(i)}, c_0^{(i)})\}$, compute OT with the augmented cost:

$$c\big((x_0, c_0),\; (x_1, c_1)\big) = \|x_0 - x_1\|^2 + \|\gamma(c_0 - c_1)\|^2$$

- **First term** $\|x_0 - x_1\|^2$: pair nearby noise and targets (straightens paths)
- **Second term** $\|\gamma(c_0 - c_1)\|^2$: pair samples with **similar observations** (prevents the bias from OT-CFM)
- $\gamma$ = hyperparameter controlling the trade-off

**The interpolation between regimes:**
- $\gamma = 0$ → recovers OT-CFM (ignores conditions, gets biased)
- $\gamma \to \infty$ → exact conditional OT (within each condition, do OT separately)
- Middle $\gamma$ → balance between straight paths and unbiased conditional generation

The noise conditions $c_0^{(1)}, \ldots, c_0^{(N)}$ are a **random permutation** of the target conditions $c^{(1)}, \ldots, c^{(N)}$ — this ensures both batches have the same condition distribution.

### Step 2: Handling Continuous Observations (Robot Setting)

In robotics, observations $o$ are continuous (RGB images + proprioception). You can't have exact condition matches in a batch. The paper's insight:

> "The conditional distribution of robot actions is not sensitive to small changes in the state of the environment."

**Solution:** Discretize observations into clusters before computing OT:

1. **Encode:** $e = E(o)$ — PCA on each image batch (dimensionality reduction to 100 features), leave proprioception unchanged
2. **Cluster:** $c = Q(e)$ — K-means clustering into $K = 64$ clusters, represent each observation by its cluster centroid

These discretized $c$ values are used **only for computing the OT coupling**. The actual flow network $v_\theta(t, x \mid o)$ is still conditioned on the **full unprocessed observation** $o$.

### Step 3: The Full Training Algorithm

> **📖 See Algorithm 1 in the paper (page 5).**

```
For each training batch:
  1. Sample (action, observation) pairs from dataset
  2. Sample noise x_0 ~ N(0, I)
  3. Encode observations: e = PCA(o), cluster: c = KMeans(e)
  4. Randomly permute conditions for noise: c_0 = randperm(c)
  5. Solve minibatch OT with cost: ||x_0 - x_1||² + ||γ(c_0 - c_1)||²
  6. Use OT coupling to pair noise with actions
  7. Compute standard CFM loss on paired (x_0, x_1)
  8. Update θ
```

**Critical insight:** Steps 3–6 add **negligible computational overhead** when GPU-accelerated (PCA is a matrix multiply, K-means is precomputed, OT is solved via Earth Mover's Distance on the batch). No extra forward passes, no teacher model, no distillation.

---

## Architecture

> **📖 See Appendix B.1 and B.3 in the paper for full details.**

The architecture is intentionally kept **identical to Diffusion Policy** to ensure a fair comparison:

| Component | Choice | Details |
|---|---|---|
| Flow/Noise network | CNN-based **U-Net** | ~240M params, same as Diffusion Policy |
| Vision encoder | **ResNet-18** (randomly initialized) | Trained end-to-end with the policy |
| ODE solver | **Midpoint method** | 2 NFE (network function evaluations) |
| Observation horizon | 2 frames | |
| Action prediction horizon | 16 steps (manipulation) / 60 steps (MimicGen) | |
| Action execution horizon | 8 steps (manipulation) / 50 steps (MimicGen) | |
| Image size | 96×96 (sim) / 240×320 or 96×120 (real) | |

**The midpoint solver vs Euler:**
- Euler with NFE=2 takes 2 steps: $x_{t+h} = x_t + h \cdot v_\theta(t, x_t)$
- Midpoint with NFE=2 uses 2 function evaluations for a higher-order update — more accurate for the same compute
- From Figure 5: midpoint consistently outperforms Euler at the same NFE

---

## Why COT Works — Geometric Intuition

> **📖 See Figure 1 in the paper — study all 6 panels carefully.**

**The core geometric insight:**

In independent CFM, a single noise point might need to go to DIFFERENT targets depending on the condition. This creates crossing paths. The velocity field $v_\theta$ must average these conflicting directions → curved, averaged flow → needs many steps.

In OT-CFM, paths are straight but **biased**: noise at position A always goes to the nearest target regardless of condition. When you condition on a different class at inference, the flow has never seen this pairing.

COT couples noise-target pairs that share similar conditions AND are nearby. Within each condition cluster, paths are optimally straight. Across clusters, each cluster's flow is trained correctly for its own condition.

**Result:** nearly straight paths that are also **correctly conditioned** → 1–2 step integration works.

---

## The $K$ (Number of Clusters) — The Key Hyperparameter

> **📖 See Figure 4 in the paper — success rate vs number of clusters.**

$$K = 1 \implies \text{COT-CFM} \equiv \text{OT-CFM} \quad (\text{all same condition, pure spatial OT})$$

$$K = |D| \implies \text{COT-CFM} \equiv \text{I-CFM} \quad (\text{every sample unique condition, OT degenerates to random})$$

| $K$ value | Behavior |
|---|---|
| Too few (e.g., 8) | Close to OT-CFM — same biasing problem |
| Sweet spot (~64, ≈ batch size) | Best performance — diverse clusters, condition-aware OT |
| Too many (→ dataset size) | Degenerates to vanilla CFM — each condition unique, OT irrelevant |

**Practical choice:** $K = 64$ (equal to batch size) works well across all tasks. This is the **only hyperparameter** that meaningfully impacts performance.

---

## Results

### Simulation — Manipulation Tasks (Table 1)

> **📖 See Table 1 (page 6) and Figure 2 for task visualizations.**

6 tasks: threading, stack, coffee, square (MimicGen), push-T, cup (dm-control). 150 rollouts per task, 2 seeds.

| Method | NFE | threading | stack | coffee | square | push-T | cup | **Avg** |
|---|---|---|---|---|---|---|---|---|
| CFM | 4 | 0.730 | 0.927 | 0.720 | 0.753 | 0.877 | 0.776 | 0.797 |
| OT-CFM | 4 | 0.723 | 0.940 | 0.623 | 0.697 | 0.712 | 0.743 | 0.740 |
| DP (DDIM) | 4 | 0.720 | 0.917 | 0.650 | 0.650 | 0.877 | 0.763 | 0.763 |
| DP (DDIM) | 20 | 0.720 | 0.940 | 0.650 | 0.707 | 0.881 | 0.787 | 0.781 |
| Adaflow | 1.29 | 0.707 | 0.937 | 0.717 | 0.667 | 0.849 | 0.823 | 0.783 |
| CFM | 2 | 0.737 | 0.913 | 0.730 | 0.690 | 0.870 | 0.797 | 0.790 |
| OT-CFM | 2 | 0.667 | 0.897 | 0.613 | 0.630 | 0.694 | 0.753 | 0.709 |
| **COT Policy** | **2** | **0.750** | **0.937** | **0.787** | **0.707** | **0.878** | **0.847** | **0.818** |

**Key observations:**
1. **COT with 2 steps beats everything** — including CFM/OT-CFM with 4 steps AND Diffusion Policy with 20 steps
2. **OT-CFM is consistently the WORST** — confirming that naive OT coupling hurts conditional generation
3. **COT's biggest wins are on multimodal tasks** (coffee +16.4% over DP, cup +8.4%)

### Multimodality — Trajectory Variance (Figure 3)

> **📖 See Figure 3 — left side shows maze trajectories, right side shows TV bar charts.**

The paper introduces **Trajectory Variance (TV)** — mean squared DTW distance from each trajectory to the barycenter. Higher TV = more diverse action plans.

**COT Policy exhibits HIGHER TV than CFM at all NFE levels**, meaning it doesn't collapse to a single mode when reducing integration steps. This is critical: some methods achieve fast inference by sacrificing diversity, which makes the robot inflexible.

At NFE=1 on maze tasks: CFM produces near-identical trajectories (low TV), while COT produces multiple distinct paths (see left panel of Figure 3).

### Real Robot Experiments (Table 2 + Figure 6)

> **📖 See Table 2 (page 8) and Figure 6 for task photos. See Figure 7 for the push-T multimodality demo.**

3 tasks on KUKA IIWA 14 robot, 2 cameras (end-effector + external), 30 teleoperated demonstrations each:

| Task | CFM (euler-1 / mid-2) SR | COT (euler-1 / mid-2) SR | CFM TTC(s) | COT TTC(s) |
|---|---|---|---|---|
| push-T | 0.2 / 0.4 | **0.8 / 0.6** | 54.4 / 44.7 | **35.7 / 35.8** |
| cup-stacking | 0.4 / 0.2 | **0.6 / 0.8** | 44.5 / 51.1 | **32.2 / 24.0** |
| cup-in-drawer | 0.2 / 0.2 | **0.8 / 0.2** | 53.3 / 54.0 | **35.3 / 52.9** |

**COT is dramatically better in real-world** — CFM produces intermittent/jerky motions because few-step integration overshoots curved paths. COT's straight paths give smooth, decisive actions.

> **📖 See Figure 7 (page 17)** — shows COT discovering BOTH modes (clockwise and anti-clockwise T rotation) even with just 1 Euler step. This is the multimodality preservation in action.

---

## Ablations

### NFE and Solver Type (Figure 5)

> **📖 See Figure 5 — three panels: Euler sweep, Midpoint sweep, convergence over epochs.**

**Left panel (Euler):** CFM needs ~10 steps to plateau. COT is already near-optimal at 2 steps. OT-CFM never catches up.

**Middle panel (Midpoint):** All methods benefit from midpoint over Euler. COT at NFE=2 already matches high-NFE CFM.

**Right panel (Training convergence):** COT, CFM, and DP converge at approximately the same epoch → **COT adds zero extra training time**.

### Effect of Clustering (Figure 4 + Appendix D.3 Table 6)

> **📖 See Figure 4 (page 7).**

COT without clustering: 0.774 avg. COT with clustering: **0.833 avg.** Clustering simplifies the hyperparameter problem — instead of tuning $\gamma$ per-dataset, set $K = 64$ and use a fixed $\gamma$ formula.

### $\gamma$ Scaling (Appendix E.1, Equation 6)

> **📖 See Figure 10 — OT matrix for different $\gamma$ values.**

The paper proposes an adaptive formula:

$$\gamma = 10 \cdot \frac{\sum_i \|x_0^{(i)} - x_1^{(i)}\|^2}{\sum_j \|c_0^{(j)} - c_1^{(j)}\|^2}$$

This makes the condition term ~10× larger than the spatial term, prioritizing condition matching while avoiding numerical instability. For $\gamma > 10000$, EMD solver produces numerical errors (diagonal artifacts in OT matrix — Figure 10).

---

## Low-Dimensional Distribution Experiments (Appendix D.1)

> **📖 See Table 5 and Figure 8 in the appendix.**

On synthetic 2D moons (discrete condition) and 1D fork (continuous condition):

| Method | NFE | Moons $W_2^2 \downarrow$ | Fork $W_2^2 \downarrow$ |
|---|---|---|---|
| CFM | 1 | 1.490 | 1.134 |
| OT-CFM | 1 | 0.777 | 2.678 |
| **COT-CFM** | **1** | **0.345** | **0.349** |
| CFM | 2 | 1.094 | 0.575 |
| OT-CFM | 2 | 0.709 | 0.992 |
| **COT-CFM** | **2** | **0.322** | **0.125** |

**2-step COT outperforms >60-step CFM and OT-CFM** with the adaptive Dormand-Prince solver! The straight paths from COT couplings are that effective.

Figure 8 shows the fork distribution flow — CFM at NFE=1 generates the conditional mean (blurred), OT-CFM is biased (wrong distribution shape), COT-CFM correctly reconstructs the fork structure.

---

## Theoretical Grounding (Appendix E.2)

**Proposition (informal):** For continuous conditions with large enough $\gamma$:

$$K \to 0: \quad \text{COT-CFM} \to \text{OT-CFM}$$
$$K \to |D|: \quad \text{COT-CFM} \to \text{I-CFM}$$

**Proof sketch:**
- $K = 1$: All conditions identical → cost reduces to $\|x_0 - x_1\|^2$ → pure spatial OT = OT-CFM
- $K = |D|$: Every observation is its own cluster → for large $\gamma$, the only zero-cost pairing is the one matching permuted indices → equivalent to random pairing = I-CFM

COT with $K \approx$ batch size is the sweet spot — enough clusters to be condition-aware, few enough that multiple samples share a cluster for meaningful OT.

---

## Limitations (the paper's own)

1. **Two hyperparameters:** $\gamma$ is not sensitive (use adaptive formula), but $K$ is. Different dataset sizes and batch sizes may need different optimal $K$.
2. **Gap between low and high NFE** — especially as action dimensionality grows. 2-step COT is good, but not always as good as 20-step. Could potentially combine with single-phase distillation methods to close this gap.
3. **Only tested on manipulation tasks** — not tested on locomotion, navigation at scale, or multi-task generalist settings.

---

## Comparison with Other Fast-Inference Methods from Your Papers

| Method | How it achieves speed | Training cost | Steps |
|---|---|---|---|
| Octo (Paper 6) | DDPM diffusion (20 steps) — NOT fast | Single phase | 20 |
| KAN-We-Flow (Paper 5) | Consistency FM + MFM + ACR | Single phase (complex losses) | **1** |
| π₀ (Paper 8) | Vanilla FM + KV-cache trick | Single phase | 10 |
| **COT Policy (this paper)** | **OT coupling during training** | **Single phase (simplest)** | **1–2** |

**Compared to KAN-We-Flow:** KAN-We-Flow replaces the UNet with RWKV-KAN for model efficiency. COT keeps the standard UNet but changes the training *coupling* for path straightness. Orthogonal contributions — they could be combined.

**Compared to π₀:** π₀ uses vanilla CFM (10 steps). COT's coupling strategy could be applied to π₀'s flow head to potentially reduce it to 1–2 steps.

---

## Summary Table

| Component | What it does | Why it's needed |
|---|---|---|
| Conditional OT coupling | Pairs noise-action using $\|x_0 - x_1\|^2 + \|\gamma(c_0 - c_1)\|^2$ | Straight paths + correct conditioning — the core contribution |
| PCA encoder $E$ | Reduces image dimensionality for clustering | Makes continuous observations tractable for OT |
| K-means discretizer $Q$ | Groups similar observations into $K=64$ clusters | Allows condition-aware OT without per-dataset $\gamma$ tuning |
| Condition permutation | Noise conditions = random permutation of target conditions | Ensures same condition distribution in both batches |
| Midpoint solver | Higher-order ODE integration in 2 NFE | Better accuracy than Euler for same computational cost |
| U-Net flow network | Standard 240M param architecture from DP | Fair comparison — all improvements come from coupling, not architecture |

---

## Figures and Tables to Study

| Reference | What to look at |
|---|---|
| **Figure 1** (p.4) | THE key figure — three coupling strategies on 2D moons. See how OT-CFM is straight but biased, CFM is unbiased but curved, COT is both straight AND unbiased |
| **Figure 2** (p.6) | Task visualizations — threading, stack, coffee, square |
| **Figure 3** (p.7) | Trajectory variance comparison — COT maintains diversity at low NFE |
| **Figure 4** (p.7) | Cluster count sweep — sweet spot at $K=64$ |
| **Figure 5** (p.8) | NFE sweep + training convergence — COT converges same speed as CFM |
| **Figure 6** (p.8) | Real robot task photos |
| **Figure 7** (p.17) | Two modes of push-T discovered by COT at NFE=1 |
| **Figure 8** (p.19) | Fork distribution flow visualization — most instructive for understanding geometry |
| **Figure 10** (p.21) | OT matrix vs $\gamma$ — numerical instability at extreme values |
| **Table 1** (p.6) | Main simulation results — COT 2-step beats DP 20-step |
| **Table 2** (p.8) | Real robot results — COT dramatically outperforms CFM |
| **Table 5** (p.18) | 2-Wasserstein distance on synthetic tasks — COT 2-step beats 60+ step baselines |

---

## Key Takeaways

1. **The coupling strategy during training determines integration straightness** — you don't need distillation or consistency losses; just pair noise and targets intelligently
2. **Naive OT coupling is HARMFUL for conditional generation** — it ignores the conditioning variable and produces biased flows. This is a non-obvious failure mode (OT-CFM is the worst method in every experiment)
3. **COT = condition-aware OT** — augment the cost with a condition distance term. Simple idea, large impact
4. **Clustering is the practical enabler** — discretizing continuous observations into K=64 clusters avoids per-dataset $\gamma$ tuning while preserving the benefit
5. **Zero extra training cost** — PCA, K-means, and minibatch OT are negligible overhead compared to the U-Net forward/backward pass
6. **2-step COT > 20-step Diffusion Policy** — on average across all tasks, with maintained multimodality
7. **Orthogonal to architecture improvements** — COT changes training, not the model. Could combine with KAN-We-Flow's architecture or π₀'s MoE design
