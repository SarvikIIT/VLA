___
## The Problem

Robot action policies need to be accurate, fast, and lightweight. Current approaches fail on at least one:

| Approach | Accurate | Fast | Lightweight |
|---|---|---|---|
| Diffusion (DP3) | ✓ | ✗ many denoising steps | ✗ 255M params, heavy UNet |
| Flow Matching (FlowPolicy) | ✓ | ✓ one step | ✗ still inherits heavy UNet |
| **KAN-We-Flow** | ✓ | ✓ | ✓ |

The heavy **UNet backbone** is the bottleneck in both. DP3 has 255M parameters. Even fast flow-matching methods still inherit UNet-style stacks.

**The question:** Can we replace the UNet with something much smaller and faster without losing accuracy?

---

## Background: What is Flow Matching?

Instead of iteratively denoising noise into an action (diffusion = 50+ steps), flow matching learns a **velocity field** — a single function that says "go in this direction" — and you follow it in **one step**.

The straight-line path between noise $a_{\text{src}}$ and target action $a_{\text{tar}}$:

$$a_t = (1-t)\,a_{\text{src}} + t\,a_{\text{tar}}, \quad t \in [0,1]$$

The network learns velocity $v_\theta$ along this path. At inference, one Euler step gives the action: (Same as numerical technique approach h here is (1-t)).

$$f_\theta(t, a_t, s, v) = a_t + (1-t)\,v_\theta(a_t, t, s, v)$$

- $s$ = robot state
- $v$ = visual representation
- $t$ = timestep along the flow path (not robot timestep)

**Why one step is enough:** flow matching learns a straight-line trajectory in action space, so a single Euler integration lands accurately at the target.

---

## KAN-We-Flow: Three Components

### 1. RWKV — Replacing UNet's Temporal Mixing

**RWKV = Receptance Weighted Key Value**

Normal attention is $O(n^2)$ — expensive for long sequences. RWKV does sequence mixing in **linear time** using exponential time-decay: recent tokens matter more, older ones decay away.
Apne quant wala **EMA**

The time-mixing output (forward scan):

$$\tilde{v}_t^{\rightarrow} = \frac{\displaystyle\sum_{i=1}^{t-1} \exp(-(t-1-i)w + k_i) \odot v_i + \exp(u + k_t) \odot v_t}{\displaystyle\sum_{i=1}^{t-1} \exp(-(t-1-i)w + k_i) + \exp(u + k_t)}$$

- $w \in \mathbb{R}^C$ = per-channel decay rate (learned) — controls how fast past tokens fade
- $k_i, v_i$ = key and value projections of token $i$
- $u$ = learned "current token" bias

They run this **bidirectionally** (forward + backward scan, then sum):

$$\tilde{v}_t = \tilde{v}_t^{\rightarrow} + \tilde{v}_t^{\leftarrow}$$

So each token sees full trajectory context in both directions. The time-mixing output is then gated by a receptance $r_t$:

$$\text{TM}(x_t) = W_o(\sigma(r_t) \odot \tilde{v}_t), \quad W_o \in \mathbb{R}^{C \times C}$$

The channel-mixing branch applies a token-wise gated MLP:

$$\text{CM}(x_t) = \sigma(r'_t) \odot \left(W'_v[\text{ReLU}(k'_t)]^2\right)$$

**Key idea:** RWKV captures long-range temporal dependencies across the action trajectory at linear cost — no quadratic attention needed.

---

### 2. GroupKAN — Replacing UNet's Channel MLPs

**KAN = Kolmogorov-Arnold Networks**

A normal MLP layer uses fixed nonlinearity $\sigma$ with learned weight matrices $W$:

$$\text{MLP}(\mathbf{Z}) = (W_{K-1} \circ \sigma \circ W_{K-2} \circ \cdots \circ W_0)\,\mathbf{Z}$$

A KAN layer replaces fixed $\sigma$ with **learnable spline functions $\phi$ placed on the edges**:

$$\text{KAN}(\mathbf{Z}) = (\Phi_{K-1} \circ \Phi_{K-2} \circ \cdots \circ \Phi_0)\,\mathbf{Z}$$

Each $\Phi_k$ is a matrix of learned univariate spline functions $\{\phi^{(k)}_{q,p}\}$ — one per edge between nodes. The network learns the **shape of the function itself**, not just scaling weights. This approximates complex nonlinearities with far fewer parameters.

**GroupKAN** — split input channels into $G=4$ groups, apply an independent KAN to each group, then concatenate:

$$\mathbf{Y}_g = \text{KAN}_g(\mathbf{X}_g) \in \mathbb{R}^{B \times T \times C/G}, \quad g = 1,\ldots,G$$

$$\mathbf{Y} = \text{Concat}(\mathbf{Y}_1, \ldots, \mathbf{Y}_G) \in \mathbb{R}^{B \times T \times C}$$

They also add **Channel Affinity Modulation (CAM)** — a learned gating vector that highlights task-relevant channels using temporal pooling $\bar{\mathbf{X}}$:

$$a = \sigma\!\left(W_2\,\varphi(W_1\bar{\mathbf{X}})\right) \in \mathbb{R}^{B \times C}$$

Final GroupKAN output:

$$\hat{\mathbf{X}} = \mathbf{X} + \text{DropPath}(\text{LN}(\mathbf{A} \odot \mathbf{Y}))$$

---

### 3. RWKV-KAN Block — Full Architecture

Each block combines both branches with pre-norm residuals:

$$z_t = x_t + \text{CM}(\text{LN}_2(x_t))$$

$$y_t = z_t + \text{TM}(\text{LN}_1(z_t))$$

The full model is a **U-shaped encoder-decoder** (3 stages down, 3 stages up) stacking these blocks instead of heavy UNet convolutions.

**Inputs concatenated as condition:**
- Point cloud → Vision Encoder → visual embedding $v$
- Robot state → State Encoder → state embedding $s$
- Timestep $t$ → Time Encoder → time embedding
- Noisy action $a_t$ → fed directly into the UNet

---

### 4. Consistency Flow Matching (CFM) Objective

Two training losses:

**Endpoint loss** — decoded action should match across nearby times:

$$\mathcal{L}_{\text{end}} = \mathbb{E}\!\left[\left\|f_\theta(t, a_t, s, v) - f_{\theta^-}(t+\Delta t, a_{t+\Delta t}, s, v)\right\|_2^2\right]$$

**Velocity loss** — velocity field should be consistent:

$$\mathcal{L}_{\text{vel}} = \mathbb{E}\!\left[\left\|v_\theta(a_t, t, s, v) - v_{\theta^-}(a_{t+\Delta t}, t+\Delta t, s, v)\right\|_2^2\right]$$

$$\mathcal{L}_{\text{CFM}} = \mathcal{L}_{\text{end}} + \alpha\,\mathcal{L}_{\text{vel}}$$

where $\theta^-$ is an exponential moving average of $\theta$. They use $K=2$ segments (MFM) for better expressivity.

---

### 5. Action Consistency Regularization (ACR)

Flow matching can drift on long horizons. ACR is a **training-only** auxiliary loss — no extra inference steps.

Given noisy action $a_t$, do a one-step decode to the horizon $t=1$:

$$\hat{a}_1 = f_\theta(t, a_t, s, v) = a_t + (1-t)\,v_\theta(a_t, t, s, v)$$

Penalize deviation from expert demonstration $a^*$ over a control window $W$:

$$\mathcal{L}_{\text{ACR}} = \frac{1}{|W|} \sum_{u \in W} \left\|(\mathcal{S}_W \hat{a}_1)_u - a^*_u\right\|_2^2$$

It biases the learned velocity field toward expert-faithful solutions, reducing drift on longer horizons.

**Final training loss:**

$$\mathcal{L} = \mathcal{L}_{\text{MFM}} + \lambda_{\text{ACR}}\,\mathcal{L}_{\text{ACR}}, \quad \lambda_{\text{ACR}} = 1$$

---

## Results

### Efficiency (Table III) — the main win

| Method | Params | FLOPs | GPU Memory |
|---|---|---|---|
| DP3 | 255M | 0.3G | 996MB |
| FlowPolicy | 255M | 0.4G | 980MB |
| MambaPolicy | 47.9M | 0.03G | 137MB |
| **KAN-We-Flow** | **33.6M** | **0.03G** | **101MB** |

- **86.8% fewer parameters than DP3**
- Inference: **8–10ms on Adroit**, **6.7–11.8ms on Meta-World** → ~100Hz real-time control
- DP3 runs at 103–141ms — over 10× slower

### Accuracy (Table I) — smaller AND better

On Meta-World:
- Easy: **92.0** vs 88.2 (MP1) and 85.0 (FlowPolicy)
- Very Hard: **71.3** vs 67.2 (MP1) and 53.0 (FlowPolicy)

Best accuracy across all difficulty tiers while being the smallest model.

---

## Ablations (Table IV) — what each part contributes

| Components Active | SR1 | SR5 |
|---|---|---|
| Baseline only | 59.5 | 45.5 |
| + RWKV time-mixing | 62.0 | 48.3 |
| + RWKV channel-mixing | 64.5 | 50.5 |
| + GroupKAN | 65.0 | 53.1 |
| + ACR | **68.0** | **56.5** |

Every component adds value. RWKV gives the largest single jump; ACR gives the final precision boost.

---

## Summary Table

| Component | Replaces | What it does |
|---|---|---|
| RWKV (time-mixing) | UNet temporal layers | Long-range trajectory context, linear complexity |
| RWKV (channel-mixing) | UNet feature MLPs | Per-feature gated mixing |
| GroupKAN | Heavy MLP layers | Learnable spline functions, fewer parameters |
| CAM | Fixed channel weighting | Highlights task-relevant channels adaptively |
| CFM objective | Standard flow loss | Enforces velocity + endpoint consistency |
| ACR | Nothing (auxiliary) | Anchors one-step decode to expert demos, training only |

---

## Key Takeaways

1. **UNet is the bottleneck** — not the flow matching paradigm itself
2. **RWKV handles time cheaply** — linear complexity vs quadratic attention, bidirectional scan captures full trajectory context
3. **KAN replaces fixed activations with learned splines** — same expressivity, far fewer parameters
4. **ACR is free at inference** — just a training loss that makes the velocity field more expert-faithful
5. **Result:** 86.8% fewer params, ~10× faster inference, still best accuracy across all benchmarks
