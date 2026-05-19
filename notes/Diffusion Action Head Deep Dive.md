# Diffusion Action Head — Complete Deep Dive

> This document explains everything about Octo's `DiffusionActionHead` from first principles.
> No shortcuts. Every line of code explained. Every equation derived.

---

## Table of Contents

1. The Problem It Solves
2. DDPM — The Math From Scratch
3. The Noise Schedule
4. Forward Process — Adding Noise
5. Reverse Process — Removing Noise
6. The Score Network (Diffusion MLP)
7. Full Class Walkthrough — Every Line
8. Training — The Loss Function
9. Inference — predict_action
10. How It Connects To Octo's Transformer
11. What Your Flow Head Replaces

---

## 1. The Problem It Solves

A robot arm has 7 joints. At each timestep, we want to predict:
- How much to move joint 1
- How much to move joint 2
- ... (7 numbers total)

The naive approach: train a neural net to output those 7 numbers directly (MSE regression).

**Why MSE fails for robot actions:**

Imagine the robot needs to pick up a cup. There are TWO valid ways to approach it — from the left or from the right. Both are correct.

With MSE, the model sees both demonstrations in training data and averages them:
- Demo 1: move left by 0.5
- Demo 2: move right by 0.5
- MSE prediction: move by 0.0 → stays still → knocks the cup over

Robot actions are **multimodal** — multiple valid actions exist for the same observation. MSE collapses them into one wrong average.

**Diffusion solves this** by learning the full distribution over actions, not just the mean. It can represent "left OR right" without averaging.

---

## 2. DDPM — The Math From Scratch

DDPM = Denoising Diffusion Probabilistic Models.

The core idea: **learn to reverse a noise-adding process.**

### The intuition

Imagine you have a clean action `a` (e.g., `[0.3, -0.1, 0.5, 0, 0, 0, 0]`).

You gradually add Gaussian noise to it over 20 steps until it becomes pure random noise.

Then you train a neural network to **reverse** this process — given noisy action, predict what the clean action was.

At inference: start from pure noise, run the learned reverse process 20 times → clean action.

### Why does this represent a distribution?

Because you start from random noise (which is sampled differently each time), the model generates different actions each run — but always valid ones from the learned distribution. Left approach sometimes, right approach other times.

### The two processes

```
FORWARD PROCESS (training, fixed, no parameters):
clean action a₀ → add noise → a₁ → add noise → a₂ → ... → a₂₀ (pure noise)

REVERSE PROCESS (inference, learned):
pure noise a₂₀ → denoise → a₁₉ → denoise → ... → a₀ (clean action)
```

The network only learns the reverse process. The forward process is a fixed mathematical formula.

---

## 3. The Noise Schedule

The noise schedule controls HOW MUCH noise is added at each step.

```python
self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
self.alphas = 1 - self.betas
self.alpha_hats = jnp.cumprod(self.alphas)
```

### betas (β) — the actual code

Here is the real `cosine_beta_schedule` from `diffusion.py`:

```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1                                          # 21
    t = jnp.linspace(0, timesteps, steps) / timesteps              # [0, 0.05, 0.1, ..., 1.0]
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2 # cosine curve, squared
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]            # normalize so ᾱ₀ = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])         # β_t = 1 - ᾱ_t/ᾱ_{t-1}
    return jnp.clip(betas, 0, 0.999)                               # safety clip
```

Line by line:

- `steps = timesteps + 1` → 21 points (we need N+1 points to get N betas via differences)
- `t = linspace(0, timesteps, steps) / timesteps` → evenly spaced values from 0 to 1: `[0, 0.05, 0.1, ..., 1.0]`
- `alphas_cumprod = cos((t+s)/(1+s) × π/2)²` → this directly defines **ᾱ** as a cosine curve. At t=0, cos(≈0)²≈1. At t=1, cos(π/2)²≈0. The small offset `s=0.008` prevents ᾱ from being exactly 0 or 1 at the endpoints (numerical safety).
- `alphas_cumprod / alphas_cumprod[0]` → normalize so the curve starts exactly at ᾱ₀ = 1
- `betas = 1 - (ᾱ[1:] / ᾱ[:-1])` → recover β from the ratio of consecutive ᾱ values. Since ᾱ_t = ∏α_i and α_t = 1-β_t, we have α_t = ᾱ_t / ᾱ_{t-1}, so β_t = 1 - ᾱ_t/ᾱ_{t-1}.
- `jnp.clip(betas, 0, 0.999)` → never let β reach exactly 1 (would zero out signal entirely and break the math)

**Key insight:** This schedule actually defines ᾱ directly as a cosine curve, then derives β from it — the reverse of the naive "pick β then compute ᾱ" approach. The cosine shape means noise is added slowly at first, faster in the middle. This is gentler than a linear β schedule and is the standard modern choice (from the "Improved DDPM" paper).

```
β values (roughly):
β = [0.0001, 0.0003, 0.001, 0.003, ..., 0.02, ..., 0.999]
step:  0       1       2      3           10          19
```

Small β at the start = gentle noise addition early on.

### alphas (α)

```python
alphas = 1 - betas
# α = [0.9999, 0.9997, 0.999, 0.997, ..., 0.98]
```

α_t = how much of the SIGNAL to keep at step t. Close to 1 = keep most signal.

### alpha_hats (ᾱ) — the key quantity

```python
alpha_hats = cumprod(alphas)
# ᾱ_t = α₁ × α₂ × α₃ × ... × α_t
```

`cumprod` = cumulative product. Each entry is the product of all alphas up to that step.

```
ᾱ₀ = α₀                    ≈ 0.9999
ᾱ₁ = α₀ × α₁               ≈ 0.9996
ᾱ₂ = α₀ × α₁ × α₂          ≈ 0.9986
...
ᾱ₁₉ = α₀ × α₁ × ... × α₁₉  ≈ 0.001  (almost zero)
```

**Why does ᾱ matter?** Because of this magical property of DDPM:

> You can jump directly from clean action a₀ to noisy action at ANY step t, without simulating all intermediate steps.

The formula is:
```
a_t = sqrt(ᾱ_t) × a₀  +  sqrt(1 - ᾱ_t) × ε
      ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
      scaled clean action   scaled noise
```

where ε ~ N(0, I) is standard Gaussian noise.

When t=0: sqrt(ᾱ₀) ≈ 1, sqrt(1-ᾱ₀) ≈ 0 → almost pure signal
When t=19: sqrt(ᾱ₁₉) ≈ 0, sqrt(1-ᾱ₁₉) ≈ 1 → almost pure noise

This is the KEY equation. Everything else follows from it.

---

## 4. Forward Process — Adding Noise

In the loss function:

```python
scale = jnp.sqrt(self.alpha_hats[time])
std   = jnp.sqrt(1 - self.alpha_hats[time])
noisy_actions = scale * actions_flat[None] + std * noise
```

This directly applies the formula above. Given:
- `actions_flat`: clean action `a₀`, shape `(batch, window, 7)`
- `time`: random integer t ∈ {0,...,19}
- `noise`: ε ~ N(0,I), same shape as actions

It produces `noisy_actions` = `a_t` — the clean action corrupted to noise level t.

**This is the entire forward process.** One formula, no loop needed. Jump straight from t=0 to any t.

```
FORWARD PROCESS DIAGRAM:

clean action a₀ = [0.3, -0.1, 0.5, 0, 0, 0, 0]
                         |
                    sample t = 12
                    sample ε ~ N(0,I)
                         |
                         ↓
noisy_action a₁₂ = sqrt(ᾱ₁₂) × a₀ + sqrt(1-ᾱ₁₂) × ε
                 = 0.6 × [0.3,-0.1,0.5,...] + 0.8 × [random noise]
                 = [0.18, -0.06, 0.3, ...] + [noise]
                 = [0.4, 0.7, -0.2, ...]   ← mostly noise
```

---

## 5. Reverse Process — Removing Noise

The reverse process runs 20 steps from t=19 → t=0.

At each step, the network predicts the noise ε that was added, then removes it.

The DDPM reverse step formula:

```
a_{t-1} = (1/sqrt(α_t)) × (a_t - ((1-α_t)/sqrt(1-ᾱ_t)) × ε_pred)  +  sqrt(β_t) × z
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
                    deterministic denoising step                            stochastic noise
```

where:
- `ε_pred` = what the network predicts the noise was
- `z ~ N(0,I)` = fresh random noise added for stochasticity (zero at the final step t=0)

**Why add noise in the reverse process?** To maintain the distribution. Without it, all samples from the same starting noise would converge to the same point. The stochastic term keeps diversity.

```
REVERSE PROCESS DIAGRAM:

pure noise a₁₉ = [random]
      ↓  network predicts ε_pred, apply formula
     a₁₈ = slightly less noisy
      ↓  network predicts ε_pred, apply formula
     a₁₇ = less noisy
      ↓
     ...
      ↓
     a₀  = clean action [0.3, -0.1, 0.5, 0, 0, 0, 0]
```

In code (from predict_action):

```python
def scan_fn(carry, time):
    current_x, rng = carry

    # 1. predict noise at this step
    eps_pred = module.apply(variables, transformer_outputs, input_time, current_x)

    # 2. DDPM reverse formula
    alpha_1 = 1 / jnp.sqrt(self.alphas[time])
    alpha_2 = (1 - self.alphas[time]) / jnp.sqrt(1 - self.alpha_hats[time])
    current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

    # 3. add stochastic noise (except at final step)
    z = jax.random.normal(key, shape=current_x.shape)
    current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)

    return (current_x, rng), ()
```

`jax.lax.scan` runs this 20 times efficiently — equivalent to a for loop but JAX-compiled.

---

## 6. The Score Network (Diffusion MLP) — REAL CODE from diffusion.py

This is the neural network inside the diffusion head. It is built by `create_diffusion_model`:

```python
def create_diffusion_model(out_dim, time_dim, num_blocks, dropout_rate, hidden_dim, use_layer_norm):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),          # time_preprocess
        MLP((2 * time_dim, time_dim)),                      # cond_encoder
        MLPResNet(                                          # reverse_network
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )
```

So the score network = `ScoreActor`, which is composed of **three sub-modules**:
1. `FourierFeatures` — encodes the timestep
2. `MLP` — further processes the time encoding
3. `MLPResNet` — the main network that predicts the noise

Let's walk through each.

---

### 6.1 — ScoreActor (the top-level wrapper)

```python
class ScoreActor(nn.Module):
    time_preprocess: nn.Module    # FourierFeatures
    cond_encoder: nn.Module       # MLP
    reverse_network: nn.Module    # MLPResNet

    def __call__(self, obs_enc, actions, time, train=False):
        t_ff = self.time_preprocess(time)              # 1. encode time → Fourier features
        cond_enc = self.cond_encoder(t_ff, train=train) # 2. process time encoding through MLP

        # 3. broadcast obs_enc to match cond_enc batch shape if needed
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            obs_enc = jnp.broadcast_to(obs_enc, new_shape)

        # 4. concatenate [time | obs | action] and predict noise
        reverse_input = jnp.concatenate([cond_enc, obs_enc, actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, train=train)
        return eps_pred
```

Step by step:

1. **`t_ff = self.time_preprocess(time)`** — the raw timestep (integer t, shape `(..., 1)`) goes through `FourierFeatures` → becomes a rich vector of size `time_dim` (32).

2. **`cond_enc = self.cond_encoder(t_ff)`** — that Fourier vector goes through a small MLP `(2*time_dim → time_dim)` = `(64 → 32)`, producing the final time conditioning vector.

3. **broadcast check** — `obs_enc` (the transformer embedding) might have a different leading shape than `cond_enc` (e.g., because of `n_diffusion_samples`). This expands `obs_enc` so the shapes line up for concatenation.

4. **concatenate `[cond_enc, obs_enc, actions]`** — the THREE inputs are stitched together along the last axis:
   - `cond_enc`: time conditioning, size 32
   - `obs_enc`: transformer embedding, size 384
   - `actions`: noisy action, size 7
   - → combined input of size `32 + 384 + 7 = 423`

   This combined vector goes into `reverse_network` (the MLPResNet), which outputs the predicted noise of size 7.

**This is the actual concatenation order: `[time | obs | action]`** — not `[obs | action | time]`. Order matters only in that the network learns weights for whatever order you give it; the point is all three signals are fed in together.

---

### 6.2 — FourierFeatures (time encoding)

```python
class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
```

**What this does:** maps a scalar timestep `t` to a `output_size`-dim vector using sine and cosine features.

Octo uses `learnable=True`:

- **`w = self.param("kernel", ...)`** — a *learnable* weight matrix of shape `(output_size//2, 1)` = `(16, 1)`. `self.param` registers a trainable parameter in Flax.
- **`f = 2π × x @ w.T`** — projects the scalar timestep up to 16 different frequencies. Because `w` is learned, the network learns which frequencies are useful for representing the timestep.
- **`concatenate([cos(f), sin(f)])`** — apply both cos and sin → `16 + 16 = 32` dims output.

**Why cos AND sin?** Together they form a complete basis — any periodic function can be represented. Using both means the encoding is injective (no two timesteps map to the same vector) and smooth (nearby timesteps → nearby vectors).

**Learnable vs fixed:** the `else` branch is the classic fixed sinusoidal encoding from "Attention Is All You Need." Octo uses the learnable version — the network discovers the best frequencies during training instead of using hand-set ones.

---

### 6.3 — MLP (the cond_encoder)

```python
class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x, train=False):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x
```

Created as `MLP((2 * time_dim, time_dim))` = `MLP((64, 32))`.

- Loops over `hidden_dims = [64, 32]`
- Layer 0: `Dense(64)` → since it's not the last layer, apply activation (swish)
- Layer 1: `Dense(32)` → it IS the last layer and `activate_final=False`, so NO activation — raw output

So this turns the 32-dim Fourier features → 64 → 32 final time conditioning vector.

**`nn.swish`** = SiLU = `x × sigmoid(x)`. A smooth activation function, like a smoother ReLU. Standard in modern diffusion models because it gives smoother gradients than ReLU.

**`kernel_init=default_init()`** = `xavier_uniform` — initializes weights so variance is preserved through layers (prevents vanishing/exploding activations at the start of training).

---

### 6.4 — MLPResNetBlock (one residual block)

```python
class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)   # expand 4×
        x = self.act(x)                       # swish
        x = nn.Dense(self.features)(x)        # project back down

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)  # match shapes if needed

        return residual + x                   # SKIP CONNECTION
```

This is the core building block. The exact order:

1. **`residual = x`** — save the input for the skip connection
2. **Dropout** (skipped, rate=0 in Octo)
3. **LayerNorm** — normalize activations
4. **`Dense(features × 4)`** — expand to 4× width (256 → 1024). This "inverted bottleneck" gives the block capacity to learn complex functions.
5. **`swish`** — nonlinearity
6. **`Dense(features)`** — project back down (1024 → 256)
7. **shape-match the residual** — if input width ≠ output width, pass residual through a Dense to match. (Happens only on the first block where input is 423-dim but features is 256.)
8. **`residual + x`** — add the skip connection. This is what makes it a ResNet block.

**Why expand 4× then shrink?** This is the standard Transformer feed-forward pattern. The wide middle layer gives expressive power; the skip connection makes it trainable at depth.

**Why the skip connection?** Gradients flow directly through the `+ residual` path, so even deep stacks train without vanishing gradients. Without it, 3+ stacked blocks would be very hard to optimize.

---

### 6.5 — MLPResNet (the full reverse network)

```python
class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)   # 423 → 256
        for _ in range(self.num_blocks):                                # 3 blocks
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
        x = self.activation(x)                                          # final swish
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)       # 256 → 7
        return x
```

- **`Dense(hidden_dim)`** — project the 423-dim concatenated input down to 256
- **loop `num_blocks` (3) times** — stack 3 `MLPResNetBlock`s
- **final `swish` + `Dense(out_dim)`** — project 256 → 7 (the predicted noise vector)

---

### 6.6 — The complete architecture (real)

```
ACTUAL SCORE NETWORK (ScoreActor):

time t  (shape: ..., 1)              obs_enc (384)        actions (7)
   │                                     │                    │
   ▼                                     │                    │
FourierFeatures (learnable)              │                    │
  w: learned (16,1)                      │                    │
  f = 2π·t·wᵀ                            │                    │
  [cos(f) | sin(f)]                      │                    │
   │ → 32-dim                            │                    │
   ▼                                     │                    │
MLP (cond_encoder)                       │                    │
  Dense(64) → swish                      │                    │
  Dense(32)                              │                    │
   │ → cond_enc (32)                     │                    │
   ▼                                     ▼                    ▼
   └──────────── concatenate [cond_enc | obs_enc | actions] ──┘
                          │ → 423-dim
                          ▼
                 MLPResNet (reverse_network)
                          │
                  Dense(423 → 256)
                          │
              ┌───────────▼───────────┐
              │  MLPResNetBlock 1     │
              │  LayerNorm            │
              │  Dense(256→1024)      │
              │  swish                │
              │  Dense(1024→256)      │
              │  + residual           │
              └───────────┬───────────┘
              ┌───────────▼───────────┐
              │  MLPResNetBlock 2     │  (same structure)
              └───────────┬───────────┘
              ┌───────────▼───────────┐
              │  MLPResNetBlock 3     │  (same structure)
              └───────────┬───────────┘
                          │
                        swish
                          │
                  Dense(256 → 7)
                          │
                          ▼
                  eps_pred (7)   ← predicted noise
```

**Correction to earlier description:** the time encoding is **learnable Fourier features**, not the classic fixed sinusoidal positional encoding. The idea is similar (map scalar → rich periodic vector) but here the frequencies are learned parameters, not hand-set. Also note the concatenation order is `[time | obs | action]`, and the residual blocks use a 4× width expansion internally.

---

## 7. Full Class Walkthrough — Every Line

```python
class DiffusionActionHead(nn.Module):
```
Flax Linen module. Inherits `nn.Module` — the base class for all Flax models.

---

```python
    readout_key: str
```
Which key to look up in `transformer_outputs`. Octo's transformer produces multiple token groups (obs_primary, obs_proprio, readout_action...). This tells the head which one to read. For the action head: `"readout_action"`.

---

```python
    use_map: bool = False
```
Two ways to go from `(batch, window, num_tokens, embed_dim)` → `(batch, window, embed_dim)`:
- `use_map=False`: mean pool across the token dimension (simple average)
- `use_map=True`: Multi-head Attention Pooling (MAPHead) — learns which tokens to attend to

For most setups, mean pooling is used.

---

```python
    action_horizon: int = 1
    action_dim: int = 7
```
`action_horizon`: how many consecutive actions to predict at once (action chunking).
`action_dim`: degrees of freedom of the robot arm (7 joints for most arms).

Total action vector size = `action_horizon × action_dim` = 7 (with horizon=1).

---

```python
    max_action: float = 5.0
```
Actions are clipped to [-5.0, 5.0] during training. Prevents extreme values from destabilizing the diffusion process. At inference, noisy actions are also clipped each step.

---

```python
    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.0
    hidden_dim: int = 256
    use_layer_norm: bool = True
    diffusion_steps: int = 20
    n_diffusion_samples: int = 1
```
These configure the score network MLP. `n_diffusion_samples=1` means one noise sample per training example (could use more for better gradient estimates but costs memory).

---

### setup()

```python
    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()
```
Only instantiate MAPHead if using attention pooling. Otherwise just mean-pool — no learned params needed.

---

```python
        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )
```
Creates the score network MLP. Output size = `action_dim × action_horizon` = 7. This MLP predicts the noise vector of the same size as the action.

---

```python
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.cumprod(self.alphas)
```
Precompute the noise schedule. These are fixed arrays — not learned parameters.
- `betas`: shape `(20,)` — noise variance at each step
- `alphas`: shape `(20,)` — signal retention at each step
- `alpha_hats`: shape `(20,)` — cumulative signal retention (used for one-shot noising)

---

### `__call__` — Forward Pass

```python
    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[ArrayLike] = None,
        noisy_actions: Optional[ArrayLike] = None,
        train: bool = True,
    ) -> jax.Array:
```

Three inputs:
1. `transformer_outputs` — the transformer's output dictionary
2. `time` — which diffusion step we're at
3. `noisy_actions` — the corrupted action at this diffusion step

---

```python
        token_group = transformer_outputs[self.readout_key]
```
Look up the readout token group by key. `token_group.tokens` has shape:
`(batch_size, window_size, num_tokens, embedding_size)`
= `(batch, 1, 1, 384)` for Octo-Small with window_size=1, 1 readout token.

---

```python
        assert token_group.tokens.ndim == 4
```
Safety check. The 4 dimensions are: batch, window, num_tokens, embed_dim. If anything upstream goes wrong and collapses a dimension, this catches it early.

---

```python
        if self.use_map:
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:
            embeddings = token_group.tokens.mean(axis=-2)
        # embeddings: (batch_size, window_size, embedding_size)
```
Pool tokens down to a single embedding vector per timestep.

`mean(axis=-2)` averages over the `num_tokens` dimension.
Result: `(batch, window, 384)` — one 384-dim vector per window step.

---

```python
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError("Must provide time and noisy_actions when calling diffusion action head")
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                dtype=jnp.float32,
            )
```
During Flax's `model.init()` call, we're "initializing" — just tracing shapes to allocate parameters. Real data hasn't arrived yet. So we substitute dummy zeros.

After initialization: `time` and `noisy_actions` must be provided — otherwise crash with a clear error.
	
This pattern is standard in Flax — always handle the init case explicitly.

---

```python
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
        return pred_eps
```
Feed everything into the MLP. Returns the predicted noise.

Output shape: same as `noisy_actions` = `(n_samples, batch, window, action_dim × action_horizon)` = `(1, batch, 1, 7)`

---

### `loss` — Training Objective

```python
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
```

Inputs:
- `actions`: ground truth clean actions, shape `(batch, window, action_horizon, action_dim)` = `(batch, 1, 1, 7)`
- `timestep_pad_mask`: `(batch, window)` bool — True if this timestep is real (not padding)
- `action_pad_mask`: same shape as actions — True if this action dimension is real (not masked for a different robot)

---

```python
        batch_size, window_size = timestep_pad_mask.shape
```
Extract batch and window size from the mask shape. Convenient way to get these without passing them explicitly.

---

```python
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
```
`rearrange` from `einops` library — reshapes tensors using named dimension notation.

`"b w h a -> b w (h a)"` means: merge the `h` (horizon) and `a` (action_dim) dimensions into one.

`(batch, window, action_horizon, action_dim)` → `(batch, window, action_horizon × action_dim)`
= `(batch, 1, 1, 7)` → `(batch, 1, 7)`

Why flatten? The score network takes a 1D action vector, not a 2D chunk. Flatten first, unflatten at the end.

---

```python
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)
```
Clip to [-5, 5]. Prevents any extreme demonstrated actions from dominating the noise schedule math. The diffusion process assumes roughly unit-scale signals — clipping keeps everything well-behaved.

---

```python
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
```
Get a random key. `make_rng("dropout")` piggybacks on the dropout RNG chain — a Flax convention for getting randomness inside a module without passing keys explicitly.

`jax.random.split(rng)` splits one key into two independent keys — one for sampling diffusion timestep, one for sampling noise. JAX has no global random state — you always split and pass keys explicitly.

---

```python
        time = jax.random.randint(
            time_key,
            (self.n_diffusion_samples, batch_size, window_size, 1),
            0,
            self.diffusion_steps,
        )
```
Sample a random diffusion timestep t for each example in the batch.

Shape: `(1, batch, window, 1)` — one random t per example.
Range: integer from 0 to 19 (exclusive of 20).

Why random? During training we don't denoise from t=20 to t=0. Instead we:
- Pick a random t
- Corrupt the action to that noise level
- Train the network to predict what noise was added

This trains the network for ALL noise levels simultaneously, which is much more efficient than always starting from scratch.

---

```python
        noise = jax.random.normal(
            noise_key, (self.n_diffusion_samples,) + actions_flat.shape
        )
```
Sample the actual Gaussian noise that will be added.

Shape: `(1, batch, window, 7)` — same shape as actions, but random N(0,I).

This is the ε that the network will be trained to predict.

---

```python
        scale = jnp.sqrt(self.alpha_hats[time])
        std   = jnp.sqrt(1 - self.alpha_hats[time])
        noisy_actions = scale * actions_flat[None] + std * noise
```
The forward diffusion formula:

```
a_t = sqrt(ᾱ_t) × a₀  +  sqrt(1 - ᾱ_t) × ε
```

`self.alpha_hats[time]` indexes into the precomputed array using the random timestep.

`actions_flat[None]` adds a dimension at the front to match `n_diffusion_samples`.

This creates the noisy action at level t — what the network will see during this training step.

---

```python
        pred_eps = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )
```
Run the forward pass — calls `__call__`. The network sees the transformer embedding + noisy action + timestep, and predicts the noise.

---

```python
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask
        mask = rearrange(mask, "b w h a -> b w (h a)")
        mask = mask[None]
```
Build a combined mask:
- `timestep_pad_mask`: which timesteps are real (not padding from the data loader)
- `action_pad_mask`: which action dimensions are real (not masked for a different robot)

The `&` combines them — a position is valid only if BOTH masks are True.

`rearrange` flattens to match `actions_flat` shape.
`[None]` adds `n_diffusion_samples` dimension.

Why masking? Octo is pretrained on data from 25 different robots. Different robots have different numbers of joints. A 6-DOF robot arm's data is zero-padded to 7 dimensions. The mask tells the loss "don't penalize predictions on padded dimensions."

---

```python
        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
```
Computes MSE between predicted noise and actual noise, applying the mask:

```
loss = mean over valid positions of ||pred_eps - noise||²
```

This is the DDPM training objective: predict the noise that was added, not the clean action directly. The reason for predicting noise rather than the clean action is a training stability trick — the noise prediction problem is the same difficulty at every timestep, while predicting the clean action directly is much harder at high noise levels.

---

```python
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics
```
Scale by `action_dim`. This makes the loss roughly independent of action dimension count — useful when comparing runs with different action spaces. Cosmetic but important for consistent logging.

---

### `predict_action` — Inference

This is the full 20-step DDPM reverse process.

```python
    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        embodiment_action_dim: Optional[int] = None,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
```

`embodiment_action_dim`: if provided, only evaluate the first N action dimensions (for robots with fewer than 7 DOF). Remaining dimensions are filled with training-compatible noise.

---

```python
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        module, variables = self.unbind()
```
`self.unbind()` separates the module definition from its variables (params). This is needed because inside `jax.lax.scan` we call the module explicitly with `module.apply(variables, ...)` — we can't use `self(...)` inside a scan.

---

```python
        action_mask = jnp.ones(
            (*sample_shape, batch_size, window_size, self.action_horizon, self.action_dim),
            dtype=bool,
        )
        if embodiment_action_dim is not None:
            action_mask = action_mask.at[..., embodiment_action_dim:].set(False)
        flat_action_mask = rearrange(action_mask, "... p a -> ... (p a)")
```
Build the action mask. If `embodiment_action_dim=6`, mask out dimension 6 (the 7th joint that this robot doesn't have).

---

```python
        def scan_fn(carry, time):
            current_x, rng = carry
            input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))
```
`scan_fn` is called once per diffusion step. `current_x` = current noisy action. `time` = current step (counts down from 19 to 0).

`broadcast_to` expands the scalar timestep to match the batch shape.

---

```python
            eps_pred = module.apply(
                variables, transformer_outputs, input_time, current_x, train=train
            )
```
Run the score network: given current noisy action and timestep, predict the noise.

---

```python
            alpha_1 = 1 / jnp.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)
```
DDPM reverse step formula:
```
a_{t-1} = (1/sqrt(α_t)) × (a_t  -  ((1-α_t)/sqrt(1-ᾱ_t)) × ε_pred)
```

- `alpha_1 = 1/sqrt(α_t)` — rescaling factor
- `alpha_2 × eps_pred` — removes the predicted noise component
- Result: one step closer to clean action

---

```python
            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)
```
Add stochastic noise. `(time > 0)` = 1 for all steps except the final one (t=0). At t=0, no noise is added — we want a clean deterministic output.

This stochasticity is what allows the model to generate different valid actions each call.

---

```python
            current_x = jnp.clip(current_x, -self.max_action, self.max_action)
```
Clip at each step. Prevents the reverse process from wandering to extreme values during intermediate steps.

---

```python
            current_x = jnp.where(
                flat_action_mask, current_x, jnp.sqrt(1 - self.alpha_hats[time]) * z
            )
            return (current_x, rng), ()
```
For masked action dimensions (robot has fewer DOF): fill with noise consistent with the training distribution rather than leaving them as garbage values.

Returns `(new_carry, output)`. The `()` means no output per step — we only care about the final carry.

---

```python
        rng, key = jax.random.split(rng)
        noise = jax.random.normal(
            key,
            (*sample_shape, batch_size, window_size, self.action_horizon * self.action_dim),
        )

        (actions_flat, _), () = jax.lax.scan(
            scan_fn,
            (noise, rng),
            jnp.arange(self.diffusion_steps - 1, -1, -1),
        )
```
Start from pure Gaussian noise.

`jax.lax.scan(fn, init, xs)`:
- `fn` = `scan_fn` (one denoising step)
- `init` = `(noise, rng)` — starting state
- `xs` = `[19, 18, 17, ..., 1, 0]` — timesteps to iterate over

Runs `scan_fn` 20 times, each time passing the previous output as carry. Equivalent to:
```python
carry = (noise, rng)
for t in [19, 18, ..., 0]:
    carry, _ = scan_fn(carry, t)
actions_flat, _ = carry
```

But compiled into a single efficient XLA operation.

---

```python
        actions = rearrange(
            actions_flat,
            "... (h a) -> ... h a",
            h=self.action_horizon,
            a=self.action_dim,
        )
        return actions[..., -1, :, :]
```
Unflatten: `(batch, window, 7)` → `(batch, window, action_horizon, action_dim)`.

`[..., -1, :, :]` — take only the LAST window timestep. The model processes a window of observations, but only the most recent timestep's action is actually executed. This is the current action to send to the robot.

Final output shape: `(batch, action_horizon, action_dim)` = `(batch, 1, 7)`

---

## 8. The Full Data Flow

```
TRAINING:

Robot demonstration data
        |
        | actions: (batch, window, horizon, action_dim) = (B, 1, 1, 7)
        |
        ↓
[FORWARD DIFFUSION]
        |
        | sample random t ∈ {0,...,19}
        | sample noise ε ~ N(0, I)
        | noisy_action = sqrt(ᾱ_t) × action + sqrt(1-ᾱ_t) × ε
        |
        ↓
[TRANSFORMER]
images + language → tokens → block transformer → readout tokens
        |
        | readout token embeddings: (B, window, 384)
        |
        ↓
[SCORE NETWORK (diffusion MLP)]
inputs: embeddings + noisy_action + t
        |
        | pred_ε: (B, window, 7)
        |
        ↓
[LOSS]
MSE(pred_ε, ε)
        |
        ↓
gradients → update score network weights only
(transformer frozen during finetuning)
```

---

```
INFERENCE:

Current camera image + language instruction
        |
        ↓
[TRANSFORMER] (frozen)
        |
        | readout embeddings: (1, 1, 384)
        |
        ↓
[REVERSE DIFFUSION LOOP — 20 steps]

step 19: a₁₉ = pure noise
         ε_pred = score_network(embeddings, a₁₉, t=19)
         a₁₈ = DDPM_step(a₁₉, ε_pred, t=19)

step 18: ε_pred = score_network(embeddings, a₁₈, t=18)
         a₁₇ = DDPM_step(a₁₈, ε_pred, t=18)

... (20 total steps)

step 0:  ε_pred = score_network(embeddings, a₁, t=0)
         a₀ = DDPM_step(a₁, ε_pred, t=0)   ← no noise added at final step
        |
        ↓
clean action a₀: (1, 1, 7)
        |
        ↓
send to robot → move joints
```

---

## 9. How It Connects To Octo's Transformer

```
FULL OCTO ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│  INPUT TOKENIZERS                                       │
│                                                         │
│  Camera images  → CNN → image patches → obs tokens     │
│  Language text  → T5  → language tokens → task tokens  │
│  (Goal image    → CNN → image patches → goal tokens)   │
└─────────────────────────────────────────────────────────┘
                          |
                          ↓
┌─────────────────────────────────────────────────────────┐
│  TRANSFORMER BACKBONE                                   │
│                                                         │
│  All tokens + readout tokens go through transformer     │
│  Block-wise causal attention mask                       │
│                                                         │
│  Readout tokens attend to everything but nothing        │
│  attends back to them (like CLS token in BERT)         │
│                                                         │
│  Output: readout_action token embeddings               │
│  Shape: (batch, window_size, 1, 384)                    │
└─────────────────────────────────────────────────────────┘
                          |
                          | transformer_outputs["readout_action"]
                          ↓
┌─────────────────────────────────────────────────────────┐
│  DIFFUSION ACTION HEAD                                  │
│                                                         │
│  Mean pool readout tokens → embeddings (B, W, 384)     │
│                                                         │
│  TRAINING:                                              │
│    corrupt actions with noise → noisy_actions          │
│    score_network(embeddings, noisy_actions, t) → ε_pred │
│    loss = MSE(ε_pred, ε)                               │
│                                                         │
│  INFERENCE:                                             │
│    start from noise → 20 denoising steps → actions     │
└─────────────────────────────────────────────────────────┘
                          |
                          ↓
              actions: (batch, action_horizon, action_dim)
              = (1, 1, 7) joint angles to move
```

---

## 10. What Your Flow Head Replaces

The flow head is a **drop-in replacement** for the diffusion head. Same interface, different internals.

### What stays the same (interface)

```python
class FlowActionHead(nn.Module):
    # Same config fields
    readout_key: str
    use_map: bool = False
    action_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0

    def __call__(self, transformer_outputs, time, noisy_actions, train):
        # Same inputs, same output shape
        ...
        return pred_velocity  # shape: (n_samples, batch, window, action_dim*action_horizon)

    def loss(self, transformer_outputs, actions, timestep_pad_mask, action_pad_mask, train):
        # Same signature
        ...
        return loss, metrics

    def predict_action(self, transformer_outputs, rng, train, embodiment_action_dim, sample_shape):
        # Same signature, same output shape
        ...
        return actions  # (batch, action_horizon, action_dim)
```

### What changes (internals)

| Component         | Diffusion Head                | Flow Head                    |
| ----------------- | ----------------------------- | ---------------------------- |
| Time              | integer t ∈ {0,...,19}        | continuous τ ∈ [0, 1]        |
| Forward process   | `sqrt(ᾱ_t)×a + sqrt(1-ᾱ_t)×ε` | `τ×a + (1-τ)×ε`              |
| Network predicts  | noise `ε`                     | velocity `v = a - ε`         |
| Loss              | `MSE(pred_ε, ε)`              | `MSE(pred_v, a - ε)`         |
| Inference steps   | 20 DDPM reverse steps         | 10 Euler steps               |
| Inference formula | complex DDPM formula          | `a_{τ+δ} = a_τ + δ × v_pred` |

### Flow head forward process

Instead of the complex DDPM noising:
```
noisy_action = sqrt(ᾱ_t) × action + sqrt(1-ᾱ_t) × ε
```

Flow matching uses a simple linear interpolation:
```
noisy_action = τ × action + (1-τ) × ε
```

When τ=0: pure noise ε
When τ=1: clean action

### Flow head loss

Instead of predicting noise:
```python
# DDPM: predict noise
loss = MSE(pred_eps, noise)
```

Flow matching predicts velocity (direction from noise to action):
```python
# CFM: predict velocity
target_velocity = action - noise  # points from noise toward action
loss = MSE(pred_velocity, target_velocity)
```

### Flow head inference

Instead of 20 DDPM steps:
```python
# Flow matching: simple Euler integration
a = noise  # start from noise, τ=0
for τ in [0.0, 0.1, 0.2, ..., 0.9]:  # 10 steps
    v = flow_model(embeddings, a, τ)   # predict velocity
    a = a + 0.1 × v                    # Euler step
# a is now the clean action at τ=1
```

Much simpler. 10 steps instead of 20. Linear path instead of complex DDPM schedule.

---

## Summary — The One Paragraph Version

The `DiffusionActionHead` works by learning to reverse a noise-adding process. During training, it takes a clean action, corrupts it to a random noise level using a cosine schedule, and trains an MLP (the score network) to predict what noise was added — given the corrupted action, the noise level, and a context embedding from the transformer. At inference, it starts from pure Gaussian noise and runs the learned denoising process 20 times in reverse, each step removing a little noise guided by the score network, until a clean action emerges. The multimodality comes naturally: different starting noises lead to different clean actions, all drawn from the learned distribution of valid robot behaviors.

Your flow head does the same job — takes transformer embeddings and produces actions — but replaces DDPM with conditional flow matching: simpler math, same interface, fewer inference steps.
