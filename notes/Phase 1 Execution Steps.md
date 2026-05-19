___
# Phase 1 — Execution Steps (Colab Workflow)

Companion to `Phase 1 Study Plan.md`. This file is the *do-this-now* checklist. Run top to bottom.

---

## Setup Decision (already made)

- **Compute:** Google Colab (free tier to start, Pro $10/mo if needed)
- **Edit:** VS Code locally for reading/editing Octo code
- **Run:** Copy-paste cells into Colab
- **Persistence:** Google Drive mount — everything lives in `/content/drive/MyDrive/octo`

Why: JAX is painful on Windows. Colab has JAX preinstalled with GPU. Zero infra setup.

---

## Tonight — Goal: Stock Octo Example Steps on Colab

**Done when:** the stock finetune script takes a few gradient steps without crashing. Nothing modified yet.

### Step 1 — Open Colab + Enable GPU
- Go to colab.research.google.com → New Notebook
- Runtime → Change runtime type → T4 GPU → Save

### Step 2 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3 — Confirm GPU is attached
```python
!nvidia-smi
```
Should show a Tesla T4 with ~15GB free. If not → re-do Step 1.

### Step 4 — One-time: Clone Octo into Drive
Only run this the FIRST session ever. After that it persists in Drive.
```python
%cd /content/drive/MyDrive
!git clone https://github.com/octo-models/octo.git
```

### Step 5 — Install Octo dependencies (every session)
Colab VMs are fresh — install runs each session.
```python
%cd /content/drive/MyDrive/octo
!pip install -e . --quiet
!pip install -r requirements.txt --quiet
```

### Step 6 — Verify JAX sees GPU
```python
import jax
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())
```
Want: `[CudaDevice(id=0)]`. If `CpuDevice` → paste output for debugging.

### Step 7 — Read Octo's README
```python
!cat /content/drive/MyDrive/octo/README.md
```
Find the finetuning example command. Paste it back if unclear.

### Step 8 — Run the stock finetune example
Probably something like:
```python
%cd /content/drive/MyDrive/octo
!python scripts/finetune.py --config=scripts/configs/finetune_config.py --name=test_run
```
**Adjust path to match what the README says.** If batch size errors (OOM), drop it in the config.

**Tonight stops here.** If Step 8 takes gradient steps → big win. Sleep.

---

## Week 1 — Goal: Map Octo's Action Head

**Done when:** you can state the diffusion head's input/output tensor shapes exactly.

### Step 9 — Locate the head module
```python
%cd /content/drive/MyDrive/octo
!grep -rn "DiffusionActionHead\|diffusion_action\|DDPM" --include="*.py"
```
This lists files defining the diffusion head. Open them in VS Code locally (read from Drive or your git clone).

### Step 10 — Document the head's interface
Open the head class. Write down in a scratch file:
- **Inputs:** readout embedding `e` (shape?), noisy action `a_k` (shape?), timestep `k` (shape?)
- **Outputs:** predicted noise epsilon (shape?)
- **Where it's instantiated** (which file imports it into the main model)
- **Where the loss is computed** (which file calls it during training)

This is G3 from the study plan. Your flow-matching head must have the same interface to be a drop-in replacement.

### Step 11 — Reproduce a baseline number
Pick whatever sim benchmark Octo supports out-of-the-box (check the repo for an `eval_*.py` script). Run it once with the stock model. Record the success rate number. **This is M0** — the control number you'll compare your flow-matching head against later.

---

## Week 2 — Parallel: JAX/Flax + CFM Toy

Do these locally OR in Colab — small enough either way.

### Step 12 — JAX/Flax tutorial (G1)
Train a tiny Flax MLP on MNIST (or any toy). Goal: get comfortable with `jit`, `grad`, `nn.Module`, `TrainState`, Optax optimizer. ~half a day.

### Step 13 — CFM loss on a 2D toy (G2)
Implement Conditional Flow Matching on a 2D Gaussian mixture target:
- Sample `t ~ U[0,1]`, source noise `eps ~ N(0,I)`, target `a` from the mixture
- Interpolate `a_t = (1-t)*eps + t*a`
- Target velocity `u = a - eps` (action minus noise — verify sign here!)
- Train: `loss = ||v_theta(a_t, t) - u||^2`
- Inference: start from noise, Euler step `a_hat = eps + v_theta(eps, t=0)` (or run 10 steps)
- Plot inference samples vs target. They should match.

**This is the sign-convention verification step.** If samples don't match the target, your sign is wrong. Fix here, not in Octo.

---

## Week 3+ — Build the Flow Head (M1)

Only start after Steps 10, 11, 13 are done.

### Step 14 — Build the flow head module
- Same I/O interface as the diffusion head (Step 10 notes)
- Internals: tau-conditioning MLP from Pi-0 Appendix B
  - `W3 · swish(W2 · concat(W1 · a_tau, phi(tau)))` where phi = sinusoidal positional encoding
- Output: velocity `v` (same shape as action)

### Step 15 — Swap the loss
Replace DDPM loss call with CFM loss in the training script. Verify it converges on a tiny subset first (10 demos, overfit).

### Step 16 — Finetune on ~100 demos
Same recipe Octo uses: transformer frozen, 50k steps, batch 16–256 (whatever Colab fits).

### Step 17 — Benchmark (M3)
Same eval as Step 11. Compare:
- Diffusion head: latency per action, success rate
- Flow head: latency per action, success rate

**This is Phase 1 done.**

---

## Stop Criteria Per Session

After each Colab session:
- ✅ Made progress on the current step → checkpoint to Drive, note what you did
- ❌ Stuck on infra (install errors, OOM, JAX bugs) → paste error into chat, debug here
- ❌ Stuck on concept → flag it, ask before grinding more

---

## Things That Will Bite You

1. **Drive disconnect mid-training** — checkpoint every N steps to `/content/drive/MyDrive/octo/checkpoints/`. Verify save path is Drive, not `/content/`.
2. **Colab 12h limit** — long runs need resumable checkpoints.
3. **Batch size OOM on T4** — drop it. Octo-Small at batch 16 is fine for proof-of-life.
4. **JAX caching** — first run after install is slow (XLA compile). Be patient.
5. **Octo repo drift** — file paths and configs may have changed since the paper. Trust the README, not external blog posts.

---

## When to Switch Off Colab

You probably won't, but:
- M2 finetuning takes >12h per run, even with Pro → consider Kaggle (30h/week) or paid cloud (Lambda, vast.ai)
- Need step-through debugger → set up WSL2 + 4050 locally as secondary

---

# What Each Step Actually Does (Plain English)

## Step 1 — Open Colab + Enable GPU
Colab gives you a free remote VM. Default is CPU-only (slow). Menu request switches you to a Tesla T4 (16GB VRAM). Without GPU enabled, training takes forever.

## Step 2 — Mount Drive
Colab VM is **temporary** — wiped when session ends. "Mounting Drive" tells the VM "treat my Google Drive as a folder at `/content/drive/MyDrive/`." Anything you save there survives session death.

Analogy: Colab VM = hotel room (wiped on checkout). Drive = your suitcase (you take it with you).

## Step 3 — Confirm GPU is attached
`nvidia-smi` prints info about attached NVIDIA GPUs. If you see "Tesla T4 — 15GB free," Step 1 worked. Sanity check that catches silent GPU-not-attached failures.

## Step 4 — Clone Octo into Drive
`git clone` downloads Octo's source code from GitHub into Drive (persistent), not the VM (wiped). One-time only — next session the code is still there.

## Step 5 — Install dependencies
Code persists in Drive, but **installed Python packages** live in the VM and get wiped. So re-install each session.
- `pip install -e .` = install Octo in editable mode (edits take effect immediately, no reinstall)
- `pip install -r requirements.txt` = install JAX, Flax, Optax, all dependencies
- `--quiet` suppresses the install spam

## Step 6 — Verify JAX sees GPU
JAX is the deep learning framework Octo uses (like PyTorch but functional). JAX can run on CPU or GPU. We check which.

Want: `[CudaDevice(id=0)]` — GPU works, training fast.
Bad: `[CpuDevice(id=0)]` — JAX didn't find CUDA, training 50× slower.

This is sanity check #2 — separate from `nvidia-smi` because JAX could fail to see a GPU that's actually attached.

## Step 7 — Read Octo's README
README tells you the *current* commands — file paths, configs, syntax. Repos drift; docs are the source of truth. `cat` prints the README inside Colab so you don't switch tabs. Look for "Finetuning" or "Quickstart" — gives the command for Step 8.

## Step 8 — Run the stock finetune example
The actual tonight goal. Runs Octo's training script with their example config. If you see `step 1: loss=...`, `step 2: loss=...` — **success**.

Why it matters: confirms JAX + GPU + Octo all play nicely on Colab. Every future change builds on this confirmed baseline. If something breaks later, you know it's YOUR change, not infra.

## Step 9 — Locate the head module
`grep -rn "DiffusionActionHead..."` searches every Python file for those keywords, returns file paths + line numbers. Octo has hundreds of files; manual searching wastes time.

## Step 10 — Document the head's interface
"Interface" = what goes in, what comes out, in what tensor shapes. For Octo's diffusion head:
- Input: embedding `e` (shape `[batch, embed_dim]`), noisy action (shape `[batch, action_dim]`), timestep (shape `[batch]`)
- Output: predicted noise (shape `[batch, action_dim]`)

Your flow head must accept the **same inputs** and produce **same-shaped outputs** to drop in cleanly. Different shape → rest of Octo won't connect.

## Step 11 — Reproduce a baseline number
Run Octo's eval script on the stock model on a sim benchmark. Record success rate. This is your **control number** (M0). Later you'll re-run with the flow head and compare:
- Diffusion: 80% success, 100ms/action
- Flow: 78% success, 10ms/action

That comparison IS Phase 1's deliverable. Can't have it without first knowing the baseline.

## Step 12 — JAX/Flax tutorial
Train a tiny MLP on toy data. Learn the primitives:
- `jit` — just-in-time compilation (makes JAX fast)
- `grad` — automatic differentiation
- `nn.Module` — Flax's model class
- `TrainState` — Flax's optimizer state tracker

Can't modify Octo without these. Skip this and you're lost the moment you try to write a new module.

## Step 13 — CFM loss on a 2D toy
Build flow matching from scratch on a fake 2D problem:
- Target: mixture of 2D Gaussians (8 dots in a circle)
- Noise: standard 2D Gaussian (blob at origin)
- Train tiny MLP to predict velocity field mapping blob → 8 dots
- Inference: start at noise, follow velocity, land on 8 dots

Why a toy?
1. **Visualize the result** — if sign is wrong, samples land in wrong place. Instant feedback.
2. **Fast** — trains in seconds. Debug the algorithm without Octo training delays.

This is the **sign-convention check** — verify `u = a - eps` is right on a problem with known ground truth before touching Octo.

## Step 14 — Build the flow head module
Write the actual flow-matching head in Octo's codebase. Flax module class. Inside:
- Take inputs matching diffusion head's interface (Step 10)
- Tau-conditioning MLP from Pi-0 Appendix B: sinusoidal-encode the timestep, concat with action, 2-layer MLP
- Output velocity vector (same shape as action)

## Step 15 — Swap the loss
Find where Octo computes DDPM loss (Step 9 found this). Replace with CFM loss. Test on 10 demos first — if loss drops and model overfits, wiring is correct.

## Step 16 — Finetune on ~100 demos
Full recipe: 50k steps, transformer frozen, only the new head trains. Real training run — hours.

## Step 17 — Benchmark
Same eval as Step 11, now with flow head. Build comparison table. **Phase 1 is done.**
