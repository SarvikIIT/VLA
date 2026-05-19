#!/usr/bin/env python3
"""
Smoke test for the FPO pipeline — runs BC + FPO loop end-to-end with
fully synthetic data. No MetaWorld, no GPU required.

Verifies:
  1. fpo.py primitives (cfm_loss_from_pred, fpo_ratio_from_diff, ppo_clip_loss)
  2. FPOActionHead staticmethod delegation
  3. BC pre-training loop
  4. FPO fine-tuning loop (compute_advantages + combined_step)
  5. Diffusion and Flow heads still work (no regression)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from octo.model.components.fpo import cfm_loss_from_pred, fpo_ratio_from_diff, ppo_clip_loss
from octo.model.components.action_heads import (
    FPOActionHead, SimpleFlowMatchActionHead, DiffusionActionHead,
)
from octo.model.components.base import TokenGroup

# ── constants (match fpo.py) ──────────────────────────────────────────
OBS_DIM      = 39
ACT_DIM      = 4
ACTION_HORIZON = 1
WINDOW       = 1
HIDDEN       = 64   # tiny for speed
BLOCKS       = 2
B            = 16   # batch size
N_DEMO       = 64   # fake demo transitions
N_ROLLOUT    = 40   # fake rollout transitions

rng = jax.random.PRNGKey(0)


def make_t_out(obs_batch):
    tokens = obs_batch[:, None, None, :]
    mask   = jnp.ones(tokens.shape[:3], dtype=bool)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


def init_head(head):
    dummy = make_t_out(jnp.zeros((2, OBS_DIM)))
    r, dr = jax.random.split(rng)
    return head.init({"params": r, "dropout": dr}, dummy)["params"]


# ── 1. fpo.py primitives ──────────────────────────────────────────────
print("1. Testing fpo.py primitives ...")

v_pred   = jnp.array(np.random.randn(B, ACT_DIM).astype("float32"))
v_target = jnp.array(np.random.randn(B, ACT_DIM).astype("float32"))

loss_vec = cfm_loss_from_pred(v_pred, v_target)
assert loss_vec.shape == (B,), f"cfm_loss shape {loss_vec.shape}"

ratio = fpo_ratio_from_diff(jnp.array(0.3))
assert abs(float(ratio) - float(jnp.exp(0.3))) < 1e-5, "ratio mismatch"

clip_loss, metrics = ppo_clip_loss(jnp.ones(B), jnp.array(np.random.randn(B).astype("float32")), 0.05)
assert set(metrics.keys()) == {"fpo_loss", "ratio_mean", "ratio_std"}, "metrics keys"

print("   fpo.py primitives OK")


# ── 2. FPOActionHead staticmethod delegation ─────────────────────────
print("2. Testing FPOActionHead static methods ...")

r1 = fpo_ratio_from_diff(jnp.array(0.5))
r2 = FPOActionHead.fpo_ratio_from_diff(jnp.array(0.5))
assert abs(float(r1) - float(r2)) < 1e-6, "fpo_ratio_from_diff mismatch"

adv = jnp.array(np.random.randn(B).astype("float32"))
rat = jnp.ones(B) * 1.05
l1, m1 = ppo_clip_loss(rat, adv, 0.05)
l2, m2 = FPOActionHead.ppo_clip_loss(rat, adv, 0.05)
assert abs(float(l1) - float(l2)) < 1e-6, "ppo_clip_loss mismatch"

print("   FPOActionHead static methods OK")


# ── 3. BC pre-training (all 3 heads) ─────────────────────────────────
print("3. Testing BC pre-training ...")

obs_data = jnp.array(np.random.randn(N_DEMO, OBS_DIM).astype("float32"))
act_data = jnp.array(np.random.randn(N_DEMO, ACT_DIM).astype("float32"))
act_data_4d = act_data[:, None, None, :]

tp_mask = jnp.ones((B, WINDOW), dtype=bool)
ap_mask = jnp.ones((B, WINDOW, ACTION_HORIZON, ACT_DIM), dtype=bool)

def run_bc(head_cls, head_kwargs, n_steps=3):
    head   = head_cls(**head_kwargs)
    params = init_head(head)
    tx     = optax.adam(3e-4)
    opt_st = tx.init(params)
    r      = jax.random.PRNGKey(1)

    for _ in range(n_steps):
        idx       = np.random.randint(0, N_DEMO, B)
        obs_b     = obs_data[idx]
        act_b     = act_data_4d[idx]
        t_out     = make_t_out(obs_b)
        r, step_r = jax.random.split(r)

        def loss_fn(p):
            bound = head.bind({"params": p}, rngs={"dropout": step_r})
            loss, _ = bound.loss(t_out, act_b, tp_mask, ap_mask, train=True)
            return loss

        grads      = jax.grad(loss_fn)(params)
        updates, opt_st = tx.update(grads, opt_st, params)
        params     = optax.apply_updates(params, updates)

    return head, params

shared = dict(readout_key="obs", action_horizon=ACTION_HORIZON, action_dim=ACT_DIM,
              hidden_dim=HIDDEN, num_blocks=BLOCKS, time_dim=32)

diff_head,  diff_params  = run_bc(DiffusionActionHead,       {**shared, "diffusion_steps": 5})
flow_head,  flow_params  = run_bc(SimpleFlowMatchActionHead, {**shared, "flow_steps": 3})
fpo_head,   fpo_params   = run_bc(FPOActionHead,             {**shared, "flow_steps": 3})

print("   BC pre-training OK (diffusion, flow, fpo)")


# ── 4. FPO fine-tuning loop ───────────────────────────────────────────
print("4. Testing FPO fine-tuning loop ...")

# fake rollout data
rollout_obs  = np.random.randn(N_ROLLOUT, OBS_DIM).astype("float32")
rollout_acts = np.random.randn(N_ROLLOUT, ACT_DIM).astype("float32")
rollout_rew  = np.zeros(N_ROLLOUT, dtype="float32")
rollout_rew[-1] = 1.0   # single terminal reward (like MetaWorld)
rollout_done = np.zeros(N_ROLLOUT, dtype=bool)
rollout_done[-1] = True

# compute_advantages (copied logic from fpo.py)
def compute_advantages(rewards, dones, gamma=0.99):
    N = len(rewards)
    returns = np.zeros(N, dtype=np.float32)
    running = 0.0
    for t in reversed(range(N)):
        if dones[t]: running = 0.0
        running = rewards[t] + gamma * running
        returns[t] = running
    adv = (returns - returns.mean()) / (returns.std() + 1e-8)
    return adv, returns

advantages, _ = compute_advantages(rollout_rew, rollout_done)

obs_jax  = jnp.array(rollout_obs)
act_jax  = jnp.array(rollout_acts)
adv_jax  = jnp.array(advantages)

# make_fpo_functions (inline — same logic as fpo.py)
def make_fpo_loss(head, n_mc=2, clip_eps=0.05):
    def fpo_loss_fn(params, params_old, obs_b, act_b, adv_b, rng):
        t_out = make_t_out(obs_b)
        diffs = []
        for _ in range(n_mc):
            rng, r1, r2 = jax.random.split(rng, 3)
            taus = jax.random.uniform(r1, (len(obs_b),))
            epss = jax.random.normal(r2, act_b.shape)
            l_new = head.apply({"params": params},     t_out, act_b, taus, epss,
                               method=FPOActionHead.cfm_loss_per_sample,
                               rngs={"dropout": rng})
            l_old = head.apply({"params": params_old}, t_out, act_b, taus, epss,
                               method=FPOActionHead.cfm_loss_per_sample,
                               rngs={"dropout": rng})
            diffs.append(l_old - l_new)
        ratio = FPOActionHead.fpo_ratio_from_diff(jnp.stack(diffs).mean(0))
        return FPOActionHead.ppo_clip_loss(ratio, adv_b, clip_eps)
    return fpo_loss_fn

fpo_loss_fn = make_fpo_loss(fpo_head, n_mc=2, clip_eps=0.05)
tx_fpo  = optax.adam(3e-5)
opt_fpo = tx_fpo.init(fpo_params)
r = jax.random.PRNGKey(2)

batch = min(16, N_ROLLOUT)
for fpo_iter in range(2):
    params_old = jax.tree.map(lambda x: x.copy(), fpo_params)
    idx = np.random.randint(0, N_ROLLOUT, batch)

    def total_loss(p):
        fl, fm = fpo_loss_fn(p, params_old,
                             obs_jax[idx], act_jax[idx], adv_jax[idx], r)
        return fl, fm

    (loss, mets), grads = jax.value_and_grad(total_loss, has_aux=True)(fpo_params)
    updates, opt_fpo  = tx_fpo.update(grads, opt_fpo, fpo_params)
    fpo_params        = optax.apply_updates(fpo_params, updates)
    print(f"   FPO iter {fpo_iter+1}: loss={float(loss):.4f} "
          f"ratio_mean={float(mets['ratio_mean']):.4f}")

print("   FPO fine-tuning loop OK")


# ── 5. predict_action (inference unchanged) ───────────────────────────
print("5. Testing predict_action (inference) ...")

dummy_obs = jnp.zeros((1, OBS_DIM))
t_out     = make_t_out(dummy_obs)
inf_rng   = jax.random.PRNGKey(3)

for name, head, params in [("diffusion", diff_head, diff_params),
                            ("flow",      flow_head, flow_params),
                            ("fpo",       fpo_head,  fpo_params)]:
    bound = head.bind({"params": params}, rngs={"dropout": inf_rng})
    act   = bound.predict_action(t_out, rng=inf_rng, train=False)
    assert act.shape[-1] == ACT_DIM, f"{name} action dim mismatch"
    print(f"   {name} predict_action shape: {act.shape} OK")

print("\n✓ ALL SMOKE TESTS PASSED")
