#!/usr/bin/env python3
"""
Comprehensive smoke test + overfit comparison for all action heads:
  - DiffusionActionHead        (Vanilla Octo — DDPM diffusion)
  - SimpleFlowMatchActionHead  (Simple flow matching, no OT)
  - FlowMatchActionHead        (Flow matching + Conditional Optimal Transport)

Tests per head:
  1. init           – model initializes, params > 0
  2. loss           – loss is finite scalar with correct keys
  3. predict_action – output shape (batch, action_horizon, action_dim)
  4. sample_shape   – sample_shape prefix is correctly prepended
  5. gradient_flow  – all parameter tensors receive non-zero gradients
  6. pad_mask       – masking a timestep changes the loss
  7. stochasticity  – different RNG keys give different outputs

Final comparison table: initial loss, final loss, loss ratio, action MSE.

Usage:
    python3 tests/smoke_test_all_heads.py
"""

import sys, os, time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import optax

from octo.model.components.action_heads import (
    DiffusionActionHead,
    SimpleFlowMatchActionHead,
    FlowMatchActionHead,
    SinkhornFlowMatchActionHead,
    ConditionedOTFlowMatchActionHead,
)
from octo.model.components.base import TokenGroup

# ── constants ─────────────────────────────────────────────────────────
BATCH = 8
WINDOW = 2
NUM_TOKENS = 4
EMB_DIM = 64
ACTION_HORIZON = 1
ACTION_DIM = 7
OVERFIT_STEPS = 500
OVERFIT_LR = 3e-4


# ── helpers ───────────────────────────────────────────────────────────

def make_transformer_outputs(batch=BATCH, window=WINDOW):
    tokens = jax.random.normal(jax.random.PRNGKey(99), (batch, window, NUM_TOKENS, EMB_DIM))
    mask = jnp.ones((batch, window, NUM_TOKENS), dtype=bool)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


def make_actions(batch=BATCH, window=WINDOW):
    return jax.random.normal(jax.random.PRNGKey(42), (batch, window, ACTION_HORIZON, ACTION_DIM))


def make_masks(batch=BATCH, window=WINDOW):
    timestep_pad_mask = jnp.ones((batch, window), dtype=bool)
    action_pad_mask = jnp.ones((batch, window, ACTION_HORIZON, ACTION_DIM), dtype=bool)
    return timestep_pad_mask, action_pad_mask


# ── head factory ──────────────────────────────────────────────────────

HEAD_CONFIGS = {
    "Vanilla Octo (Diffusion)": lambda: DiffusionActionHead(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        time_dim=32,
        num_blocks=3,
        dropout_rate=0.0,
        hidden_dim=128,
        use_layer_norm=True,
        diffusion_steps=20,
        n_diffusion_samples=1,
    ),
    "Simple Flow": lambda: SimpleFlowMatchActionHead(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        time_dim=32,
        num_blocks=3,
        dropout_rate=0.0,
        hidden_dim=128,
        use_layer_norm=True,
        flow_steps=10,
        n_flow_samples=1,
    ),
    "Flow + COT": lambda: FlowMatchActionHead(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        time_dim=32,
        num_blocks=3,
        dropout_rate=0.0,
        hidden_dim=128,
        use_layer_norm=True,
        flow_steps=10,
        n_flow_samples=1,
    ),
    "Sinkhorn Flow": lambda: SinkhornFlowMatchActionHead(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        time_dim=32,
        num_blocks=3,
        dropout_rate=0.0,
        hidden_dim=128,
        use_layer_norm=True,
        flow_steps=10,
        n_flow_samples=1,
        sinkhorn_reg=0.1,
        sinkhorn_iters=20,
    ),
    "Conditioned OT": lambda: ConditionedOTFlowMatchActionHead(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        time_dim=32,
        num_blocks=3,
        dropout_rate=0.0,
        hidden_dim=128,
        use_layer_norm=True,
        flow_steps=10,
        n_flow_samples=2,
        sinkhorn_reg=0.05,
        sinkhorn_iters=20,
        obs_weight=1.0,
        history_weight=0.5,
        use_history=True,
    ),
}


def init_head(name):
    head = HEAD_CONFIGS[name]()
    rng = jax.random.PRNGKey(0)
    params = head.init({"params": rng, "dropout": rng}, make_transformer_outputs())
    return head, params


# ══════════════════════════════════════════════════════════════════════
#  SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════

def test_init(name):
    head, params = init_head(name)
    count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert count > 0, "No parameters"
    return f"params={count:,}"


def test_loss(name):
    head, params = init_head(name)
    rng = jax.random.PRNGKey(1)
    loss, metrics = head.apply(
        params, make_transformer_outputs(), make_actions(),
        *make_masks(), train=True, rngs={"dropout": rng}, method=head.loss,
    )
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert jnp.isfinite(loss), f"Loss not finite: {loss}"
    assert "loss" in metrics and "mse" in metrics
    return f"loss={float(loss):.4f}"


def test_predict_action(name):
    head, params = init_head(name)
    rng = jax.random.PRNGKey(2)
    actions = head.apply(
        params, make_transformer_outputs(), rng, train=False,
        method=head.predict_action,
    )
    expected = (BATCH, ACTION_HORIZON, ACTION_DIM)
    assert actions.shape == expected, f"Expected {expected}, got {actions.shape}"
    assert jnp.all(jnp.isfinite(actions)), "Actions contain NaN/Inf"
    return f"shape={actions.shape}"


def test_sample_shape(name):
    head, params = init_head(name)
    rng = jax.random.PRNGKey(3)
    sample_shape = (4,)
    actions = head.apply(
        params, make_transformer_outputs(), rng, train=False,
        sample_shape=sample_shape, method=head.predict_action,
    )
    expected = sample_shape + (BATCH, ACTION_HORIZON, ACTION_DIM)
    assert actions.shape == expected, f"Expected {expected}, got {actions.shape}"
    return f"shape={actions.shape}"


def test_gradient_flow(name):
    head, params = init_head(name)
    rng = jax.random.PRNGKey(4)
    t_out = make_transformer_outputs()
    acts = make_actions()
    tp, ap = make_masks()

    def loss_fn(p):
        loss, _ = head.apply(
            p, t_out, acts, tp, ap, train=True,
            rngs={"dropout": rng}, method=head.loss,
        )
        return loss

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    zero = [g for g in leaves if jnp.all(g == 0)]
    assert len(zero) == 0, f"{len(zero)}/{len(leaves)} tensors have zero grad"
    return f"all {len(leaves)} tensors OK"


def test_pad_mask(name):
    head, params = init_head(name)
    rng = jax.random.PRNGKey(5)
    t_out = make_transformer_outputs()
    acts = make_actions()
    _, ap = make_masks()

    full_mask = jnp.ones((BATCH, WINDOW), dtype=bool)
    partial_mask = full_mask.at[:, -1].set(False)

    l1, _ = head.apply(params, t_out, acts, full_mask, ap, train=True, rngs={"dropout": rng}, method=head.loss)
    l2, _ = head.apply(params, t_out, acts, partial_mask, ap, train=True, rngs={"dropout": rng}, method=head.loss)
    assert float(l1) != float(l2), "Masking had no effect"
    return f"full={float(l1):.4f} masked={float(l2):.4f}"


def test_stochasticity(name):
    head, params = init_head(name)
    t_out = make_transformer_outputs()
    a1 = head.apply(params, t_out, jax.random.PRNGKey(0), train=False, method=head.predict_action)
    a2 = head.apply(params, t_out, jax.random.PRNGKey(1), train=False, method=head.predict_action)
    assert not jnp.allclose(a1, a2), "Identical across seeds"
    return "different seeds -> different actions"


SMOKE_TESTS = [
    ("init", test_init),
    ("loss", test_loss),
    ("predict_action", test_predict_action),
    ("sample_shape", test_sample_shape),
    ("gradient_flow", test_gradient_flow),
    ("pad_mask", test_pad_mask),
    ("stochasticity", test_stochasticity),
]


# ══════════════════════════════════════════════════════════════════════
#  OVERFIT COMPARISON
# ══════════════════════════════════════════════════════════════════════

def overfit_head(name, num_steps=OVERFIT_STEPS, lr=OVERFIT_LR):
    head = HEAD_CONFIGS[name]()

    rng = jax.random.PRNGKey(0)
    rng, obs_key, act_key = jax.random.split(rng, 3)

    fake_obs_tokens = jax.random.normal(obs_key, (BATCH, WINDOW, NUM_TOKENS, EMB_DIM))
    fake_obs_mask = jnp.ones((BATCH, WINDOW, NUM_TOKENS), dtype=bool)
    fake_actions = jax.random.normal(act_key, (BATCH, WINDOW, ACTION_HORIZON, ACTION_DIM))
    timestep_pad_mask = jnp.ones((BATCH, WINDOW), dtype=bool)
    action_pad_mask = jnp.ones((BATCH, WINDOW, ACTION_HORIZON, ACTION_DIM), dtype=bool)

    transformer_outputs = {"obs": TokenGroup(tokens=fake_obs_tokens, mask=fake_obs_mask)}

    rng, init_rng = jax.random.split(rng)
    variables = head.init({"params": init_rng, "dropout": init_rng}, transformer_outputs)
    params = variables["params"]

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    def loss_fn(params, rng):
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        loss, metrics = bound.loss(
            transformer_outputs, fake_actions, timestep_pad_mask, action_pad_mask, train=True,
        )
        return loss, metrics

    @jax.jit
    def train_step(params, opt_state, rng):
        rng, step_rng = jax.random.split(rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, step_rng)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, rng, metrics

    losses = []
    t0 = _time.time()
    for step in range(num_steps):
        params, opt_state, rng, metrics = train_step(params, opt_state, rng)
        if step % 50 == 0 or step == num_steps - 1:
            loss_val = float(metrics["loss"])
            losses.append(loss_val)
            print(f"    [{name:28s}] step {step:4d}  loss={loss_val:.6f}")
    elapsed = _time.time() - t0

    rng, pred_rng = jax.random.split(rng)
    bound = head.bind({"params": params}, rngs={"dropout": pred_rng})
    pred_actions = bound.predict_action(transformer_outputs, rng=pred_rng, train=False)
    gt = fake_actions[:, -1, :, :]
    action_mse = float(jnp.mean((pred_actions - gt) ** 2))

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / (losses[0] + 1e-12),
        "action_mse": action_mse,
        "time_s": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    head_names = list(HEAD_CONFIGS.keys())

    # ── Part 1: Smoke tests ──
    print("=" * 72)
    print("PART 1: SMOKE TESTS")
    print("=" * 72)

    total_pass, total_fail = 0, 0

    for hname in head_names:
        print(f"\n--- {hname} ---")
        for tname, tfunc in SMOKE_TESTS:
            try:
                detail = tfunc(hname)
                print(f"  [PASS] {tname}: {detail}")
                total_pass += 1
            except Exception as e:
                print(f"  [FAIL] {tname}: {e}")
                total_fail += 1

    print(f"\nSmoke tests: {total_pass} passed, {total_fail} failed")

    # ── Part 2: Overfit comparison ──
    print("\n" + "=" * 72)
    print(f"PART 2: OVERFIT COMPARISON ({OVERFIT_STEPS} steps, batch={BATCH})")
    print("=" * 72)

    comparison = {}
    for hname in head_names:
        print(f"\n  Training: {hname}")
        comparison[hname] = overfit_head(hname)

    # ── Results table ──
    print("\n" + "=" * 72)
    print("RESULTS TABLE")
    print("=" * 72)

    header = f"{'Method':<30s} | {'Init Loss':>10s} | {'Final Loss':>10s} | {'Ratio':>8s} | {'Act MSE':>8s} | {'Time(s)':>7s}"
    print(header)
    print("-" * len(header))
    for hname in head_names:
        r = comparison[hname]
        print(
            f"{hname:<30s} | {r['initial_loss']:10.4f} | {r['final_loss']:10.4f} | "
            f"{r['loss_ratio']:8.4f} | {r['action_mse']:8.4f} | {r['time_s']:7.1f}"
        )

    # ── Pass/fail ──
    print("\n" + "=" * 72)
    print("OVERFIT PASS/FAIL")
    print("=" * 72)
    overfit_pass = True
    for hname in head_names:
        r = comparison[hname]
        converged = r["loss_ratio"] < 0.2
        status = "PASS" if converged else "FAIL"
        if not converged:
            overfit_pass = False
        print(f"  [{status}] {hname}: ratio={r['loss_ratio']:.4f} (need <0.2)")

    overall = total_fail == 0 and overfit_pass
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)
