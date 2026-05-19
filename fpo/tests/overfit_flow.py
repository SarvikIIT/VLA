"""
Overfit test for FlowMatchActionHead.

Two tests:
  1. Unit test   – synthetic transformer outputs, trains just the flow head.
  2. Full test   – loads one real batch from the debug dataset, trains the
                   entire Octo model end-to-end with FlowMatchActionHead.

Both verify the loss drops toward zero, confirming the flow matching
forward/backward/inference loop is wired up correctly.

Usage:
    python tests/overfit_flow.py              # run both tests
    python tests/overfit_flow.py --unit       # unit test only (no data needed)
    python tests/overfit_flow.py --full       # full pipeline test
"""

import argparse
import sys
import os

# Prevent TF from grabbing GPUs (must happen before other imports)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

# ── ensure project root is importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octo.model.components.flow import create_flow_model
from octo.model.components.action_heads import FlowMatchActionHead
from octo.model.components.base import TokenGroup


# ═══════════════════════════════════════════════════════════════════════
# 1.  UNIT TEST – flow head only, synthetic data
# ═══════════════════════════════════════════════════════════════════════

def test_unit_overfit():
    """Train just the FlowMatchActionHead on a fixed synthetic batch."""

    print("\n" + "=" * 60)
    print("UNIT TEST: Overfitting FlowMatchActionHead (head-only)")
    print("=" * 60)

    # ── Dimensions ──
    batch_size, window_size = 4, 1
    obs_dim = 64
    action_dim, action_horizon = 7, 1
    num_steps = 300
    lr = 3e-4

    # ── Synthetic inputs ──
    rng = jax.random.PRNGKey(0)
    rng, obs_key, act_key = jax.random.split(rng, 3)

    fake_obs_tokens = jax.random.normal(
        obs_key, (batch_size, window_size, 4, obs_dim)
    )  # 4 tokens per timestep
    fake_obs_mask = jnp.ones((batch_size, window_size, 4), dtype=bool)
    fake_actions = jax.random.normal(
        act_key, (batch_size, window_size, action_horizon, action_dim)
    )
    timestep_pad_mask = jnp.ones((batch_size, window_size), dtype=bool)
    action_pad_mask = jnp.ones(
        (batch_size, window_size, action_horizon, action_dim), dtype=bool
    )

    transformer_outputs = {
        "obs": TokenGroup(tokens=fake_obs_tokens, mask=fake_obs_mask)
    }

    # ── Instantiate head ──
    head = FlowMatchActionHead(
        readout_key="obs",
        action_horizon=action_horizon,
        action_dim=action_dim,
        hidden_dim=128,
        use_layer_norm=True,
        flow_steps=10,
        n_flow_samples=1,
        time_dim=32,
        num_blocks=3,
    )

    rng, init_rng = jax.random.split(rng)
    variables = head.init(
        {"params": init_rng, "dropout": init_rng},
        transformer_outputs,
    )
    params = variables["params"]

    # ── Optimizer ──
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    # ── Train step ──
    def loss_fn(params, rng):
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        loss, metrics = bound.loss(
            transformer_outputs,
            fake_actions,
            timestep_pad_mask,
            action_pad_mask,
            train=True,
        )
        return loss, metrics

    @jax.jit
    def train_step(params, opt_state, rng):
        rng, step_rng = jax.random.split(rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, step_rng
        )
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, rng, metrics

    # ── Training loop ──
    losses = []
    for step in range(num_steps):
        params, opt_state, rng, metrics = train_step(params, opt_state, rng)
        if step % 50 == 0 or step == num_steps - 1:
            loss_val = float(metrics["loss"])
            losses.append(loss_val)
            print(f"  step {step:4d}  loss={loss_val:.6f}")

    # ── Predict and compare ──
    rng, pred_rng = jax.random.split(rng)
    bound = head.bind({"params": params}, rngs={"dropout": pred_rng})
    pred_actions = bound.predict_action(
        transformer_outputs, rng=pred_rng, train=False
    )
    # pred_actions: (batch, action_horizon, action_dim)
    gt = fake_actions[:, -1, :, :]  # last window step
    action_mse = float(jnp.mean((pred_actions - gt) ** 2))

    # ── Checks ──
    initial, final = losses[0], losses[-1]
    print(f"\n  Initial loss : {initial:.6f}")
    print(f"  Final loss   : {final:.6f}")
    print(f"  Action MSE   : {action_mse:.6f}")

    passed = True
    if final < initial * 0.05:
        print("  [PASS] Loss decreased >20x")
    else:
        print(f"  [FAIL] Loss ratio = {final/initial:.4f}  (expected <0.05)")
        passed = False

    if action_mse < 0.5:
        print("  [PASS] Predicted actions close to GT")
    else:
        print(f"  [WARN] Action MSE = {action_mse:.4f}  (expected <0.5)")

    return passed


# ═══════════════════════════════════════════════════════════════════════
# 2.  FULL PIPELINE TEST – real data, full Octo model
# ═══════════════════════════════════════════════════════════════════════

def test_full_overfit():
    """Load one real batch, train full Octo + FlowMatchActionHead."""

    print("\n" + "=" * 60)
    print("FULL TEST: Overfitting Octo + FlowMatchActionHead (end-to-end)")
    print("=" * 60)

    from ml_collections import ConfigDict
    from octo.data.dataset import make_interleaved_dataset
    from octo.model.octo_model import OctoModel
    from octo.utils.spec import ModuleSpec
    from octo.utils.train_utils import (
        TrainState,
        create_optimizer,
        process_text,
    )
    from tests.debug_config import get_config

    # ── Config: swap head to FlowMatchActionHead ──
    config = get_config()
    config.model.heads.action = ModuleSpec.create(
        FlowMatchActionHead,
        readout_key="obs",
        action_horizon=1,
        action_dim=7,
        hidden_dim=256,
        use_layer_norm=True,
        flow_steps=10,
        n_flow_samples=1,
        time_dim=32,
        num_blocks=3,
    )
    config.dataset_kwargs.batch_size = 8

    # ── Text processor + data ──
    text_processor = ModuleSpec.instantiate(config.text_processor)()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    print("  Loading debug dataset...")
    train_data = make_interleaved_dataset(**config.dataset_kwargs, train=True)
    batch = next(
        map(process_batch, train_data.iterator(prefetch=0))
    )
    batch = jax.tree.map(jnp.array, batch)
    print(f"  Batch action shape: {batch['action'].shape}")

    # ── Model init ──
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    print("  Initializing model...")
    model = OctoModel.from_config(
        config.to_dict(),
        batch,
        text_processor,
        verbose=False,
        rng=init_rng,
        dataset_statistics=train_data.dataset_statistics,
    )

    # ── Optimizer (higher LR for fast overfitting) ──
    opt_cfg = config.optimizer.to_dict()
    opt_cfg["learning_rate"] = {
        "name": "rsqrt",
        "init_value": 0.0,
        "peak_value": 3e-4,
        "warmup_steps": 50,
        "timescale": 10000,
    }
    tx, lr_callable, param_norm_callable = create_optimizer(
        model.params, **opt_cfg
    )
    train_state = TrainState.create(rng, model, tx)

    # ── Loss fn (same as train.py) ──
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind(
            {"params": params}, rngs={"dropout": rng}
        )
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # ── Train ──
    num_steps = 500
    losses = []
    print(f"  Training for {num_steps} steps on one fixed batch...\n")

    for step in range(num_steps):
        train_state, info = train_step(train_state, batch)
        if step % 50 == 0 or step == num_steps - 1:
            loss_val = float(info["loss"])
            mse_val = float(info["mse"])
            losses.append(loss_val)
            print(f"  step {step:4d}  loss={loss_val:.6f}  mse={mse_val:.6f}")

    # ── Predict actions ──
    print("\n  Predicting actions...")
    updated_model = train_state.model
    pred_actions = updated_model.sample_actions(
        batch["observation"],
        batch["task"],
        rng=jax.random.PRNGKey(0),
    )
    gt_actions = batch["action"][:, -1, :, :]
    action_mse = float(jnp.mean((pred_actions - gt_actions) ** 2))
    action_mae = float(jnp.mean(jnp.abs(pred_actions - gt_actions)))

    # ── Checks ──
    initial, final = losses[0], losses[-1]
    print(f"\n  Initial loss  : {initial:.6f}")
    print(f"  Final loss    : {final:.6f}")
    print(f"  Action MSE    : {action_mse:.6f}")
    print(f"  Action MAE    : {action_mae:.6f}")
    print(f"  GT range      : [{float(gt_actions.min()):.3f}, {float(gt_actions.max()):.3f}]")
    print(f"  Pred range    : [{float(pred_actions.min()):.3f}, {float(pred_actions.max()):.3f}]")

    passed = True
    if final < initial * 0.1:
        print("  [PASS] Loss decreased >10x")
    else:
        print(f"  [FAIL] Loss ratio = {final/initial:.4f}  (expected <0.1)")
        passed = False

    if final < 0.5:
        print("  [PASS] Final loss < 0.5")
    else:
        print(f"  [WARN] Final loss = {final:.6f}")

    if action_mse < 1.0:
        print("  [PASS] Predicted actions reasonably close to GT")
    else:
        print(f"  [WARN] Action MSE = {action_mse:.4f}  (expected <1.0)")

    return passed


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overfit tests for FlowMatchActionHead")
    parser.add_argument("--unit", action="store_true", help="Run unit test only")
    parser.add_argument("--full", action="store_true", help="Run full pipeline test only")
    args = parser.parse_args()

    run_unit = args.unit or (not args.unit and not args.full)
    run_full = args.full or (not args.unit and not args.full)

    results = {}

    if run_unit:
        results["unit"] = test_unit_overfit()

    if run_full:
        results["full"] = test_full_overfit()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:6s}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)
