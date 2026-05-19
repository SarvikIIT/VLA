#!/usr/bin/env python3
"""
Multi-Task Meta-World Benchmark: ONE model trained on ALL tasks.

This is where OT methods should shine — the pooled action distribution
is genuinely multi-modal (10 different task strategies mixed together).
OT coupling helps the model learn distinct flow paths for each mode.

Pipeline:
  1. Collect expert demos from all selected tasks
  2. Pool all (obs, action) pairs into one dataset
  3. Train a SINGLE action head on the pooled data
  4. Evaluate success rate on each task separately
  5. Compare: Diffusion vs Simple Flow vs Sinkhorn OT

Usage:
    python3 tests/metaworld_multitask_eval.py                     # 5 tasks
    python3 tests/metaworld_multitask_eval.py --quick              # 3 tasks, fast
    python3 tests/metaworld_multitask_eval.py --tasks reach-v3 push-v3 drawer-open-v3
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import metaworld

from octo.model.components.action_heads import (
    DiffusionActionHead,
    SimpleFlowMatchActionHead,
    FlowMatchActionHead,
    SinkhornFlowMatchActionHead,
    ConditionedOTFlowMatchActionHead,
)
from octo.model.components.base import TokenGroup


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

MT5_TASKS = [
    "reach-v3", "push-v3", "drawer-open-v3", "drawer-close-v3", "window-open-v3",
]

MT10_TASKS = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "window-close-v3",
]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task Meta-World benchmark")
    p.add_argument("--heads", nargs="+",
                    default=["diffusion", "flow", "sinkhorn", "conditioned"],
                    choices=["diffusion", "flow", "cot", "sinkhorn", "conditioned"])
    p.add_argument("--tasks", nargs="+", default=MT5_TASKS)
    p.add_argument("--demo-episodes", type=int, default=50)
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256,
                    help="Batch size for training (default: 256)")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  EXPERT DEMO COLLECTION
# ═══════════════════════════════════════════════════════════════════════

def collect_scripted_demos(task_name, n_episodes=50, max_steps=150):
    mt1 = metaworld.MT1(task_name, seed=42)
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode=None)

    from metaworld.policies import (
        SawyerReachV3Policy, SawyerPushV3Policy,
        SawyerPickPlaceV3Policy, SawyerDoorOpenV3Policy,
        SawyerDrawerOpenV3Policy, SawyerDrawerCloseV3Policy,
        SawyerButtonPressTopdownV3Policy,
        SawyerPegInsertionSideV3Policy,
        SawyerWindowOpenV3Policy, SawyerWindowCloseV3Policy,
    )

    POLICIES = {
        "reach-v3": SawyerReachV3Policy,
        "push-v3": SawyerPushV3Policy,
        "pick-place-v3": SawyerPickPlaceV3Policy,
        "door-open-v3": SawyerDoorOpenV3Policy,
        "drawer-open-v3": SawyerDrawerOpenV3Policy,
        "drawer-close-v3": SawyerDrawerCloseV3Policy,
        "button-press-topdown-v3": SawyerButtonPressTopdownV3Policy,
        "peg-insert-side-v3": SawyerPegInsertionSideV3Policy,
        "window-open-v3": SawyerWindowOpenV3Policy,
        "window-close-v3": SawyerWindowCloseV3Policy,
    }

    policy = POLICIES[task_name]()
    obs_list, act_list = [], []
    successes = 0

    for ep in range(n_episodes):
        task = mt1.train_tasks[ep % len(mt1.train_tasks)]
        env.set_task(task)
        obs, _ = env.reset()
        for step in range(max_steps):
            action = policy.get_action(obs)
            obs_list.append(obs)
            act_list.append(action)
            obs, rew, term, trunc, info = env.step(action)
            if info.get("success", 0):
                successes += 1
                break

    env.close()
    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        successes / n_episodes,
    )


# ═══════════════════════════════════════════════════════════════════════
#  HEAD SETUP
# ═══════════════════════════════════════════════════════════════════════

OBS_DIM = 39
ACT_DIM = 4
ACTION_HORIZON = 1
WINDOW = 1

HEAD_LABELS = {
    "diffusion":   "Diffusion (DDPM)",
    "flow":        "Simple Flow",
    "cot":         "Flow+COT (Hungarian)",
    "sinkhorn":    "Flow+Sinkhorn OT",
    "conditioned": "Conditioned OT (Ours)",
}


def make_head(key, hidden_dim, num_blocks):
    shared = dict(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACT_DIM,
        time_dim=32,
        num_blocks=num_blocks,
        dropout_rate=0.0,
        hidden_dim=hidden_dim,
        use_layer_norm=True,
    )
    if key == "diffusion":
        return DiffusionActionHead(**shared, diffusion_steps=20, n_diffusion_samples=1)
    elif key == "flow":
        return SimpleFlowMatchActionHead(**shared, flow_steps=10, n_flow_samples=1)
    elif key == "cot":
        return FlowMatchActionHead(**shared, flow_steps=10, n_flow_samples=1)
    elif key == "sinkhorn":
        return SinkhornFlowMatchActionHead(
            **shared, flow_steps=10, n_flow_samples=2,
            sinkhorn_reg=0.05, sinkhorn_iters=20,
        )
    elif key == "conditioned":
        return ConditionedOTFlowMatchActionHead(
            **shared, flow_steps=10, n_flow_samples=2,
            sinkhorn_reg=0.05, sinkhorn_iters=20,
            obs_weight=0.3, history_weight=0.0, use_history=False,
        )
    raise ValueError(key)


# ═══════════════════════════════════════════════════════════════════════
#  TRAIN ON POOLED DEMOS
# ═══════════════════════════════════════════════════════════════════════

def train_head_multitask(head_key, obs_data, act_data, obs_mean, obs_std,
                         hidden_dim, num_blocks,
                         train_steps, lr, batch_size=256):
    """Train a SINGLE action head on pooled data from ALL tasks."""
    head = make_head(head_key, hidden_dim, num_blocks)

    N = obs_data.shape[0]
    # Normalize observations (per-dimension standardization)
    obs_norm = (obs_data - obs_mean) / obs_std
    obs_jax = jnp.array(obs_norm)
    act_jax = jnp.array(act_data[:, None, None, :])  # (N, 1, 1, 4)

    # Init
    dummy_obs = obs_jax[:2, None, None, :]
    dummy_mask = jnp.ones((2, 1, 1), dtype=bool)
    dummy_t_out = {"obs": TokenGroup(tokens=dummy_obs, mask=dummy_mask)}

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    variables = head.init({"params": init_rng, "dropout": init_rng}, dummy_t_out)
    params = variables["params"]
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    tp_mask = jnp.ones((batch_size, WINDOW), dtype=bool)
    ap_mask = jnp.ones((batch_size, WINDOW, ACTION_HORIZON, ACT_DIM), dtype=bool)

    def loss_fn(params, obs_batch, act_batch, rng):
        tokens = obs_batch[:, None, None, :]
        mask = jnp.ones(tokens.shape[:3], dtype=bool)
        t_out = {"obs": TokenGroup(tokens=tokens, mask=mask)}
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        loss, metrics = bound.loss(t_out, act_batch, tp_mask, ap_mask, train=True)
        return loss, metrics

    @jax.jit
    def train_step(params, opt_state, rng, obs_batch, act_batch):
        rng, step_rng = jax.random.split(rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, obs_batch, act_batch, step_rng
        )
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, rng, metrics

    print(f"    Training on {N:,} pooled transitions ({param_count:,} params) ...")
    t0 = time.time()
    for step in range(train_steps):
        rng, idx_rng = jax.random.split(rng)
        idx = jax.random.randint(idx_rng, (batch_size,), 0, N)
        obs_batch = obs_jax[idx]
        act_batch = act_jax[idx]
        params, opt_state, rng, metrics = train_step(
            params, opt_state, rng, obs_batch, act_batch
        )
        if step % max(train_steps // 5, 1) == 0 or step == train_steps - 1:
            print(f"      step {step:5d}  loss={float(metrics['loss']):.4f}")
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    return head, params


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_success_rate(head, params, task_name, n_episodes, max_steps,
                          obs_mean, obs_std):
    mt1 = metaworld.MT1(task_name, seed=123)
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode=None)

    successes = 0
    rng = jax.random.PRNGKey(0)
    obs_mean_jnp = jnp.array(obs_mean)
    obs_std_jnp = jnp.array(obs_std)

    @jax.jit
    def predict(params, obs, rng):
        # Normalize observation
        obs_norm = (obs - obs_mean_jnp) / obs_std_jnp
        tokens = obs_norm[None, None, None, :]
        mask = jnp.ones((1, 1, 1), dtype=bool)
        t_out = {"obs": TokenGroup(tokens=tokens, mask=mask)}
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        action = bound.predict_action(t_out, rng=rng, train=False)
        return action[0, 0, :]

    for ep in range(n_episodes):
        task = mt1.train_tasks[ep % len(mt1.train_tasks)]
        env.set_task(task)
        obs, _ = env.reset()
        episode_success = False

        for step in range(max_steps):
            rng, act_rng = jax.random.split(rng)
            obs_jax = jnp.array(obs, dtype=jnp.float32)
            action = predict(params, obs_jax, act_rng)
            action_np = np.array(action, dtype=np.float64)
            action_np = np.clip(action_np, -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(action_np)
            if info.get("success", 0):
                episode_success = True
                break

        if episode_success:
            successes += 1

    env.close()
    return successes / n_episodes


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    if args.quick:
        args.tasks = ["reach-v3", "push-v3", "drawer-open-v3"]
        args.demo_episodes = 30
        args.train_steps = 2000
        args.episodes = 10
        args.hidden = 256
        args.blocks = 3
        args.batch = 256

    device = jax.devices()[0]
    n_tasks = len(args.tasks)
    print("=" * 80)
    print("  MULTI-TASK Meta-World Benchmark")
    print("  (ONE model trained on ALL tasks — genuinely multi-modal)")
    print("=" * 80)
    print(f"  Device         : {device}")
    print(f"  Tasks          : {n_tasks} ({', '.join(args.tasks)})")
    print(f"  Heads          : {', '.join(HEAD_LABELS[h] for h in args.heads)}")
    print(f"  Demo episodes  : {args.demo_episodes} per task")
    print(f"  Batch size     : {args.batch}")
    print(f"  Train steps    : {args.train_steps}")
    print(f"  Eval episodes  : {args.episodes}")
    print(f"  Hidden/blocks  : {args.hidden}/{args.blocks}")
    print("=" * 80)

    # ── Collect demos from ALL tasks ──
    print("\n  Collecting expert demos from all tasks ...")
    all_obs, all_act = [], []
    task_demo_info = {}
    for task_name in args.tasks:
        print(f"    {task_name} ...", end=" ", flush=True)
        obs, act, demo_sr = collect_scripted_demos(
            task_name, args.demo_episodes, args.max_steps
        )
        all_obs.append(obs)
        all_act.append(act)
        task_demo_info[task_name] = (len(obs), demo_sr)
        print(f"{len(obs)} transitions, expert SR={demo_sr:.0%}")

    # Pool all data
    pooled_obs = np.concatenate(all_obs, axis=0)
    pooled_act = np.concatenate(all_act, axis=0)

    # Compute normalization stats
    obs_mean = pooled_obs.mean(axis=0)
    obs_std = pooled_obs.std(axis=0) + 1e-6
    print(f"\n  Pooled dataset: {len(pooled_obs):,} total transitions from {n_tasks} tasks")
    print(f"  Action distribution: mean={pooled_act.mean():.3f}, std={pooled_act.std():.3f}")
    print(f"  Obs normalized: per-dimension standardization (mean=0, std=1)")
    print(f"  (Multi-modal — {n_tasks} distinct task strategies mixed together)")

    # ── Train and evaluate each head ──
    results = {h: {} for h in args.heads}

    for head_key in args.heads:
        label = HEAD_LABELS[head_key]
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        head, params = train_head_multitask(
            head_key, pooled_obs, pooled_act, obs_mean, obs_std,
            args.hidden, args.blocks,
            args.train_steps, args.lr, args.batch,
        )

        for task_name in args.tasks:
            print(f"    Evaluating {task_name} ({args.episodes} eps) ...", end=" ", flush=True)
            sr = evaluate_success_rate(
                head, params, task_name, args.episodes, args.max_steps,
                obs_mean, obs_std
            )
            results[head_key][task_name] = sr
            print(f"SR={sr:.0%}")

    # ── Results Table ──
    print("\n\n")
    print("=" * 80)
    print("  MULTI-TASK Meta-World Results (one model, all tasks)")
    print("=" * 80)

    heads = args.heads
    labels = [HEAD_LABELS[h] for h in heads]
    tc = 30
    hc = max(16, max(len(l) for l in labels) + 2)

    sep = "  " + "-" * (tc + (hc + 3) * len(heads))
    print(sep)
    hdr = f"  {'Task':<{tc}}"
    for l in labels:
        hdr += f" | {l:>{hc}}"
    print(hdr)
    print(sep)

    avg_sr = {h: [] for h in heads}
    for task_name in args.tasks:
        row = f"  {task_name:<{tc}}"
        srs = [results[h][task_name] for h in heads]
        best_idx = int(np.argmax(srs))
        for i, sr in enumerate(srs):
            s = f"{sr:.0%}"
            if i == best_idx and sr > 0:
                s += " *"
            row += f" | {s:>{hc}}"
            avg_sr[heads[i]].append(sr)
        print(row)

    print(sep)
    row = f"  {'AVERAGE':<{tc}}"
    avgs = [np.mean(avg_sr[h]) for h in heads]
    best_idx = int(np.argmax(avgs))
    for i, a in enumerate(avgs):
        s = f"{a:.1%}"
        if i == best_idx:
            s += " *"
        row += f" | {s:>{hc}}"
    print(row)
    print(sep)
    print("  (* = best in row)")

    # ── Summary ──
    print(f"\n  Summary:")
    for h in heads:
        avg = np.mean(avg_sr[h])
        print(f"    {HEAD_LABELS[h]:25s}: {avg:.1%} average success rate")

    cond_avg = np.mean(avg_sr.get("conditioned", [0]))
    sk_avg = np.mean(avg_sr.get("sinkhorn", [0]))
    flow_avg = np.mean(avg_sr.get("flow", [0]))
    diff_avg = np.mean(avg_sr.get("diffusion", [0]))

    our = cond_avg if "conditioned" in heads else sk_avg
    our_label = "Conditioned OT" if "conditioned" in heads else "Sinkhorn"

    if flow_avg > 0:
        pct = (our - flow_avg) / flow_avg * 100
        print(f"\n    {our_label} vs Simple Flow: {'+' if pct > 0 else ''}{pct:.1f}%")
    if diff_avg > 0:
        pct = (our - diff_avg) / diff_avg * 100
        print(f"    {our_label} vs Diffusion:   {'+' if pct > 0 else ''}{pct:.1f}%")

    print()
    sys.exit(0)
