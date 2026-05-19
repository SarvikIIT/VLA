#!/usr/bin/env python3
"""
Meta-World Success Rate Benchmark: Diffusion vs Simple Flow vs Flow + COT

Pipeline:
  1. Create Meta-World MT10 environments (10 manipulation tasks)
  2. For each action head: train on expert demos, then evaluate success rate
  3. Report per-task and average success rates

Since we don't have a pretrained Octo backbone here, we use a lightweight
setup: the action heads are conditioned on the raw Meta-World observation
(39D proprio+object state) directly, without vision. This isolates the
effect of the action head on task success.

Usage:
    python3 tests/metaworld_eval.py                         # full MT10
    python3 tests/metaworld_eval.py --tasks push-v3 reach-v3  # pick tasks
    python3 tests/metaworld_eval.py --quick                  # fast check (3 tasks)
    python3 tests/metaworld_eval.py --episodes 50            # more eval episodes
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
)
from octo.model.components.base import TokenGroup


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

MT10_TASKS = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "window-close-v3",
]


def parse_args():
    p = argparse.ArgumentParser(description="Meta-World success rate benchmark")
    p.add_argument("--heads", nargs="+",
                    default=["diffusion", "flow", "cot", "sinkhorn"],
                    choices=["diffusion", "flow", "cot", "sinkhorn"])
    p.add_argument("--tasks", nargs="+", default=MT10_TASKS,
                    help="Meta-World task names (default: MT10)")
    p.add_argument("--demo-episodes", type=int, default=50,
                    help="Expert demo episodes for training (default: 50)")
    p.add_argument("--train-steps", type=int, default=1000,
                    help="Training steps per task (default: 1000)")
    p.add_argument("--episodes", type=int, default=20,
                    help="Eval episodes per task (default: 20)")
    p.add_argument("--max-steps", type=int, default=150,
                    help="Max steps per episode (default: 150)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--quick", action="store_true",
                    help="Quick: 3 tasks, 20 demos, 500 train steps, 10 eval eps")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  EXPERT DEMO COLLECTION
# ═══════════════════════════════════════════════════════════════════════

def collect_scripted_demos(task_name, n_episodes=50, max_steps=150):
    """Collect demos using Meta-World's scripted policies.

    Returns:
        obs_all: (N, obs_dim) all observations
        act_all: (N, act_dim) all actions
    """
    mt1 = metaworld.MT1(task_name, seed=42)
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode=None)

    # Meta-World scripted policies
    from metaworld.policies import SawyerReachV3Policy, SawyerPushV3Policy
    from metaworld.policies import SawyerPickPlaceV3Policy, SawyerDoorOpenV3Policy
    from metaworld.policies import SawyerDrawerOpenV3Policy, SawyerDrawerCloseV3Policy
    from metaworld.policies import SawyerButtonPressTopdownV3Policy
    from metaworld.policies import SawyerPegInsertionSideV3Policy
    from metaworld.policies import SawyerWindowOpenV3Policy, SawyerWindowCloseV3Policy

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
    demo_sr = successes / n_episodes
    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        demo_sr,
    )


# ═══════════════════════════════════════════════════════════════════════
#  HEAD SETUP
# ═══════════════════════════════════════════════════════════════════════

OBS_DIM = 39
ACT_DIM = 4
ACTION_HORIZON = 1
WINDOW = 1

HEAD_LABELS = {
    "diffusion": "Diffusion",
    "flow":      "Simple Flow",
    "cot":       "Flow+COT (Hungarian)",
    "sinkhorn":  "Flow+Sinkhorn OT",
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
    raise ValueError(key)


def obs_to_transformer_outputs(obs_batch):
    """Convert raw observations to the format action heads expect.

    obs_batch: (batch, obs_dim) -> TokenGroup with (batch, 1, 1, obs_dim)
    """
    tokens = obs_batch[:, None, None, :]  # (B, W=1, T=1, obs_dim)
    mask = jnp.ones(tokens.shape[:3], dtype=bool)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


# ═══════════════════════════════════════════════════════════════════════
#  TRAIN ON DEMOS
# ═══════════════════════════════════════════════════════════════════════

def train_head_on_demos(head_key, obs_data, act_data, hidden_dim, num_blocks,
                        train_steps, lr, batch_size=128):
    """Train an action head on expert demonstrations."""
    head = make_head(head_key, hidden_dim, num_blocks)

    # Prepare data as (N, W=1, H=1, dim)
    N = obs_data.shape[0]
    obs_jax = jnp.array(obs_data)
    act_jax = jnp.array(act_data[:, None, None, :])  # (N, 1, 1, act_dim)

    # Create a dummy batch for init
    dummy_obs = obs_jax[:2, None, None, :]  # (2, 1, 1, 39)
    dummy_mask = jnp.ones((2, 1, 1), dtype=bool)
    dummy_t_out = {"obs": TokenGroup(tokens=dummy_obs, mask=dummy_mask)}

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    variables = head.init({"params": init_rng, "dropout": init_rng}, dummy_t_out)
    params = variables["params"]

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    tp_mask_b = jnp.ones((batch_size, WINDOW), dtype=bool)
    ap_mask_b = jnp.ones((batch_size, WINDOW, ACTION_HORIZON, ACT_DIM), dtype=bool)

    def loss_fn(params, obs_batch, act_batch, rng):
        tokens = obs_batch[:, None, None, :]
        mask = jnp.ones(tokens.shape[:3], dtype=bool)
        t_out = {"obs": TokenGroup(tokens=tokens, mask=mask)}
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        loss, metrics = bound.loss(t_out, act_batch, tp_mask_b, ap_mask_b, train=True)
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

    for step in range(train_steps):
        rng, idx_rng = jax.random.split(rng)
        idx = jax.random.randint(idx_rng, (batch_size,), 0, N)
        obs_batch = obs_jax[idx]
        act_batch = act_jax[idx]
        params, opt_state, rng, metrics = train_step(
            params, opt_state, rng, obs_batch, act_batch
        )
        if step == 0 or step == train_steps - 1:
            print(f"      step {step:5d}  loss={float(metrics['loss']):.4f}")

    return head, params


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATE SUCCESS RATE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_success_rate(head, params, task_name, n_episodes, max_steps):
    """Run the trained head in Meta-World and return success rate."""
    mt1 = metaworld.MT1(task_name, seed=123)
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode=None)

    successes = 0
    rng = jax.random.PRNGKey(0)

    @jax.jit
    def predict(params, obs, rng):
        tokens = obs[None, None, None, :]  # (1, 1, 1, 39)
        mask = jnp.ones((1, 1, 1), dtype=bool)
        t_out = {"obs": TokenGroup(tokens=tokens, mask=mask)}
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        # All heads are generative
        action = bound.predict_action(t_out, rng=rng, train=False)
        return action[0, 0, :]  # (act_dim,)

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
        args.demo_episodes = 20
        args.train_steps = 500
        args.episodes = 10
        args.hidden = 128
        args.blocks = 2

    device = jax.devices()[0]
    print("=" * 80)
    print("  Meta-World Success Rate Benchmark")
    print("=" * 80)
    print(f"  Device         : {device}")
    print(f"  Tasks          : {len(args.tasks)} ({', '.join(args.tasks)})")
    print(f"  Heads          : {', '.join(HEAD_LABELS[h] for h in args.heads)}")
    print(f"  Demo episodes  : {args.demo_episodes}")
    print(f"  Train steps    : {args.train_steps}")
    print(f"  Eval episodes  : {args.episodes}")
    print(f"  Max steps/ep   : {args.max_steps}")
    print(f"  Hidden/blocks  : {args.hidden}/{args.blocks}")
    print("=" * 80)

    # Collect demos for each task
    print("\n  Collecting expert demos ...")
    task_demos = {}
    for task_name in args.tasks:
        print(f"    {task_name} ...", end=" ", flush=True)
        obs, act, demo_sr = collect_scripted_demos(
            task_name, args.demo_episodes, args.max_steps
        )
        task_demos[task_name] = (obs, act)
        print(f"{len(obs)} transitions, expert SR={demo_sr:.0%}")

    # Train and evaluate each head on each task
    # results[head_key][task_name] = success_rate
    results = {h: {} for h in args.heads}

    for head_key in args.heads:
        label = HEAD_LABELS[head_key]
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        for task_name in args.tasks:
            obs, act = task_demos[task_name]
            print(f"\n    [{task_name}] Training ...")
            head, params = train_head_on_demos(
                head_key, obs, act, args.hidden, args.blocks,
                args.train_steps, args.lr,
            )
            print(f"    [{task_name}] Evaluating {args.episodes} episodes ...")
            sr = evaluate_success_rate(
                head, params, task_name, args.episodes, args.max_steps
            )
            results[head_key][task_name] = sr
            print(f"    [{task_name}] Success Rate: {sr:.0%}")

    # ── Results Table ──
    print("\n\n")
    print("=" * 80)
    print("  Meta-World Success Rate Results")
    print("=" * 80)

    heads = args.heads
    labels = [HEAD_LABELS[h] for h in heads]
    tc = 30
    hc = max(14, max(len(l) for l in labels) + 2)

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

    # Average
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
        print(f"    {HEAD_LABELS[h]:20s}: {avg:.1%} average success rate")

    sk_avg = np.mean(avg_sr.get("sinkhorn", [0]))
    cot_avg = np.mean(avg_sr.get("cot", [0]))
    diff_avg = np.mean(avg_sr.get("diffusion", [0]))
    flow_avg = np.mean(avg_sr.get("flow", [0]))

    if "sinkhorn" in heads and "diffusion" in heads and diff_avg > 0:
        pct = (sk_avg - diff_avg) / diff_avg * 100
        print(f"\n    Sinkhorn vs Diffusion: {'+' if pct > 0 else ''}{pct:.1f}% success rate")

    if "sinkhorn" in heads and "flow" in heads and flow_avg > 0:
        pct = (sk_avg - flow_avg) / flow_avg * 100
        print(f"    Sinkhorn vs Simple Flow: {'+' if pct > 0 else ''}{pct:.1f}% success rate")

    if "sinkhorn" in heads and "cot" in heads and cot_avg > 0:
        pct = (sk_avg - cot_avg) / cot_avg * 100
        print(f"    Sinkhorn vs Hungarian COT: {'+' if pct > 0 else ''}{pct:.1f}% success rate")

    print()
    sys.exit(0)
