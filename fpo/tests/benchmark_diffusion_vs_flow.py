#!/usr/bin/env python3
"""
Complete Comparison: Vanilla Octo (Diffusion) vs Simple Flow vs Flow + COT

Creates STRUCTURED synthetic data with clustered observations → clustered
actions, so the COT coupling has real structure to exploit.

Also compares step counts: COT should match simple flow accuracy with
fewer Euler steps.

Usage:
    python3 tests/benchmark_diffusion_vs_flow.py                    # full run
    python3 tests/benchmark_diffusion_vs_flow.py --quick            # fast sanity
    python3 tests/benchmark_diffusion_vs_flow.py --heads flow cot   # pick heads
    python3 tests/benchmark_diffusion_vs_flow.py --steps 1000 --batch 256
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

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Octo action heads")
    p.add_argument("--heads", nargs="+",
                    default=["diffusion", "flow", "cot", "sinkhorn", "conditioned_ot", "flow_fpo"],
                    choices=["diffusion", "flow", "cot", "sinkhorn", "conditioned_ot", "flow_fpo"])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=128,
                    help="Batch size — must be >> 64 for COT clustering (default: 128)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=3,
                    help="Number of MLPResNet blocks (default: 3)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--quick", action="store_true",
                    help="Quick: 300 steps, batch 128, 10 trials")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  STRUCTURED SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════
# Create data where observations fall into K clusters, and actions
# depend on which cluster the observation belongs to. This gives COT
# real structure to exploit via its PCA → K-means → OT pipeline.

WINDOW = 2
NUM_TOKENS = 4
EMB_DIM = 64
ACTION_HORIZON = 4
ACTION_DIM = 7
N_OBS_CLUSTERS = 8  # distinct observation modes


def make_structured_data(batch_size, seed=0):
    """Create clustered obs → action mapping.

    Each observation cluster gets a distinct action center.  Noise is added
    so the mapping is not trivial.  This mimics real robotics data where
    different visual scenes lead to different action modes.
    """
    rng = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    # Cluster assignments for each sample
    cluster_ids = jax.random.randint(k1, (batch_size,), 0, N_OBS_CLUSTERS)

    # Per-cluster observation centers (K, tokens, emb)
    obs_centers = jax.random.normal(k2, (N_OBS_CLUSTERS, NUM_TOKENS, EMB_DIM)) * 3.0

    # Per-cluster action centers (K, horizon, dim)
    act_centers = jax.random.normal(k3, (N_OBS_CLUSTERS, ACTION_HORIZON, ACTION_DIM)) * 2.0

    # Build batch by indexing into cluster centers + noise
    obs_tokens = obs_centers[cluster_ids]  # (B, tokens, emb)
    obs_tokens = obs_tokens[:, None, :, :].repeat(WINDOW, axis=1)  # (B, W, tokens, emb)
    obs_noise = jax.random.normal(k4, obs_tokens.shape) * 0.3
    obs_tokens = obs_tokens + obs_noise

    actions = act_centers[cluster_ids]  # (B, horizon, dim)
    actions = actions[:, None, :, :].repeat(WINDOW, axis=1)  # (B, W, horizon, dim)
    act_noise = jax.random.normal(k5, actions.shape) * 0.2
    actions = actions + act_noise

    obs_mask = jnp.ones((batch_size, WINDOW, NUM_TOKENS), dtype=bool)
    tp_mask = jnp.ones((batch_size, WINDOW), dtype=bool)
    ap_mask = jnp.ones((batch_size, WINDOW, ACTION_HORIZON, ACTION_DIM), dtype=bool)

    t_out = {"obs": TokenGroup(tokens=obs_tokens, mask=obs_mask)}
    return t_out, actions, tp_mask, ap_mask, cluster_ids


# ═══════════════════════════════════════════════════════════════════════
#  HEAD REGISTRY
# ═══════════════════════════════════════════════════════════════════════

HEAD_LABELS = {
    "diffusion":      "Vanilla Octo (Diffusion)",
    "flow":           "Simple Flow Matching",
    "cot":            "Flow + COT (Hungarian)",
    "sinkhorn":       "Flow + Sinkhorn OT",
    "conditioned_ot": "Flow + Conditioned OT",
    "flow_fpo":       "Flow + FPO (Ours)",
}


def make_head(key, hidden_dim, num_blocks=3, flow_steps=10):
    shared = dict(
        readout_key="obs",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        time_dim=32,
        num_blocks=num_blocks,
        dropout_rate=0.0,
        hidden_dim=hidden_dim,
        use_layer_norm=True,
    )
    if key == "diffusion":
        return DiffusionActionHead(**shared, diffusion_steps=20, n_diffusion_samples=1)
    elif key == "flow":
        return SimpleFlowMatchActionHead(**shared, flow_steps=flow_steps, n_flow_samples=1)
    elif key == "cot":
        return FlowMatchActionHead(**shared, flow_steps=flow_steps, n_flow_samples=1)
    elif key == "sinkhorn":
        return SinkhornFlowMatchActionHead(
            **shared, flow_steps=flow_steps, n_flow_samples=2,
            sinkhorn_reg=0.05, sinkhorn_iters=20,
        )
    elif key == "conditioned_ot":
        return ConditionedOTFlowMatchActionHead(
            **shared, flow_steps=flow_steps, n_flow_samples=2,
            sinkhorn_reg=0.05, sinkhorn_iters=20,
            obs_weight=0.3, history_weight=0.5, use_history=True,
        )
    elif key == "flow_fpo":
        # FPO uses the same flow head — difference is training, not architecture
        return SimpleFlowMatchActionHead(**shared, flow_steps=flow_steps, n_flow_samples=1)
    raise ValueError(key)


def inf_steps_label(key, flow_steps=10):
    if key == "diffusion":
        return "20 (DDPM)"
    return f"{flow_steps} (Euler)"


# ═══════════════════════════════════════════════════════════════════════
#  TRAIN + BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

def run_head(key, batch_size, hidden_dim, num_blocks, train_steps, lr, warmup, trials, flow_steps=10):
    label = HEAD_LABELS[key]
    head = make_head(key, hidden_dim, num_blocks=num_blocks, flow_steps=flow_steps)

    t_out, actions, tp_mask, ap_mask, cluster_ids = make_structured_data(batch_size)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    variables = head.init({"params": init_rng, "dropout": init_rng}, t_out)
    params = variables["params"]
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    def loss_fn(params, rng):
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        loss, metrics = bound.loss(t_out, actions, tp_mask, ap_mask, train=True)
        return loss, metrics

    @jax.jit
    def train_step(params, opt_state, rng):
        rng, step_rng = jax.random.split(rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, step_rng)
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, rng, metrics

    step_label = inf_steps_label(key, flow_steps)
    print(f"\n  [{label} | {step_label}] Training {train_steps} steps ...")
    losses = []
    t0 = time.time()
    for step in range(train_steps):
        params, opt_state, rng, metrics = train_step(params, opt_state, rng)
        if step % max(train_steps // 8, 1) == 0 or step == train_steps - 1:
            l = float(metrics["loss"])
            losses.append(l)
            print(f"    step {step:5d}  loss={l:.4f}")
    bc_time = time.time() - t0

    # FPO fine-tuning phase (only for flow_fpo)
    fpo_time = 0.0
    if key == "flow_fpo":
        fpo_steps = max(train_steps // 4, 50)
        fpo_lr = lr * 0.1
        print(f"    FPO fine-tuning ({fpo_steps} steps, lr={fpo_lr:.1e}) ...")

        # FPO: use loss ratio as surrogate for policy improvement
        # r̂_FPO(θ) = exp( L(θ_old) - L(θ_new) ), PPO-clip update
        params_old = jax.tree.map(lambda x: x.copy(), params)
        tx_fpo = optax.adam(fpo_lr)
        opt_fpo = tx_fpo.init(params)
        clip_eps = 0.05

        def fpo_loss_fn(params, params_old, rng):
            rng, r1, r2 = jax.random.split(rng, 3)
            # Current policy loss
            bound_new = head.bind({"params": params}, rngs={"dropout": r1})
            loss_new, _ = bound_new.loss(t_out, actions, tp_mask, ap_mask, train=True)
            # Old policy loss
            bound_old = head.bind({"params": params_old}, rngs={"dropout": r2})
            loss_old, _ = bound_old.loss(t_out, actions, tp_mask, ap_mask, train=True)
            # FPO ratio: higher when new policy is better
            ratio = jnp.exp(jnp.clip(loss_old - loss_new, -5.0, 5.0))
            # PPO clip with uniform positive advantage (we want to improve everywhere)
            clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            fpo_loss = -jnp.minimum(ratio, clipped)
            # BC regularization to prevent collapse
            total = fpo_loss + 0.5 * loss_new
            return total, {"fpo_loss": fpo_loss, "bc_loss": loss_new, "ratio": ratio}

        @jax.jit
        def fpo_step(params, params_old, opt_state, rng):
            rng, step_rng = jax.random.split(rng)
            (loss, metrics), grads = jax.value_and_grad(fpo_loss_fn, has_aux=True)(
                params, params_old, step_rng
            )
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1.0/(grad_norm+1e-8)), grads)
            updates, new_opt = tx_fpo.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt, rng, metrics

        t0_fpo = time.time()
        for step in range(fpo_steps):
            if step % max(fpo_steps // 3, 1) == 0:
                params_old = jax.tree.map(lambda x: x.copy(), params)
            params, opt_fpo, rng, fpo_metrics = fpo_step(params, params_old, opt_fpo, rng)
            if step % max(fpo_steps // 4, 1) == 0 or step == fpo_steps - 1:
                print(f"    fpo step {step:4d}  ratio={float(fpo_metrics['ratio']):.3f}  "
                      f"bc={float(fpo_metrics['bc_loss']):.4f}")
        fpo_time = time.time() - t0_fpo

    train_time = bc_time + fpo_time

    final_loss = float(metrics["loss"])

    # Predict
    rng, pred_rng = jax.random.split(rng)
    bound = head.bind({"params": params}, rngs={"dropout": pred_rng})
    pred = bound.predict_action(t_out, rng=pred_rng, train=False)
    gt = actions[:, -1, :, :]
    mse = float(jnp.mean((pred - gt) ** 2))
    mae = float(jnp.mean(jnp.abs(pred - gt)))

    # Inference timing
    print(f"    Benchmarking inference ({warmup}+{trials} iters) ...")

    @jax.jit
    def predict_jit(params, rng):
        bound = head.bind({"params": params}, rngs={"dropout": rng})
        return bound.predict_action(t_out, rng=rng, train=False)

    for _ in range(warmup):
        rng, k = jax.random.split(rng)
        out = predict_jit(params, k)
        jax.block_until_ready(out)

    timings = []
    for _ in range(trials):
        rng, k = jax.random.split(rng)
        t0 = time.perf_counter()
        out = predict_jit(params, k)
        jax.block_until_ready(out)
        timings.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(timings)

    return {
        "key": key,
        "label": label,
        "flow_steps": flow_steps if key != "diffusion" else 20,
        "step_label": step_label,
        "param_count": param_count,
        "final_loss": final_loss,
        "action_mse": mse,
        "action_mae": mae,
        "avg_ms": avg_ms,
        "calls_sec": 1000.0 / avg_ms,
        "train_time": train_time,
    }


# ═══════════════════════════════════════════════════════════════════════
#  TABLE PRINTER
# ═══════════════════════════════════════════════════════════════════════

def print_table(results, title="Results"):
    labels = [f"{r['label']}" for r in results]
    mc = 24
    dc = max(26, max(len(l) for l in labels) + 2)
    sep = "  " + "-" * (mc + (dc + 3) * len(results))

    print(f"\n  {title}")
    print(sep)
    hdr = f"  {'Metric':<{mc}}"
    for r in results:
        hdr += f" | {r['label']:>{dc}}"
    print(hdr)
    print(sep)

    def row(metric, values, fmt=None, bold_best=None):
        line = f"  {metric:<{mc}}"
        best_idx = None
        if bold_best and all(isinstance(v, (int, float)) for v in values):
            best_idx = int(np.argmin(values)) if bold_best == "min" else int(np.argmax(values))
        for i, v in enumerate(values):
            s = fmt(v) if fmt else str(v)
            if i == best_idx:
                s = s + " *"
            line += f" | {s:>{dc}}"
        print(line)

    row("Total params",
        [r["param_count"] for r in results], fmt=lambda x: f"{x:,}")
    row("Inference steps",
        [r["step_label"] for r in results])
    row("Final training loss",
        [r["final_loss"] for r in results], fmt=lambda x: f"{x:.3f}", bold_best="min")
    row("Action pred MSE",
        [r["action_mse"] for r in results], fmt=lambda x: f"{x:.4f}", bold_best="min")
    row("Action pred MAE",
        [r["action_mae"] for r in results], fmt=lambda x: f"{x:.4f}", bold_best="min")
    row("Inference ms/call",
        [r["avg_ms"] for r in results], fmt=lambda x: f"{x:.2f} ms", bold_best="min")
    row("Inference calls/sec",
        [r["calls_sec"] for r in results], fmt=lambda x: f"{x:.1f}", bold_best="max")

    # Speedup vs diffusion
    diff_r = next((r for r in results if r["key"] == "diffusion"), None)
    if diff_r:
        speedups = []
        for r in results:
            if r["key"] == "diffusion":
                speedups.append("baseline")
            else:
                sp = diff_r["avg_ms"] / r["avg_ms"]
                speedups.append(f"{sp:.2f}x {'faster' if sp >= 1 else 'slower'}")
        row("Speedup vs Diffusion", speedups)

    row("Train wall-time (s)",
        [r["train_time"] for r in results], fmt=lambda x: f"{x:.1f}")

    print(sep)
    print("  (* = best in row)\n")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.steps = 300
        args.batch = 128
        args.trials = 10
        args.warmup = 3
        args.hidden = 128

    device = jax.devices()[0]
    print("=" * 75)
    print("  Action Head Benchmark (Structured Clustered Data)")
    print("=" * 75)
    print(f"  Device      : {device}")
    print(f"  Heads       : {', '.join(HEAD_LABELS[h] for h in args.heads)}")
    print(f"  Batch       : {args.batch}  (>> 64 so k-means clustering is meaningful)")
    print(f"  Obs clusters: {N_OBS_CLUSTERS}  (distinct observation modes)")
    print(f"  Steps       : {args.steps}")
    print(f"  LR          : {args.lr}")
    print(f"  Hidden dim  : {args.hidden}")
    print(f"  Num blocks  : {args.blocks}")
    print(f"  Inf trials  : {args.warmup} warmup + {args.trials} timed")
    print(f"  Action      : horizon={ACTION_HORIZON}  dim={ACTION_DIM}")
    print("=" * 75)

    # ────────────────────────────────────────────────────────────────
    # TABLE 1: Standard comparison (all heads at their default steps)
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  TABLE 1: Standard Comparison (default step counts)")
    print("=" * 75)

    results_std = []
    for h in args.heads:
        r = run_head(h, args.batch, args.hidden, args.blocks, args.steps,
                     args.lr, args.warmup, args.trials, flow_steps=10)
        results_std.append(r)

    print_table(results_std, title="TABLE 1: Standard Comparison")

    # ────────────────────────────────────────────────────────────────
    # TABLE 2: Step-count ablation (COT at fewer steps vs simple flow)
    # ────────────────────────────────────────────────────────────────
    if "flow" in args.heads and ("cot" in args.heads or "sinkhorn" in args.heads):
        print("\n" + "=" * 75)
        print("  TABLE 2: Step-Count Ablation — Can OT methods use fewer steps?")
        print("=" * 75)

        results_steps = []

        # Simple flow at 10 steps (baseline)
        r_flow_10 = next((r for r in results_std if r["key"] == "flow"), None)
        if r_flow_10:
            r_flow_10 = dict(r_flow_10)
            r_flow_10["label"] = "Simple Flow (10 steps)"
            results_steps.append(r_flow_10)

        # Sinkhorn at 10, 5, 3 steps
        if "sinkhorn" in args.heads:
            for n_steps in [10, 5, 3]:
                r_sk = next((r for r in results_std if r["key"] == "sinkhorn"), None) if n_steps == 10 else None
                if r_sk is None or n_steps != 10:
                    r_sk = run_head("sinkhorn", args.batch, args.hidden, args.blocks,
                                     args.steps, args.lr, args.warmup, args.trials,
                                     flow_steps=n_steps)
                else:
                    r_sk = dict(r_sk)
                r_sk["label"] = f"Sinkhorn ({n_steps} steps)"
                results_steps.append(r_sk)

        # COT at 10, 5, 3 steps
        if "cot" in args.heads:
            for n_steps in [10, 5, 3]:
                r_cot = next((r for r in results_std if r["key"] == "cot"), None) if n_steps == 10 else None
                if r_cot is None or n_steps != 10:
                    r_cot = run_head("cot", args.batch, args.hidden, args.blocks,
                                     args.steps, args.lr, args.warmup, args.trials,
                                     flow_steps=n_steps)
                else:
                    r_cot = dict(r_cot)
                r_cot["label"] = f"COT-Hungarian ({n_steps} steps)"
                results_steps.append(r_cot)

        # CondOT at 10, 5, 3 steps
        if "conditioned_ot" in args.heads:
            for n_steps in [10, 5, 3]:
                r_co = next((r for r in results_std if r["key"] == "conditioned_ot"), None) if n_steps == 10 else None
                if r_co is None or n_steps != 10:
                    r_co = run_head("conditioned_ot", args.batch, args.hidden, args.blocks,
                                     args.steps, args.lr, args.warmup, args.trials,
                                     flow_steps=n_steps)
                else:
                    r_co = dict(r_co)
                r_co["label"] = f"CondOT ({n_steps} steps)"
                results_steps.append(r_co)

        print_table(results_steps, title="TABLE 2: Step-Count Ablation")

    # ────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────
    print("=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    mse_best = min(results_std, key=lambda r: r["action_mse"])
    speed_best = min(results_std, key=lambda r: r["avg_ms"])
    print(f"  Best action MSE  : {mse_best['label']} ({mse_best['action_mse']:.4f})")
    print(f"  Fastest inference: {speed_best['label']} ({speed_best['avg_ms']:.2f} ms)")

    diff_r = next((r for r in results_std if r["key"] == "diffusion"), None)
    flow_r = next((r for r in results_std if r["key"] == "flow"), None)
    cot_r = next((r for r in results_std if r["key"] == "cot"), None)
    sk_r = next((r for r in results_std if r["key"] == "sinkhorn"), None)
    co_r = next((r for r in results_std if r["key"] == "conditioned_ot"), None)

    # Pairwise comparisons against diffusion
    for label, r in [("Simple Flow", flow_r), ("COT (Hungarian)", cot_r),
                      ("Sinkhorn OT", sk_r), ("Conditioned OT", co_r)]:
        if r and diff_r:
            mse_pct = (1 - r["action_mse"] / diff_r["action_mse"]) * 100
            speed_x = diff_r["avg_ms"] / r["avg_ms"]
            print(f"\n  {label} vs Diffusion:")
            print(f"    MSE  : {mse_pct:.1f}% {'lower' if mse_pct > 0 else 'higher'}")
            print(f"    Speed: {speed_x:.2f}x faster")

    # OT vs Simple Flow
    if flow_r:
        for label, r in [("COT", cot_r), ("Sinkhorn", sk_r), ("CondOT", co_r)]:
            if r and flow_r["action_mse"] > 1e-8:
                mse_pct = (1 - r["action_mse"] / flow_r["action_mse"]) * 100
                print(f"\n  {label} vs Simple Flow:")
                print(f"    MSE  : {mse_pct:.1f}% {'lower' if mse_pct > 0 else 'higher'}")

    print()
    sys.exit(0)
