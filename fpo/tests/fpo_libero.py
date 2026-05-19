#!/usr/bin/env python3
"""
FPO on LIBERO — pixel-based BC + Flow Policy Optimization benchmark.

Mirrors tests/fpo.py (Meta-World), but for the pixel-based LIBERO
benchmark. The ONLY structural difference: LIBERO observations are camera
images, so a CNN encoder is composed in front of the (unchanged) action
heads — exactly the same way the Meta-World script wrapped the 39-dim
state as a TokenGroup, here `image -> encoder -> TokenGroup`.

Heads are reused verbatim from octo.model.components.action_heads:
  Diffusion (BC) | Flow Match (BC) | FPOActionHead (BC -> FPO RL)

LIBERO has NO scripted experts (unlike Meta-World), so demos are loaded
from the official human-teleop HDF5 datasets instead of generated.

────────────────────────────────────────────────────────────────────────
SETUP (Colab, run once — LIBERO is NOT pip-only):

    !git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    %cd LIBERO
    !pip install -q -r requirements.txt
    !pip install -q -e .
    !python benchmark_scripts/download_libero_datasets.py --datasets libero_object
    %cd ..

Then:
    !python tests/fpo_libero.py --suite libero_object
────────────────────────────────────────────────────────────────────────

NOTE: LIBERO's API and HDF5 key names vary across versions. The version-
sensitive bits are centralized in the CONFIG section below. If a run dies
with a KeyError / ImportError, paste it — only the CONFIG constants should
need changing.
"""

import argparse
import os
import sys
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as fnn

from octo.model.components.action_heads import (
    DiffusionActionHead,
    SimpleFlowMatchActionHead,
    FPOActionHead,
)
from octo.model.components.base import TokenGroup


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG  (the only version-sensitive part — edit here if LIBERO differs)
# ═══════════════════════════════════════════════════════════════════════

IMG_SIZE = 128
ACT_DIM = 7                       # LIBERO: 6-DOF OSC delta + gripper
WINDOW = 1
ACTION_HORIZON = 1

# Env observation key for the agentview RGB image (OffScreenRenderEnv).
ENV_IMG_KEY = "agentview_image"

# Candidate HDF5 paths inside a demo file for the agentview RGB stream
# (tried in order — LIBERO/robomimic naming has changed across versions).
DEMO_IMG_KEYS = ["obs/agentview_rgb", "obs/agentview_image", "obs/agentview"]
DEMO_ACT_KEY = "actions"


def libero_paths():
    """Resolve LIBERO dataset / bddl roots. Imported lazily so the file
    still compiles without LIBERO installed."""
    from libero.libero import get_libero_path
    return {
        "datasets": get_libero_path("datasets"),
        "bddl": get_libero_path("bddl_files"),
    }


def get_task_suite(suite_name):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite_name]()


def demo_file_for(suite_name, task):
    """Path to the demo HDF5 for a task. Falls back across name conventions."""
    root = libero_paths()["datasets"]
    cands = [
        os.path.join(root, suite_name, f"{task.name}_demo.hdf5"),
        os.path.join(root, suite_name, f"{task.name}.hdf5"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No demo HDF5 for task '{task.name}' in {os.path.join(root, suite_name)}. "
        f"Did download_libero_datasets.py run for suite '{suite_name}'? "
        f"Tried: {cands}"
    )


def make_env(suite_name, task):
    """Build an OffScreenRenderEnv for a LIBERO task."""
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(libero_paths()["bddl"], task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=IMG_SIZE,
        camera_widths=IMG_SIZE,
    )
    return env


def env_success(env, info):
    """Robust success check across LIBERO versions."""
    try:
        return bool(env.check_success())
    except Exception:
        pass
    if isinstance(info, dict):
        return bool(info.get("success", 0))
    return False


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FPO on LIBERO (pixel-based)")
    p.add_argument("--suite", default="libero_object",
                   choices=["libero_object", "libero_spatial", "libero_goal",
                            "libero_10", "libero_90"])
    p.add_argument("--n-tasks", type=int, default=5,
                   help="Number of tasks from the suite to benchmark")
    p.add_argument("--demos-per-task", type=int, default=40)
    p.add_argument("--bc-steps", type=int, default=4000)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feat-dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--fpo-iters", type=int, default=15)
    p.add_argument("--rollout-eps", type=int, default=20)
    p.add_argument("--fpo-epochs", type=int, default=5)
    p.add_argument("--fpo-lr", type=float, default=3e-5)
    p.add_argument("--bc-reg", type=float, default=0.5)
    p.add_argument("--clip-eps", type=float, default=0.05)
    p.add_argument("--n-mc", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--record", action="store_true",
                   help="Record one episode per method per task as a GIF")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  CNN ENCODER  (image -> feature vector; trained jointly with the head)
# ═══════════════════════════════════════════════════════════════════════

class LiberoEncoder(fnn.Module):
    feat_dim: int = 256

    @fnn.compact
    def __call__(self, img):
        # img: (B, H, W, 3) float in [0, 1]
        x = img
        for ch in (32, 64, 128, 128):
            x = fnn.Conv(ch, (3, 3), strides=(2, 2))(x)
            x = fnn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = fnn.Dense(self.feat_dim)(x)
        x = fnn.relu(x)
        return x  # (B, feat_dim)


def feats_to_t_out(feats):
    """(B, F) -> transformer_outputs dict, same shape contract the heads
    expect: tokens (B, W=1, n_tok=1, F)."""
    tokens = feats[:, None, None, :]
    mask = jnp.ones(tokens.shape[:3], dtype=bool)
    return {"obs": TokenGroup(tokens=tokens, mask=mask)}


def make_head(head_type, hidden_dim, num_blocks):
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
    if head_type == "diffusion":
        return DiffusionActionHead(**shared, diffusion_steps=20, n_diffusion_samples=1)
    if head_type == "flow":
        return SimpleFlowMatchActionHead(**shared, flow_steps=10, n_flow_samples=1)
    if head_type == "flow_fpo":
        return FPOActionHead(**shared, flow_steps=10, n_flow_samples=1)
    raise ValueError(head_type)


def init_params(encoder, head, rng):
    rng, e_rng, h_rng = jax.random.split(rng, 3)
    dummy_img = jnp.zeros((1, IMG_SIZE, IMG_SIZE, 3), jnp.float32)
    e_vars = encoder.init(e_rng, dummy_img)
    feats = encoder.apply(e_vars, dummy_img)
    dummy_t = feats_to_t_out(feats)
    h_vars = head.init({"params": h_rng, "dropout": h_rng}, dummy_t)
    return {"enc": e_vars["params"], "head": h_vars["params"]}


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# ═══════════════════════════════════════════════════════════════════════
#  DEMO LOADING  (LIBERO human-teleop HDF5 — no scripted experts)
# ═══════════════════════════════════════════════════════════════════════

def _find_img_dataset(grp):
    for k in DEMO_IMG_KEYS:
        if k in grp:
            return k
    raise KeyError(
        f"None of {DEMO_IMG_KEYS} found in demo group (have: {list(grp.keys())} "
        f"/ obs: {list(grp['obs'].keys()) if 'obs' in grp else 'n/a'}). "
        f"Update DEMO_IMG_KEYS in CONFIG."
    )


def load_demos(suite_name, task, n_demos):
    """Returns imgs (N,H,W,3) float[0,1], acts (N,ACT_DIM) float32."""
    path = demo_file_for(suite_name, task)
    imgs, acts = [], []
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda s: int(s.split("_")[-1]))
        for dk in demo_keys[:n_demos]:
            grp = data[dk]
            img_key = _find_img_dataset(grp)
            ep_img = np.asarray(grp[img_key])               # (T,H,W,3) uint8
            ep_act = np.asarray(grp[DEMO_ACT_KEY])          # (T,ACT_DIM)
            T = min(len(ep_img), len(ep_act))
            imgs.append(ep_img[:T])
            acts.append(ep_act[:T])
    imgs = np.concatenate(imgs, 0).astype(np.float32) / 255.0
    acts = np.concatenate(acts, 0).astype(np.float32)
    return imgs, acts


def preprocess_obs(obs):
    """Env obs dict -> (H,W,3) float[0,1]."""
    img = np.asarray(obs[ENV_IMG_KEY], dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return img


# ═══════════════════════════════════════════════════════════════════════
#  BC PRE-TRAINING  (encoder + head trained jointly)
# ═══════════════════════════════════════════════════════════════════════

def bc_pretrain(encoder, head, imgs, acts, steps, lr, batch_size):
    N = imgs.shape[0]
    imgs_j = jnp.asarray(imgs)
    acts_j = jnp.asarray(acts[:, None, None, :])            # (N,1,1,A)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = init_params(encoder, head, init_rng)

    tx = optax.adam(lr)
    opt_state = tx.init(params)
    tpm = jnp.ones((batch_size, WINDOW), bool)
    apm = jnp.ones((batch_size, WINDOW, ACTION_HORIZON, ACT_DIM), bool)
    head_cls = type(head)

    def loss_fn(params, img_b, act_b, rng):
        feats = encoder.apply({"params": params["enc"]}, img_b)
        t_out = feats_to_t_out(feats)
        loss, metrics = head.apply(
            {"params": params["head"]}, t_out, act_b, tpm, apm,
            train=True, rngs={"dropout": rng}, method=head_cls.loss,
        )
        return loss, metrics

    @jax.jit
    def step(params, opt_state, rng, img_b, act_b):
        rng, s_rng = jax.random.split(rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, img_b, act_b, s_rng)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, rng, metrics

    for it in range(steps):
        rng, idx_rng = jax.random.split(rng)
        idx = jax.random.randint(idx_rng, (batch_size,), 0, N)
        params, opt_state, rng, metrics = step(
            params, opt_state, rng, imgs_j[idx], acts_j[idx])
        if it % max(steps // 5, 1) == 0 or it == steps - 1:
            print(f"    BC step {it:5d}  loss={float(metrics['loss']):.4f}")
    return params


# ═══════════════════════════════════════════════════════════════════════
#  ROLLOUT / EVAL
# ═══════════════════════════════════════════════════════════════════════

def _make_predict(encoder, head):
    head_cls = type(head)

    @jax.jit
    def predict(params, img, rng):
        feats = encoder.apply({"params": params["enc"]}, img[None])
        t_out = feats_to_t_out(feats)
        act = head.apply(
            {"params": params["head"]}, t_out, rng=rng, train=False,
            method=head_cls.predict_action)
        return act[0, 0, :]
    return predict


def rollout(encoder, head, params, suite_name, task, n_eps, max_steps):
    env = make_env(suite_name, task)
    predict = _make_predict(encoder, head)
    rng = jax.random.PRNGKey(123)
    O, A, R, D = [], [], [], []
    succ = 0
    for ep in range(n_eps):
        obs = env.reset()
        done = False
        for t in range(max_steps):
            rng, k = jax.random.split(rng)
            img = preprocess_obs(obs)
            a = np.asarray(predict(params, jnp.asarray(img), k), np.float64)
            a = np.clip(a, -1.0, 1.0)
            O.append(img.astype(np.float32))
            A.append(a.astype(np.float32))
            obs, rew, done, info = env.step(a)
            R.append(float(rew))
            s = env_success(env, info)
            D.append(bool(done or s))
            if s:
                succ += 1
                break
            if done:
                break
    env.close()
    return {
        "imgs": np.asarray(O, np.float32),
        "acts": np.asarray(A, np.float32),
        "rewards": np.asarray(R, np.float32),
        "dones": np.asarray(D, bool),
        "success_rate": succ / max(n_eps, 1),
    }


def evaluate(encoder, head, params, suite_name, task, n_eps, max_steps):
    env = make_env(suite_name, task)
    predict = _make_predict(encoder, head)
    rng = jax.random.PRNGKey(7)
    succ = 0
    for ep in range(n_eps):
        obs = env.reset()
        for t in range(max_steps):
            rng, k = jax.random.split(rng)
            a = np.asarray(
                predict(params, jnp.asarray(preprocess_obs(obs)), k), np.float64)
            a = np.clip(a, -1.0, 1.0)
            obs, rew, done, info = env.step(a)
            if env_success(env, info):
                succ += 1
                break
            if done:
                break
    env.close()
    return succ / max(n_eps, 1)


def record_episode(encoder, head, params, suite_name, task, max_steps, out_path):
    """Run one episode, capture agentview frames, save as MP4. Returns success bool."""
    import imageio
    env = make_env(suite_name, task)
    predict = _make_predict(encoder, head)
    rng = jax.random.PRNGKey(42)
    obs = env.reset()
    frames = []
    success = False

    for t in range(max_steps):
        rng, k = jax.random.split(rng)

        # grab raw agentview frame (uint8 H×W×3) before stepping
        raw = np.asarray(obs[ENV_IMG_KEY])
        if raw.dtype != np.uint8:
            raw = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
        frames.append(raw)

        a = np.asarray(predict(params, jnp.asarray(preprocess_obs(obs)), k), np.float64)
        a = np.clip(a, -1.0, 1.0)
        obs, rew, done, info = env.step(a)
        if env_success(env, info):
            success = True
            break
        if done:
            break

    env.close()
    if frames:
        # imageio-ffmpeg writes MP4; falls back to GIF if ffmpeg not found
        try:
            imageio.mimwrite(out_path, frames, fps=15, quality=8,
                             codec="libx264", pixelformat="yuv420p")
        except Exception:
            gif_path = out_path.replace(".mp4", ".gif")
            imageio.mimsave(gif_path, frames, fps=15)
            out_path = gif_path
        print(f"      Video saved: {out_path}  ({'SUCCESS' if success else 'fail'})")
    return success


# ═══════════════════════════════════════════════════════════════════════
#  ADVANTAGES (returns-to-go, same as Meta-World script)
# ═══════════════════════════════════════════════════════════════════════

def compute_advantages(rewards, dones, gamma=0.99):
    N = len(rewards)
    returns = np.zeros(N, np.float32)
    run = 0.0
    for t in reversed(range(N)):
        if dones[t]:
            run = 0.0
        run = rewards[t] + gamma * run
        returns[t] = run
    adv = (returns - returns.mean()) / (returns.std() + 1e-8)
    return adv, returns


# ═══════════════════════════════════════════════════════════════════════
#  FPO  (objective owned by FPOActionHead; loop only orchestrates params)
# ═══════════════════════════════════════════════════════════════════════

def make_fpo_loss(encoder, head, n_mc, clip_eps):

    def cfm_loss_batch(params, imgs, acts, taus, epses):
        feats = encoder.apply({"params": params["enc"]}, imgs)
        t_out = feats_to_t_out(feats)
        return head.apply(
            {"params": params["head"]}, t_out, acts, taus, epses,
            method=FPOActionHead.cfm_loss_per_sample)

    def fpo_ratio(params, params_old, imgs, acts, rng):
        B = imgs.shape[0]
        diffs = []
        for _ in range(n_mc):
            rng, tk, ek = jax.random.split(rng, 3)
            taus = jax.random.uniform(tk, (B,))
            epses = jax.random.normal(ek, acts.shape)
            ln = cfm_loss_batch(params, imgs, acts, taus, epses)
            lo = cfm_loss_batch(params_old, imgs, acts, taus, epses)
            diffs.append(lo - ln)
        return FPOActionHead.fpo_ratio_from_diff(jnp.stack(diffs).mean(0))

    def fpo_loss(params, params_old, imgs, acts, adv, rng):
        ratio = fpo_ratio(params, params_old, imgs, acts, rng)
        return FPOActionHead.ppo_clip_loss(ratio, adv, clip_eps)

    return jax.jit(fpo_loss)


def fpo_finetune(encoder, head, params, suite_name, task, args, demo_imgs, demo_acts):
    fpo_loss_fn = make_fpo_loss(encoder, head, args.n_mc, args.clip_eps)
    tx = optax.adam(args.fpo_lr)
    opt_state = tx.init(params)
    head_cls = type(head)

    demo_imgs_j = jnp.asarray(demo_imgs)
    demo_acts_j = jnp.asarray(demo_acts[:, None, None, :])
    N_demo = len(demo_imgs)
    bc_b = min(32, N_demo)
    tpm = jnp.ones((bc_b, WINDOW), bool)
    apm = jnp.ones((bc_b, WINDOW, ACTION_HORIZON, ACT_DIM), bool)

    def bc_loss_fn(params, img_b, act_b, rng):
        feats = encoder.apply({"params": params["enc"]}, img_b)
        t_out = feats_to_t_out(feats)
        loss, _ = head.apply(
            {"params": params["head"]}, t_out, act_b, tpm, apm,
            train=True, rngs={"dropout": rng}, method=head_cls.loss)
        return loss

    @jax.jit
    def combined_step(params, params_old, opt_state,
                      rl_img, rl_act, adv, bc_img, bc_act, rng):
        rng, f_rng, b_rng = jax.random.split(rng, 3)

        def total(p):
            fl, fm = fpo_loss_fn(p, params_old, rl_img, rl_act, adv, f_rng)
            bl = bc_loss_fn(p, bc_img, bc_act, b_rng)
            tot = fl + args.bc_reg * bl
            fm["bc_loss"] = bl
            fm["total_loss"] = tot
            return tot, fm

        (loss, m), grads = jax.value_and_grad(total, has_aux=True)(params)
        gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1.0 / (gn + 1e-8)), grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, rng, m

    rng = jax.random.PRNGKey(99)
    best_sr, best_params = 0.0, params
    for it in range(args.fpo_iters):
        print(f"\n    FPO Iter {it+1}/{args.fpo_iters}")
        ro = rollout(encoder, head, params, suite_name, task,
                     args.rollout_eps, args.max_steps)
        sr = ro["success_rate"]
        print(f"      Rollout SR: {sr:.0%} ({len(ro['imgs'])} steps)")
        if sr > best_sr:
            best_sr = sr
            best_params = jax.tree.map(lambda x: x.copy(), params)
        if sr >= 1.0:
            print("      Perfect! Stopping early.")
            break
        if len(ro["imgs"]) < 10:
            print("      Too few transitions, skipping update")
            continue

        adv, _ = compute_advantages(ro["rewards"], ro["dones"], args.gamma)
        img_j = jnp.asarray(ro["imgs"])
        act_j = jnp.asarray(ro["acts"])
        adv_j = jnp.asarray(adv)
        params_old = jax.tree.map(lambda x: x.copy(), params)
        Nt = len(img_j)
        bs = min(32, Nt)
        for ep in range(args.fpo_epochs):
            rng, p_rng = jax.random.split(rng)
            perm = jax.random.permutation(p_rng, Nt)
            ish, ash, dsh = img_j[perm], act_j[perm], adv_j[perm]
            for st in range(0, Nt - bs + 1, bs):
                rng, bi_rng = jax.random.split(rng)
                bidx = jax.random.randint(bi_rng, (bc_b,), 0, N_demo)
                params, opt_state, rng, m = combined_step(
                    params, params_old, opt_state,
                    ish[st:st + bs], ash[st:st + bs], dsh[st:st + bs],
                    demo_imgs_j[bidx], demo_acts_j[bidx], rng)
            if ep == 0 or ep == args.fpo_epochs - 1:
                print(f"      Epoch {ep}: total={float(m['total_loss']):.4f}, "
                      f"ratio={float(m['ratio_mean']):.3f}, "
                      f"bc={float(m['bc_loss']):.4f}")
    return best_params


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, stream):
        self.stream, self.buf = stream, []

    def write(self, s):
        self.stream.write(s)
        self.buf.append(s)

    def flush(self):
        self.stream.flush()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.n_tasks = 2
        args.demos_per_task = 20
        args.bc_steps = 1500
        args.fpo_iters = 5
        args.rollout_eps = 10
        args.eval_episodes = 10

    _tee = _Tee(sys.stdout)
    sys.stdout = _tee

    METHODS = [
        ("diffusion", "Diffusion (BC)"),
        ("flow",      "Flow Match (BC)"),
        ("flow_fpo",  "Flow+FPO (Ours)"),
    ]
    keys = [m[0] for m in METHODS]
    labels = {m[0]: m[1] for m in METHODS}

    print("=" * 90)
    print(f"  LIBERO ({args.suite}) — Pixel-based FPO Benchmark")
    print("  Diffusion (BC) | Flow Match (BC) | Flow+FPO (Ours)")
    print("=" * 90)
    print(f"  Device       : {jax.devices()[0]}")
    print(f"  Suite        : {args.suite}  (first {args.n_tasks} tasks)")
    print(f"  Demos/task   : {args.demos_per_task}")
    print(f"  BC steps     : {args.bc_steps}")
    print(f"  FPO iters    : {args.fpo_iters}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print("=" * 90)

    suite = get_task_suite(args.suite)
    n_tasks = min(args.n_tasks, suite.n_tasks)

    encoders = {k: LiberoEncoder(args.feat_dim) for k in keys}
    heads = {k: make_head(k, args.hidden, args.blocks) for k in keys}

    results = {}
    inf_heads = {}

    for ti in range(n_tasks):
        task = suite.get_task(ti)
        print(f"\n{'='*90}\n  Task {ti}: {task.name}\n{'='*90}")

        print(f"\n  [1] Loading {args.demos_per_task} LIBERO demos ...")
        imgs, acts = load_demos(args.suite, task, args.demos_per_task)
        print(f"      {len(imgs)} transitions  img={imgs.shape[1:]}  act={acts.shape[1:]}")

        tr = {}

        for ht in ("diffusion", "flow"):
            print(f"\n  [+] Training {labels[ht]} ...")
            t0 = _time.time()
            p = bc_pretrain(encoders[ht], heads[ht], imgs, acts,
                            args.bc_steps, args.bc_lr, args.batch_size)
            sr = evaluate(encoders[ht], heads[ht], p, args.suite, task,
                          args.eval_episodes, args.max_steps)
            print(f"      {labels[ht]} SR: {sr:.0%}  ({_time.time()-t0:.0f}s)")
            tr[ht] = sr
            if ti == 0:
                inf_heads[ht] = (encoders[ht], heads[ht], p)
            if args.record:
                gif = os.path.join(os.path.dirname(__file__), f"sim_{task.name}_{ht}.gif")
                record_episode(encoders[ht], heads[ht], p, args.suite, task,
                               args.max_steps, gif)

        print(f"\n  [+] BC pre-training FPOActionHead ...")
        fpo_p = bc_pretrain(encoders["flow_fpo"], heads["flow_fpo"], imgs, acts,
                            args.bc_steps, args.bc_lr, args.batch_size)
        print(f"\n  [+] FPO fine-tuning ({args.fpo_iters} iters) ...")
        fpo_p = fpo_finetune(encoders["flow_fpo"], heads["flow_fpo"], fpo_p,
                             args.suite, task, args, imgs, acts)
        sr = evaluate(encoders["flow_fpo"], heads["flow_fpo"], fpo_p,
                      args.suite, task, args.eval_episodes, args.max_steps)
        print(f"      {labels['flow_fpo']} SR: {sr:.0%}")
        tr["flow_fpo"] = sr
        if ti == 0:
            inf_heads["flow_fpo"] = (encoders["flow_fpo"], heads["flow_fpo"], fpo_p)
        if args.record:
            gif = os.path.join(os.path.dirname(__file__), f"sim_{task.name}_fpo.gif")
            record_episode(encoders["flow_fpo"], heads["flow_fpo"], fpo_p,
                           args.suite, task, args.max_steps, gif)

        results[task.name] = tr

    # ── TABLE ──
    print("\n\n" + "=" * 90)
    print(f"  TABLE: LIBERO ({args.suite}) SUCCESS RATE (%)")
    print("=" * 90)
    tc, hc = 34, 16
    sep = "  " + "-" * (tc + (hc + 3) * len(keys))
    print(sep)
    hdr = f"  {'Task':<{tc}}"
    for k in keys:
        hdr += f" | {labels[k]:>{hc}}"
    print(hdr)
    print(sep)
    agg = {k: [] for k in keys}
    for tn, r in results.items():
        srs = [r.get(k, 0) for k in keys]
        best = max(srs) if srs else 0
        row = f"  {tn:<{tc}}"
        for i, k in enumerate(keys):
            s = f"{r.get(k, 0):.0%}"
            if srs[i] == best and best > 0:
                s += " *"
            row += f" | {s:>{hc}}"
            agg[k].append(r.get(k, 0))
        print(row)
    print(sep)
    avgs = {k: float(np.mean(agg[k])) if agg[k] else 0.0 for k in keys}
    best_avg = max(avgs.values()) if avgs else 0
    row = f"  {'AVERAGE':<{tc}}"
    for k in keys:
        s = f"{avgs[k]:.1%}"
        if avgs[k] == best_avg:
            s += " *"
        row += f" | {s:>{hc}}"
    print(row)
    print(sep + "\n  (* = best in row)")

    print("\n  Note: same network + identical inference cost across flow/FPO;")
    print("  only the training objective differs. Diffusion = 20 DDPM steps,")
    print("  flow/FPO = 10 Euler steps (~2x faster), as in the Meta-World run.")

    sys.stdout = _tee.stream
    out = "".join(_tee.buf)
    path = os.path.join(os.path.dirname(__file__), f"results_libero_{args.suite}.txt")
    i = out.find("TABLE:")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out[i:] if i != -1 else out)
    print(f"  Results table saved to: {path}")
    sys.exit(0)
