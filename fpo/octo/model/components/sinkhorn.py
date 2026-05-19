"""Differentiable Optimal Transport via Sinkhorn iterations.

Unlike scipy.optimize.linear_sum_assignment (Hungarian), Sinkhorn:
  - Is fully differentiable — gradients flow through the coupling
  - Runs in pure JAX — JIT-compilable, no Python callbacks
  - Produces soft couplings that are more expressive

Includes:
  - Standard Sinkhorn OT (action-space cost only)
  - History-Conditioned OT (action + observation cost)

Usage:
    coupling = sinkhorn_coupling(cost_matrix, reg=0.1, n_iters=20)
    matched = sinkhorn_match_noise(noise, actions)
    matched = conditioned_sinkhorn_match(noise, actions, obs_embeds, obs_weight=1.0)
"""

import jax
import jax.numpy as jnp
from jax import Array


def sinkhorn_coupling(
    cost: Array,
    reg: float = 0.1,
    n_iters: int = 50,
) -> Array:
    """Compute Sinkhorn optimal transport coupling.

    Args:
        cost: (N, M) cost matrix. Lower = better match.
        reg: Entropic regularization. Lower = sharper coupling (closer to
             Hungarian). Higher = smoother. 0.05-0.5 is typical.
        n_iters: Sinkhorn iterations. 20-50 is usually enough.

    Returns:
        coupling: (N, M) doubly-stochastic transport plan.
            Rows sum to 1/N, columns sum to 1/M.
            Fully differentiable w.r.t. cost.
    """
    N, M = cost.shape
    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / reg  # (N, M) log of Gibbs kernel

    # Initialize dual variables
    log_u = jnp.zeros(N)  # (N,)
    log_v = jnp.zeros(M)  # (M,)

    def sinkhorn_step(carry, _):
        log_u, log_v = carry
        # Row normalization
        log_u = -jax.nn.logsumexp(log_K + log_v[None, :], axis=1)
        # Column normalization
        log_v = -jax.nn.logsumexp(log_K + log_u[:, None], axis=0)
        return (log_u, log_v), None

    (log_u, log_v), _ = jax.lax.scan(
        sinkhorn_step, (log_u, log_v), None, length=n_iters
    )

    # Compute coupling: T_ij = exp(log_u_i + log_K_ij + log_v_j)
    log_T = log_u[:, None] + log_K + log_v[None, :]
    coupling = jnp.exp(log_T)

    # Normalize to ensure exact doubly-stochastic
    coupling = coupling / (coupling.sum() + 1e-12) * N

    return coupling


def sinkhorn_match_noise(
    noise: Array,
    actions: Array,
    reg: float = 0.05,
    n_iters: int = 50,
) -> Array:
    """Match noise samples to action targets using Sinkhorn OT.

    Standard action-space cost only.

    Args:
        noise: (N, D) noise samples
        actions: (N, D) target actions
        reg: Sinkhorn regularization (lower = sharper)
        n_iters: Sinkhorn iterations

    Returns:
        matched_noise: (N, D) noise reordered by OT coupling.
    """
    cost = jnp.sum((noise[:, None, :] - actions[None, :, :]) ** 2, axis=-1)
    coupling = sinkhorn_coupling(cost, reg=reg, n_iters=n_iters)
    assignments = jnp.argmax(coupling, axis=0)
    matched_noise = noise[assignments]
    return matched_noise


def conditioned_sinkhorn_match(
    noise: Array,
    actions: Array,
    obs_embeddings: Array,
    obs_weight: float = 1.0,
    history_embeddings: Array = None,
    history_weight: float = 0.5,
    reg: float = 0.05,
    n_iters: int = 50,
) -> Array:
    """History-conditioned Sinkhorn OT for flow matching.

    Cost = action_cost + obs_weight * obs_cost + history_weight * history_cost

    This prevents OT from matching noise across different contexts:
      - Different tasks get different couplings (obs_cost)
      - Different temporal phases get different couplings (history_cost)
      - Within each context, action-space OT straightens flow paths

    Args:
        noise: (N, D_action) noise samples
        actions: (N, D_action) target actions
        obs_embeddings: (N, D_obs) observation embeddings for each sample.
            obs_cost[i,j] = ||obs_i - obs_j||² penalizes cross-context matching.
        obs_weight: Weight for observation cost term. Higher = more context-aware.
        history_embeddings: (N, D_hist) optional temporal/history features.
            If provided, adds temporal consistency penalty.
        history_weight: Weight for history cost term.
        reg: Sinkhorn regularization
        n_iters: Sinkhorn iterations

    Returns:
        matched_noise: (N, D_action) noise reordered by conditioned OT.
    """
    # Action-space cost: ||noise_i - action_j||²
    action_cost = jnp.sum(
        (noise[:, None, :] - actions[None, :, :]) ** 2, axis=-1
    )  # (N, N)

    # Normalize action cost to unit scale
    action_scale = jnp.maximum(action_cost.mean(), 1e-8)
    action_cost = action_cost / action_scale

    # Observation-space cost: ||obs_i - obs_j||²
    # Penalizes matching noise_i to action_j when obs_i ≠ obs_j
    # (i.e., when the noise came from a different context than the action)
    obs_cost = jnp.sum(
        (obs_embeddings[:, None, :] - obs_embeddings[None, :, :]) ** 2, axis=-1
    )  # (N, N)
    obs_scale = jnp.maximum(obs_cost.mean(), 1e-8)
    obs_cost = obs_cost / obs_scale

    # Combined cost
    cost = action_cost + obs_weight * obs_cost

    # Optional history/temporal cost
    if history_embeddings is not None:
        hist_cost = jnp.sum(
            (history_embeddings[:, None, :] - history_embeddings[None, :, :]) ** 2,
            axis=-1,
        )
        hist_scale = jnp.maximum(hist_cost.mean(), 1e-8)
        hist_cost = hist_cost / hist_scale
        cost = cost + history_weight * hist_cost

    coupling = sinkhorn_coupling(cost, reg=reg, n_iters=n_iters)
    assignments = jnp.argmax(coupling, axis=0)
    matched_noise = noise[assignments]

    return matched_noise
