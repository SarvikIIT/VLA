"""FPO building blocks — math primitives for Flow Policy Optimization.

Paper: "Flow Matching Policy Gradients" (arXiv:2507.21053)

This module mirrors `flow.py`: it holds the pure, model-agnostic pieces
that the FPO action head composes. Nothing here knows about the velocity
network or Octo's transformer outputs — those live in `FPOActionHead`.

Three primitives:

    cfm_loss_from_pred(v_pred, v_target) -> (B,)
        Per-sample flow-matching loss given a predicted velocity and the
        target velocity (act - eps). The network call itself stays in the
        head because it needs `transformer_outputs`.

    fpo_ratio_from_diff(avg_loss_diff) -> ratio
        r_hat = exp(L_old - L_new), clamped for numerical stability.
        `avg_loss_diff` is the MC-averaged (L_old - L_new) for one sample.

    ppo_clip_loss(ratio, advantages, clip_eps) -> (loss, metrics)
        PPO-clip surrogate using the FPO ratio.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def cfm_loss_from_pred(v_pred: ArrayLike, v_target: ArrayLike) -> jax.Array:
    """Per-sample flow-matching loss: ||v_pred - v_target||^2 along action dim.

    v_pred, v_target: (B, A)  ->  returns (B,)
    """
    return jnp.sum((v_pred - v_target) ** 2, axis=-1)


def fpo_ratio_from_diff(avg_loss_diff: ArrayLike) -> jax.Array:
    """r_hat = exp(L_old - L_new), clamped to [-5, 5] before the exp.

    `avg_loss_diff` is already MC-averaged across (tau, eps) draws.
    """
    return jnp.exp(jnp.clip(avg_loss_diff, -5.0, 5.0))


def ppo_clip_loss(
    ratio: ArrayLike, advantages: ArrayLike, clip_eps: float,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """PPO-clip surrogate using the FPO ratio.

    loss = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)]
    """
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
    metrics = {
        "fpo_loss": loss,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
    }
    return loss, metrics
