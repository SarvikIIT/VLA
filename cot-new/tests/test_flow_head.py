"""
Smoke tests for FlowMatchActionHead.
Checks: init, loss, predict_action, gradient flow, masking, embodiment_action_dim.
"""
import jax
import jax.numpy as jnp
from octo.model.components.action_heads import FlowMatchActionHead
from octo.model.components.base import TokenGroup

# ── constants (match Octo defaults) ───────────────────────────────────────────
BATCH         = 2
WINDOW        = 2
NUM_TOKENS    = 1
EMB_DIM       = 384
ACTION_HORIZON = 4
ACTION_DIM    = 7

# ── helpers ───────────────────────────────────────────────────────────────────

def make_transformer_outputs(batch=BATCH, window=WINDOW):
    tokens = jnp.ones((batch, window, NUM_TOKENS, EMB_DIM), dtype=jnp.float32)
    mask   = jnp.ones((batch, window, NUM_TOKENS), dtype=bool)
    return {"readout_action": TokenGroup(tokens=tokens, mask=mask)}

def make_actions(batch=BATCH, window=WINDOW):
    return jax.random.normal(jax.random.PRNGKey(42), (batch, window, ACTION_HORIZON, ACTION_DIM))

def make_masks(batch=BATCH, window=WINDOW):
    timestep_pad_mask = jnp.ones((batch, window), dtype=bool)
    action_pad_mask   = jnp.ones((batch, window, ACTION_HORIZON, ACTION_DIM), dtype=bool)
    return timestep_pad_mask, action_pad_mask

def build_head_and_params(rng=jax.random.PRNGKey(0), **kwargs):
    head = FlowMatchActionHead(
        readout_key="readout_action",
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        **kwargs,
    )
    params = head.init(
        {"params": rng, "dropout": rng},
        make_transformer_outputs(),
    )
    return head, params

# ── tests ─────────────────────────────────────────────────────────────────────

def test_init():
    """Model initializes without error."""
    head, params = build_head_and_params()
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert param_count > 0
    print(f"  params: {param_count:,}")


def test_loss_shape_and_validity():
    """Loss is a finite scalar."""
    head, params = build_head_and_params()
    rng = jax.random.PRNGKey(1)
    transformer_outputs = make_transformer_outputs()
    actions = make_actions()
    timestep_pad_mask, action_pad_mask = make_masks()

    loss, metrics = head.apply(
        params,
        transformer_outputs, actions, timestep_pad_mask, action_pad_mask,
        train=True,
        rngs={"dropout": rng},
        method=head.loss,
    )

    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert "loss" in metrics and "mse" in metrics
    assert jnp.isfinite(metrics["mse"])
    print(f"  loss={loss:.4f}  mse={metrics['mse']:.4f}")


def test_predict_action_shape():
    """predict_action returns (batch, action_horizon, action_dim)."""
    head, params = build_head_and_params()
    rng = jax.random.PRNGKey(2)

    actions = head.apply(
        params,
        make_transformer_outputs(), rng,
        train=False,
        method=head.predict_action,
    )

    expected = (BATCH, ACTION_HORIZON, ACTION_DIM)
    assert actions.shape == expected, f"Expected {expected}, got {actions.shape}"
    assert jnp.all(jnp.isfinite(actions)), "Actions contain NaN/Inf"
    assert jnp.all(jnp.abs(actions) <= 5.0 + 1e-5), "Actions outside [-max_action, max_action]"
    print(f"  shape={actions.shape}  range=[{actions.min():.3f}, {actions.max():.3f}]")


def test_predict_action_sample_shape():
    """sample_shape prefix is correctly prepended."""
    head, params = build_head_and_params()
    rng = jax.random.PRNGKey(3)
    sample_shape = (4,)

    actions = head.apply(
        params,
        make_transformer_outputs(), rng,
        train=False,
        sample_shape=sample_shape,
        method=head.predict_action,
    )

    expected = sample_shape + (BATCH, ACTION_HORIZON, ACTION_DIM)
    assert actions.shape == expected, f"Expected {expected}, got {actions.shape}"
    print(f"  sample_shape={actions.shape}")


def test_gradient_flows():
    """Gradients are non-zero for all parameters."""
    head, params = build_head_and_params()
    rng = jax.random.PRNGKey(4)
    transformer_outputs = make_transformer_outputs()
    actions = make_actions()
    timestep_pad_mask, action_pad_mask = make_masks()

    def loss_fn(p):
        loss, _ = head.apply(
            p,
            transformer_outputs, actions, timestep_pad_mask, action_pad_mask,
            train=True,
            rngs={"dropout": rng},
            method=head.loss,
        )
        return loss

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    zero_leaves = [g for g in leaves if jnp.all(g == 0)]
    assert len(zero_leaves) == 0, f"{len(zero_leaves)} parameter tensors have zero gradient"
    print(f"  all {len(leaves)} param tensors have non-zero gradients")


def test_timestep_pad_mask():
    """Padding a timestep should not affect loss on other timesteps."""
    head, params = build_head_and_params()
    rng = jax.random.PRNGKey(5)
    transformer_outputs = make_transformer_outputs()
    actions = make_actions()
    _, action_pad_mask = make_masks()

    full_mask    = jnp.ones((BATCH, WINDOW), dtype=bool)
    partial_mask = full_mask.at[:, -1].set(False)

    loss_full, _    = head.apply(params, transformer_outputs, actions, full_mask,    action_pad_mask, train=True, rngs={"dropout": rng}, method=head.loss)
    loss_partial, _ = head.apply(params, transformer_outputs, actions, partial_mask, action_pad_mask, train=True, rngs={"dropout": rng}, method=head.loss)

    assert loss_full != loss_partial, "Masking a timestep had no effect on loss"
    print(f"  loss_full={loss_full:.4f}  loss_masked={loss_partial:.4f}")


def test_embodiment_action_dim():
    """Masked action dims in predict_action stay as noise (don't affect unmasked dims)."""
    head, params = build_head_and_params()
    rng = jax.random.PRNGKey(6)
    transformer_outputs = make_transformer_outputs()
    embodiment_dim = 4

    actions_full = head.apply(params, transformer_outputs, rng, train=False, method=head.predict_action)
    actions_masked = head.apply(params, transformer_outputs, rng, train=False, embodiment_action_dim=embodiment_dim, method=head.predict_action)

    assert jnp.allclose(actions_full[..., :embodiment_dim], actions_masked[..., :embodiment_dim]), \
        "embodiment_action_dim mask affected unmasked dims"
    print(f"  unmasked dims match: True  masked dims [{embodiment_dim}:] are noise")


def test_different_rngs_give_different_actions():
    """Different RNG keys should produce different action samples."""
    head, params = build_head_and_params()
    transformer_outputs = make_transformer_outputs()

    a1 = head.apply(params, transformer_outputs, jax.random.PRNGKey(0), train=False, method=head.predict_action)
    a2 = head.apply(params, transformer_outputs, jax.random.PRNGKey(1), train=False, method=head.predict_action)

    assert not jnp.allclose(a1, a2), "Different RNG keys produced identical actions"
    print(f"  actions differ across seeds: True")


# ── runner ────────────────────────────────────────────────────────────────────

TESTS = [
    test_init,
    test_loss_shape_and_validity,
    test_predict_action_shape,
    test_predict_action_sample_shape,
    test_gradient_flows,
    test_timestep_pad_mask,
    test_embodiment_action_dim,
    test_different_rngs_give_different_actions,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for test in TESTS:
        try:
            print(f"[RUN] {test.__name__}")
            test()
            print(f"[PASS] {test.__name__}\n")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}\n")
            failed += 1

    print(f"{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
