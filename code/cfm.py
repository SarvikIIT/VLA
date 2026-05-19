import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt

# ── data ──────────────────────────────────────────────────────────────────────

def sample_8gaussians(key, n):
    angles = jnp.arange(8) * (2 * jnp.pi / 8)
    centers = jnp.stack([5.0 * jnp.cos(angles), 5.0 * jnp.sin(angles)], axis=-1)  # (8, 2)
    key1, key2 = jax.random.split(key)
    which = jax.random.randint(key1, (n,), 0, 8)
    noise = jax.random.normal(key2, (n, 2)) * 0.3
    return centers[which] + noise  # (n, 2)

# ── model ─────────────────────────────────────────────────────────────────────

class VelocityMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x); x = nn.swish(x)
        x = nn.Dense(256)(x); x = nn.swish(x)
        x = nn.Dense(256)(x); x = nn.swish(x)
        x = nn.Dense(2)(x)
        return x

# ── training ──────────────────────────────────────────────────────────────────

def make_train_step(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, key):
        key_a, key_eps, key_tau = jax.random.split(key, 3)

        a   = sample_8gaussians(key_a, 2048)
        eps = jax.random.normal(key_eps, (2048, 2))
        tau = jax.random.uniform(key_tau, (2048, 1))

        a_tau = (1 - tau) * eps + tau * a
        v_target = a - eps

        inp = jnp.concatenate([a_tau, tau], axis=-1)

        def loss_fn(p):
            v_pred = model.apply(p, inp)
            return jnp.mean((v_pred - v_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss
    return train_step

# ── inference ─────────────────────────────────────────────────────────────────

def euler_sample(model, params, key, n=1000, steps=10):
    x = jax.random.normal(key, (n, 2))
    dt = 1.0 / steps
    for i in range(steps):
        tau = jnp.full((n, 1), i * dt)
        inp = jnp.concatenate([x, tau], axis=-1)
        v = model.apply(params, inp)
        x = x + dt * v
    return x

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    key = jax.random.PRNGKey(0)
    model = VelocityMLP()

    key, init_key, train_key, sample_key = jax.random.split(key, 4)
    dummy_inp = jnp.zeros((1, 3))
    params = model.init(init_key, dummy_inp)

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)

    print("training...")
    for step in range(5000):
        train_key, k = jax.random.split(train_key)
        params, opt_state, loss = train_step(params, opt_state, k)
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss:.4f}")

    print("sampling...")
    pts = euler_sample(model, params, sample_key, n=2000, steps=10)
    pts = jnp.array(pts)

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5)
    plt.title("CFM 2D toy — 8 Gaussians (10 Euler steps)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("cfm_output.png", dpi=120)
    plt.show()
    print("saved cfm_output.png")

if __name__ == "__main__":
    main()
