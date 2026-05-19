import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt


def sample_8gaussians(key, n):
    angles = jnp.arange(8) * (2 * jnp.pi / 8)
    centers = jnp.stack([5.0 * jnp.cos(angles), 5.0 * jnp.sin(angles)], axis=-1) # This is the center of each block of the 8 gaussians
    key1, key2 = jax.random.split(key)
    which = jax.random.randint(key1, (n,), 0, 8) # This is the index of the gaussian that each point belongs to
    noise = jax.random.normal(key2, (n, 2)) * 0.3
    return centers[which] + noise # This is the final data point, which is the center of the gaussian plus some noise


class VelocityMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x) # Input is 3 dimensional, output is 256 dimensional
        x = nn.swish(x)
        x = nn.Dense(256)(x)
        x = nn.swish(x)
        x = nn.Dense(256)(x)
        x = nn.swish(x)
        x = nn.Dense(2)(x)
        return x
def make_train_step(model, optimizer):
    @jax.jit
    def train_steps(params, opt_state, key):
        key1, key2, key3 = jax.random.split(key, 3)
        a = sample_8gaussians(key1, 2048)
        eps = jax.random.normal(key2, (2048, 2)) # noise for N(0,1) for 2048 points and x and y coordinates
        tau = jax.random.uniform(key3, (2048, 1)) # the t parameter for each point, which is between 0 and 1
        a_tau = (1 - tau) * eps + tau * a # the point at time t, which is a linear interpolation between the noise and the original point
        v_target = a - eps # correct value of the velocity
        inp = jnp.concatenate([a_tau,tau], axis=-1) # input to the model is the point at time t and the time t itself

        def loss_fn(p):
            v_pred = model.apply(p, inp) # predicted velocity from the model
            return jnp.mean((v_pred - v_target) ** 2) # MSE

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, optimizer_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, optimizer_state_new, loss
    return train_steps

def final(model, params, key, n = 1000, steps = 10):
    x = jax.random.normal(key, (n,2)) # start with random points
    dt = 1.0 / steps
    for i in range(steps):
        tau = jnp.full((n,1), i * dt) # create a tau vector for the current time step
        inp = jnp.concatenate([x, tau], axis=-1) # input to the model is the current point and the time
        v = model.apply(params, inp) # get the velocity from the model
        x = x + v * dt # update the point using euler method
    return x

def main():
    key = jax.random.PRNGKey(0)
    model = VelocityMLP()

    key, init_key, train_key, sample_key = jax.random.split(key, 4)
    dummy_inp = jnp.zeros((1, 3)) # dummy input for initializing the model, which is 3 dimensional (x, y, tau)
    params = model.init(init_key, dummy_inp) # initialize the model parameters
    optimizer = optax.adam(3e-4) # create the optimizer
    opt_state = optimizer.init(params) # initialize the optimizer state
    train_step = make_train_step(model, optimizer) # create the training step function

    for step in range(7000):
        train_key, k = jax.random.split(train_key)
        params, opt_state, loss = train_step(params, opt_state, k)
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss:.4f}")

    pts = final(model, params, sample_key)
    pts = jnp.array(pts)

    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], s=5, alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()