"""Lightweight JAX K-means for COT Policy observation clustering.

Usage:
    centroids = fit_kmeans(embeddings, n_clusters=64, n_iters=100)
    cluster_ids = assign_clusters(embeddings, centroids)
"""

import jax
import jax.numpy as jnp
from jax import Array


def fit_kmeans(data: Array, n_clusters: int, n_iters: int = 100, seed: int = 0) -> Array:
    """Fit K-means on data using Lloyd's algorithm.

    Args:
        data: shape (N, dim)
        n_clusters: number of clusters K (clamped to N if N < K)
        n_iters: number of Lloyd iterations
        seed: random seed for centroid initialization

    Returns:
        centroids: shape (n_clusters, dim)
    """
    N, dim = data.shape
    # Clamp n_clusters to the number of available data points
    k = jnp.minimum(n_clusters, N)

    rng = jax.random.PRNGKey(seed)
    indices = jax.random.choice(rng, N, shape=(n_clusters,), replace=True)
    centroids = data[indices]  # (n_clusters, dim)

    def step(centroids, _):
        # (N, K)
        dists = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        assignments = jnp.argmin(dists, axis=-1)  # (N,)

        # Use scatter-add for JIT-compatible centroid update
        one_hot = jax.nn.one_hot(assignments, n_clusters)  # (N, K)
        counts = one_hot.sum(axis=0)  # (K,)
        sums = one_hot.T @ data  # (K, dim)
        safe_counts = jnp.maximum(counts, 1.0)[:, None]
        new_centroids = sums / safe_counts
        # Keep old centroids for empty clusters
        new_centroids = jnp.where(counts[:, None] > 0, new_centroids, centroids)
        return new_centroids, None

    centroids, _ = jax.lax.scan(step, centroids, None, length=n_iters)
    return centroids


def assign_clusters(data: Array, centroids: Array) -> Array:
    """Assign each data point to its nearest centroid.

    Args:
        data: shape (N, dim)
        centroids: shape (n_clusters, dim)

    Returns:
        cluster_ids: shape (N,) integer cluster assignments
    """
    dists = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # (N, K)
    return jnp.argmin(dists, axis=-1)                                            # (N,)
