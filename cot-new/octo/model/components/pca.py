"""Lightweight JAX PCA for COT Policy observation embedding compression.

Usage:
    mean, components = fit_pca(embeddings, n_components=32)
    reduced = transform_pca(embeddings, mean, components)
"""

import jax.numpy as jnp
from jax import Array


def fit_pca(data: Array, n_components: int) -> tuple[Array, Array]:
    """Compute PCA components from data using SVD.

    Args:
        data: shape (N, emb_dim) — observation embeddings, one per sample
        n_components: number of principal components to keep (clamped to min(N, emb_dim))

    Returns:
        mean: shape (emb_dim,) — per-feature mean, used to center data
        components: shape (n_components, emb_dim) — top-k eigenvectors (rows),
            zero-padded if fewer than n_components are available
    """
    N, emb_dim = data.shape
    mean = data.mean(axis=0)                         # (emb_dim,)
    centered = data - mean                           # (N, emb_dim)
    _, _, Vt = jnp.linalg.svd(centered, full_matrices=False)
    # Vt has shape (min(N, emb_dim), emb_dim); pad if needed
    k_avail = Vt.shape[0]
    if k_avail < n_components:
        pad = jnp.zeros((n_components - k_avail, emb_dim), dtype=Vt.dtype)
        Vt = jnp.concatenate([Vt, pad], axis=0)
    components = Vt[:n_components]                   # (n_components, emb_dim)
    return mean, components


def transform_pca(data: Array, mean: Array, components: Array) -> Array:
    """Project data onto pre-computed PCA components.

    Args:
        data: shape (N, emb_dim)
        mean: shape (emb_dim,) — from fit_pca
        components: shape (n_components, emb_dim) — from fit_pca

    Returns:
        reduced: shape (N, n_components)
    """
    centered = data - mean                           # (N, emb_dim)
    return centered @ components.T                   # (N, n_components)
