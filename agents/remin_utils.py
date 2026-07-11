import jax.numpy as jnp
from jax.scipy.special import gammaln


def log_comb_float(n, k):
    """Computes log of binomial coefficients in a stable way."""
    n_arr = jnp.asarray(n, dtype=jnp.float32)
    k_arr = jnp.asarray(k, dtype=jnp.float32)
    valid = (k_arr >= 0.0) & (n_arr >= k_arr)

    safe_n = jnp.where(valid, n_arr, 0.0)
    safe_k = jnp.where(valid, k_arr, 0.0)
    log_comb = gammaln(safe_n + 1.0) - gammaln(safe_k + 1.0) - gammaln(safe_n - safe_k + 1.0)

    return jnp.where(valid, log_comb, -jnp.inf)


def compute_remin(values: jnp.ndarray, m: int, axis: int = 0) -> jnp.ndarray:
    """Unbiased estimator of the expected worst-of-m order statistic.

    Args:
        values: Input array.
        m: Retry parameter. Must satisfy 1 <= m <= values.shape[axis].
        axis: Axis containing IID samples.
    Returns:
        Estimated expected minimum over m draws along the given axis.
    """
    values = jnp.moveaxis(values, axis, 0)
    num_samples = values.shape[0]
    if num_samples < m:
        raise ValueError(f'Number of samples {num_samples} must be greater than or equal to m={m}.')

    sorted_values = jnp.sort(values, axis=0)
    indices = jnp.arange(num_samples, dtype=jnp.float32)

    # The i-th smallest value is the minimum when the remaining m-1 elements are
    # chosen from the larger num_samples - i - 1 values.
    log_weights = log_comb_float(num_samples - indices - 1.0, m - 1) - log_comb_float(num_samples, m)
    weights = jnp.exp(log_weights)
    weights = jnp.nan_to_num(weights, nan=0.0)

    weight_shape = (num_samples,) + (1,) * (sorted_values.ndim - 1)
    return jnp.sum(sorted_values * weights.reshape(weight_shape), axis=0)
