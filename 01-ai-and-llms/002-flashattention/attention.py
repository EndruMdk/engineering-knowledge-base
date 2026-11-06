"""CPU demonstrations of dense and tiled attention; no mask, dropout or backward."""

import numpy as np


def safe_softmax(x):
    """Normalize each row of finite logits without overflowing the exponential."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def _validate(q, k, v):
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K and V must all have the same (N, d) shape.")
    if 0 in q.shape:
        raise ValueError("N and d must be positive.")
    if q.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError("Use float32 or float64 arrays.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K and V must have the same dtype.")


def dense_attention(q, k, v):
    """Reference: materialize the complete score and probability matrices."""
    _validate(q, k, v)
    scale = q.dtype.type(1 / np.sqrt(q.shape[1]))
    scores = (q @ k.T) * scale
    probabilities = safe_softmax(scores)
    return probabilities @ v


def tiled_attention(q, k, v, query_block=64, key_block=64):
    """Online softmax with an unnormalized output accumulator.

    Inputs must be finite and produce representable dot products. This is an
    educational NumPy implementation, not a fused FlashAttention GPU kernel.
    """
    _validate(q, k, v)
    if query_block <= 0 or key_block <= 0:
        raise ValueError("Block sizes must be positive.")
    n, d = q.shape
    scale = q.dtype.type(1 / np.sqrt(d))
    output = np.empty_like(q)

    for row in range(0, n, query_block):
        qi = q[row:row + query_block]
        # Each query row has its own maximum, denominator and output numerator.
        m = np.full((len(qi), 1), -np.inf, dtype=q.dtype)
        l = np.zeros((len(qi), 1), dtype=q.dtype)
        acc = np.zeros_like(qi)

        for col in range(0, n, key_block):
            kj = k[col:col + key_block]
            vj = v[col:col + key_block]
            scores = (qi @ kj.T) * scale

            new_m = np.maximum(m, np.max(scores, axis=1, keepdims=True))
            alpha = np.exp(m - new_m)

            # Reuse this tile for unnormalized weights; never build an N x N P.
            scores -= new_m
            np.exp(scores, out=scores)

            l = alpha * l + np.sum(scores, axis=1, keepdims=True)
            acc = alpha * acc + scores @ vj
            m = new_m
            del scores

        output[row:row + len(qi)] = acc / l

    return output
