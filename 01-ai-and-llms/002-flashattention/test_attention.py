"""Correctness checks, including cases with known answers."""

import unittest

import numpy as np

from attention import dense_attention, safe_softmax, tiled_attention


class AttentionTests(unittest.TestCase):
    def test_random_inputs_and_incomplete_blocks(self):
        for dtype in (np.float64, np.float32):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed)
                for n in (1, 7, 129):
                    q, k, v = [rng.standard_normal((n, 16)).astype(dtype)
                               for _ in range(3)]
                    reference = dense_attention(q.astype(np.float64),
                                                k.astype(np.float64),
                                                v.astype(np.float64))
                    tolerance = (1e-10, 1e-12) if dtype == np.float64 else (1e-4, 1e-5)
                    for br, bc in ((1, 17), (8, 11), (64, 64), (256, 256)):
                        with self.subTest(dtype=dtype, seed=seed, n=n, br=br, bc=bc):
                            actual = tiled_attention(q, k, v, br, bc)
                            np.testing.assert_allclose(actual, reference,
                                                       rtol=tolerance[0], atol=tolerance[1])

    def test_zero_queries_produce_mean_value(self):
        rng = np.random.default_rng(12)
        q = np.zeros((17, 8))
        k, v = [rng.standard_normal(q.shape) for _ in range(2)]
        expected = np.broadcast_to(v.mean(axis=0), v.shape)
        for actual in (dense_attention(q, k, v), tiled_attention(q, k, v, 5, 6)):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_constant_values_are_preserved(self):
        rng = np.random.default_rng(8)
        q, k = [rng.standard_normal((19, 8)) for _ in range(2)]
        v = np.broadcast_to(np.arange(8, dtype=np.float64), q.shape).copy()
        np.testing.assert_allclose(tiled_attention(q, k, v, 7, 3), v,
                                   rtol=1e-12, atol=1e-12)

    def test_softmax_shift_and_extreme_logits(self):
        x = np.array([[0.0, 1.0, 2.0]])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            np.testing.assert_allclose(safe_softmax(x + 1000), safe_softmax(x))
            np.testing.assert_allclose(safe_softmax(x - 1000), safe_softmax(x))
            # Underflow of a negligible contribution is allowed, not eliminated.
            with np.errstate(under='ignore'):
                np.testing.assert_array_equal(safe_softmax(np.array([[0., -1000.]])),
                                               np.array([[1., 0.]]))

    def test_new_larger_maximum_rescales_previous_contributions(self):
        q = np.ones((3, 1))
        k = np.array([[1000.], [1001.], [1002.]])
        v = np.array([[10.], [20.], [30.]])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            np.testing.assert_allclose(tiled_attention(q, k, v, 2, 1),
                                       dense_attention(q, k, v), rtol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
