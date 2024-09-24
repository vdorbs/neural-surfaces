from neural_surfaces.utils import sphere_exp, sphere_log
from torch import float64, nan, ones, ones_like, randn, randn_like, zeros, zeros_like
from torch.linalg import norm
from torch.testing import assert_close
from unittest import main, TestCase


class TestUtils(TestCase):
    def test_exp_and_log(self):
        xs = randn(100, 3, dtype=float64)
        xs /= norm(xs, dim=-1, keepdims=True)

        vs = randn_like(xs)
        vs -= (xs * vs).sum(dim=-1, keepdims=True) * xs
        vs /= norm(vs, dim=-1).max()

        assert_close(norm(sphere_exp(xs, vs), dim=-1), ones(len(xs), dtype=float64))
        assert_close(vs, sphere_log(xs, sphere_exp(xs, vs)))

        ys = randn_like(xs)
        ys /= norm(ys, dim=-1, keepdims=True)

        assert_close((xs * sphere_log(xs, ys)).sum(dim=-1), zeros(len(xs), dtype=float64))
        assert_close(ys, sphere_exp(xs, sphere_log(xs, ys)))

        assert_close(sphere_log(xs, xs), zeros_like(xs))


if __name__ == '__main__':
    main()
