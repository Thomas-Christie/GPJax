import jax.numpy as jnp
import paramax
from paramax import AbstractUnwrappable
import pytest


def test_positive_real_unwraps_to_array():
    from gpjax.parameters import PositiveReal

    p = PositiveReal(jnp.array(2.0))
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert isinstance(val, jnp.ndarray)
    assert jnp.allclose(val, jnp.array(2.0), atol=1e-5)


def test_positive_real_preserves_positivity():
    from gpjax.parameters import PositiveReal

    p = PositiveReal(jnp.array(0.01))
    val = p.unwrap()
    assert val > 0


def test_non_negative_real_unwraps():
    from gpjax.parameters import NonNegativeReal

    p = NonNegativeReal(jnp.array(0.5))
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert jnp.allclose(val, jnp.array(0.5), atol=1e-5)


def test_real_unwraps_to_identity():
    from gpjax.parameters import Real

    p = Real(jnp.array(-3.0))
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert jnp.allclose(val, jnp.array(-3.0))


def test_sigmoid_bounded_unwraps_within_bounds():
    from gpjax.parameters import SigmoidBounded

    p = SigmoidBounded(jnp.array(0.5), low=0.0, high=1.0)
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert 0.0 <= val <= 1.0
    assert jnp.allclose(val, jnp.array(0.5), atol=1e-5)


def test_sigmoid_bounded_custom_bounds():
    from gpjax.parameters import SigmoidBounded

    p = SigmoidBounded(jnp.array(5.0), low=2.0, high=8.0)
    val = p.unwrap()
    assert 2.0 <= val <= 8.0
    assert jnp.allclose(val, jnp.array(5.0), atol=1e-5)


def test_lower_triangular_unwraps():
    from gpjax.parameters import LowerTriangular

    L = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    p = LowerTriangular(L)
    assert isinstance(p, AbstractUnwrappable)
    val = p.unwrap()
    assert jnp.allclose(val, L, atol=1e-5)


def test_paramax_unwrap_on_module():
    """unwrap() on an eqx.Module containing parameters produces plain arrays."""
    import equinox as eqx
    from gpjax.parameters import PositiveReal, Real

    class Dummy(eqx.Module):
        a: PositiveReal
        b: Real

    m = Dummy(a=PositiveReal(jnp.array(2.0)), b=Real(jnp.array(-1.0)))
    unwrapped = paramax.unwrap(m)
    assert isinstance(unwrapped.a, jnp.ndarray)
    assert isinstance(unwrapped.b, jnp.ndarray)
    assert jnp.allclose(unwrapped.a, jnp.array(2.0), atol=1e-5)
    assert jnp.allclose(unwrapped.b, jnp.array(-1.0))


def test_val_unwraps_nested_wrappers():
    """_val must resolve wrappers nested *inside* a parameter, not just the outer node.

    paramax.non_trainable wraps array leaves, so freezing a parameter yields e.g.
    PositiveReal(_unconstrained=NonTrainable(...)). A single .unwrap() call would
    feed the NonTrainable object straight into the softplus bijection of the
    PositiveReal.
    """
    from gpjax.parameters import (
        PositiveReal,
        _val,
    )

    p = paramax.non_trainable(PositiveReal(jnp.array(2.0)))
    assert isinstance(p._unconstrained, paramax.NonTrainable)
    assert jnp.equal(_val(p), jnp.array(2.0))


def test_lower_triangular_positive_diagonal():
    """LowerTriangular enforces positive diagonal via softplus."""
    from gpjax.parameters import LowerTriangular

    L = jnp.array([[2.0, 0.0], [0.5, 3.0]])
    p = LowerTriangular(L)
    val = p.unwrap()
    assert jnp.allclose(val, L, atol=1e-5)
    assert val[0, 0] > 0
    assert val[1, 1] > 0


@pytest.mark.parametrize(
    "factory, storage_field",
    [
        (
            lambda v: __import__(
                "gpjax.parameters", fromlist=["PositiveReal"]
            ).PositiveReal(v),
            "_unconstrained",
        ),
        (
            lambda v: __import__(
                "gpjax.parameters", fromlist=["NonNegativeReal"]
            ).NonNegativeReal(v),
            "_unconstrained",
        ),
        (lambda v: __import__("gpjax.parameters", fromlist=["Real"]).Real(v), "value"),
        (
            lambda v: __import__(
                "gpjax.parameters", fromlist=["SigmoidBounded"]
            ).SigmoidBounded(v, low=0.0, high=1.0),
            "_unconstrained",
        ),
    ],
)
def test_scalar_parameters_preserve_float32(factory, storage_field):
    """Regression: parameters should preserve float32 inputs (discussion #628)."""
    value = jnp.asarray(0.5, dtype=jnp.float32)
    p = factory(value)
    assert getattr(p, storage_field).dtype == jnp.float32
    assert p.unwrap().dtype == jnp.float32


def test_lower_triangular_preserves_float32():
    """Regression: LowerTriangular preserves float32 inputs (discussion #628).

    Relies on the local dtype-preserving variant of
    SoftplusLowerCholeskyTransform (temporary workaround for numpyro's
    untyped allocations in vec_to_tril_matrix).
    """
    from gpjax.parameters import LowerTriangular

    L = jnp.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
    p = LowerTriangular(L)
    assert p._flat.dtype == jnp.float32
    val = p.unwrap()
    assert val.dtype == jnp.float32
    assert jnp.allclose(val, L, atol=1e-5)


def test_coregionalization_matrix():
    """CoregionalizationMatrix produces a PSD matrix."""
    from gpjax.parameters import CoregionalizationMatrix
    import jax.random as jr

    cm = CoregionalizationMatrix(num_outputs=3, rank=2, key=jr.key(0))
    B = cm.B
    assert B.shape == (3, 3)
    # PSD check: all eigenvalues >= 0
    eigvals = jnp.linalg.eigvalsh(B)
    assert jnp.all(eigvals >= -1e-6)


def test_value_resolves_parameters_and_passes_arrays_through():
    from gpjax.parameters import PositiveReal, value

    p = PositiveReal(jnp.array(2.0))
    assert jnp.allclose(value(p), jnp.array(2.0), atol=1e-5)

    raw = jnp.array(3.0)
    assert value(raw) is raw


def test_value_preserves_the_gradient_block_on_frozen_parameters():
    """`value` must resolve through a `non_trainable` wrapper, keeping both the
    constrained value and the stop-gradient."""
    from gpjax.parameters import PositiveReal, value
    import jax

    frozen = paramax.non_trainable(PositiveReal(jnp.array(2.0)))
    assert jnp.allclose(value(frozen), jnp.array(2.0), atol=1e-5)

    grad = jax.grad(lambda p: value(p) ** 2)(frozen)
    assert all(jnp.allclose(leaf, 0.0) for leaf in jax.tree_util.tree_leaves(grad))


def test_parameter_prior_defaults_to_none():
    from gpjax.parameters import (
        LowerTriangular,
        NonNegativeReal,
        PositiveReal,
        Real,
        SigmoidBounded,
    )

    assert PositiveReal(jnp.array(1.0)).prior is None
    assert NonNegativeReal(jnp.array(1.0)).prior is None
    assert Real(jnp.array(1.0)).prior is None
    assert SigmoidBounded(jnp.array(0.5)).prior is None
    assert LowerTriangular(jnp.eye(2)).prior is None


def test_parameter_prior_is_static_and_not_optimised():
    """A numpyro distribution is itself a pytree of arrays. The prior must be a
    static field, or `eqx.partition(model, eqx.is_array)` would sweep the
    prior's own hyperparameters into the trainable partition."""
    import equinox as eqx
    from gpjax.parameters import PositiveReal
    import jax
    import numpyro.distributions as dist

    p = PositiveReal(jnp.array(1.0), prior=dist.LogNormal(jnp.log(3.0), 0.15))
    params, static = eqx.partition(p, eqx.is_array)

    # Only the parameter's own unconstrained value is trainable.
    assert len(jax.tree_util.tree_leaves(params)) == 1
    assert eqx.combine(params, static).prior is p.prior


def test_collect_log_prior_sums_priors_and_skips_unpriored_parameters():
    from gpjax.parameters import PositiveReal, Real, collect_log_prior
    import numpyro.distributions as dist

    ls_prior = dist.LogNormal(jnp.log(2.0), 0.5)
    mean_prior = dist.Normal(0.0, 1.0)
    tree = {
        "a": PositiveReal(jnp.array(0.7), prior=ls_prior),
        "b": Real(jnp.array(0.4), prior=mean_prior),
        "c": PositiveReal(jnp.array(1.3)),  # no prior
    }

    expected = (
        ls_prior.log_prob(jnp.array(0.7)).sum()
        + mean_prior.log_prob(jnp.array(0.4)).sum()
    )
    assert jnp.allclose(collect_log_prior(tree), expected)


def test_collect_log_prior_is_zero_without_priors():
    from gpjax.parameters import PositiveReal, collect_log_prior

    assert jnp.allclose(collect_log_prior({"a": PositiveReal(jnp.array(1.0))}), 0.0)


def test_collect_log_prior_handles_vector_valued_parameters():
    from gpjax.parameters import PositiveReal, collect_log_prior
    import numpyro.distributions as dist

    prior = dist.LogNormal(jnp.log(2.0), 0.5)
    ls = jnp.array([0.5, 1.0, 2.0])
    assert jnp.allclose(
        collect_log_prior(PositiveReal(ls, prior=prior)), prior.log_prob(ls).sum()
    )
