from .likelihoods import Gaussian, Likelihood
from .kernel import Kernel
from .mean_functions import MeanFunction
from .mean_functions import ZeroMean
from .utils import get_factorisations
from objax import TrainVar, Module
import jax.numpy as jnp
from jax import nn
from jax.scipy.linalg import cho_solve, solve_triangular
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


class Prior(Module):
    def __init__(self,
                 kernel: Kernel,
                 mean_function: MeanFunction = ZeroMean(),
                 jitter: float = 1e-6):
        self.meanf = mean_function
        self.kernel = kernel
        self.jitter = jitter

    def sample(self, X: jnp.ndarray, key, n_samples: int = 1):
        mu = self.meanf(X)
        cov = self.kernel(X, X)
        if cov.shape[0] == cov.shape[1]:
            Inn = jnp.eye(cov.shape[0])*self.jitter
            cov += Inn
        return jr.multivariate_normal(key,
                                      mu.squeeze(),
                                      cov,
                                      shape=(n_samples, ))

    def __mul__(self, other: Likelihood):
        if self.kernel.spectral is True:
            return SpectralPosterior(self, other)
        else:
            return Posterior(self, other)


class Posterior(Module):
    def __init__(self, prior: Prior, likelihood: Gaussian):
        self.kernel = prior.kernel
        self.meanf = prior.meanf
        self.likelihood = likelihood
        self.jitter = prior.jitter

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        mu = self.meanf(X)
        cov = self.kernel(X, X)
        if cov.shape[0] == cov.shape[1]:
            Inn = jnp.eye(cov.shape[0])*self.jitter
            cov += Inn
            cov += self.likelihood.noise.constrained_value * Inn
        L = jnp.linalg.cholesky(cov)
        # TODO: Return the logpdf w.r.t. the Cholesky, not the full cov.
        # lpdf = multivariate_normal.logpdf(y.squeeze(), mu.squeeze(), cov)
        # return lpdf
        return tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray):
        rv = self.marginal_ll(X, y)
        return -rv.log_prob(y.squeeze()).mean()

    def predict(self, Xstar, X, y):
        sigma = self.likelihood.noise.constrained_value
        L, alpha = get_factorisations(X, y, sigma, self.kernel, self.meanf)
        Kfx = self.kernel(Xstar, X)
        mu = jnp.dot(Kfx, alpha)
        v = cho_solve(L, Kfx.T)
        Kxx = self.kernel(Xstar, Xstar)
        cov = Kxx - jnp.dot(Kfx, v)
        return mu, cov


class SpectralPosterior(Posterior):
    def __init__(self, prior: Prior, likelihood: Gaussian):
        super().__init__(prior, likelihood)

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        N = X.shape[0]
        m = self.kernel.num_basis
        l_var = self.likelihood.noise.value
        k_var = self.kernel.variance.value
        phi = self.kernel(X, self.kernel.features.value.T)
        A = (k_var / m) * jnp.matmul(phi.T, phi) + l_var * jnp.eye(m * 2)
        Rt = jnp.linalg.cholesky(A)
        RtiPhit = solve_triangular(Rt, phi.T)
        # assert RtiPhit.shape == (N, N)
        RtiPhity = jnp.matmul(RtiPhit, y)
        # assert RtiPhity.shape == y.shape
        term1 = (jnp.sum(y**2) -
                 jnp.sum(RtiPhity**2) * k_var / m) * 0.5 / l_var
        term2 = jnp.sum(jnp.log(jnp.diag(
            Rt.T))) + (N * 0.5 - m) * jnp.log(l_var) + (N * 0.5 *
                                                        jnp.log(2 * jnp.pi))
        tot = term1 + term2
        return tot.reshape()