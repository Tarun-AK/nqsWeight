from collections.abc import Iterable
from functools import partial
from typing import Optional, Union

import flax
import jax
from flax import linen as nn
from flax.linen.initializers import zeros
from jax import numpy as jnp
from jax.nn.initializers import normal, zeros
from netket.experimental.models.rnn import RNN
from netket.experimental.nn.rnn import (
    FastRNNLayer,
    LSTMCell,
    RNNCell,
    default_kernel_init,
    ensure_prev_neighbors,
)
from netket.graph import AbstractGraph
from netket.models.autoreg import AbstractARNN
from netket.nn.masked_linear import default_kernel_init
from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc


class MPDOCell(RNNCell):
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias."""

    @nn.compact
    def __call__(self, inputs, cell_mem, hidden):
        MT = self.param("MT", zeros, (2, 4, 4), jnp.float64)
        MTs = jnp.einsum(inputs, MT, "ij,jkl->ikl")
        output = jnp.einsum(cell_mem, MTs, "ik,ikl->il")
        return output, output


class MPDO(AbstractARNN):
    @nn.compact
    def __call__(self, inputs):
        gammas = self.param(
            "gammas",
            normal(0.1),
            (self.hilbert.size, 4),
            jnp.float64,
        )
        MT = self.param("MT", normal(0.1), (2, 4, 4), jnp.float64)

        vL = self.param("vL", normal(0.1), (4,), jnp.float64)
        h = jnp.tile(jnp.array(vL), (inputs.shape[0], 1))
        hs = jnp.stack([h, h], axis=1)

        output = jnp.ones((inputs.shape[0],), dtype=jnp.float64)

        for index in range(inputs.shape[1]):
            input = jax.nn.one_hot((inputs[:, index] - 1) / 2, num_classes=2)
            ps = jnp.einsum("ikm,m->ik", hs, gammas[index])
            ps = ps / (jnp.sum(ps, axis=-1, keepdims=True) + 1e-54)
            p = jnp.einsum("ik,ik->i", ps, input)
            # jax.debug.print("{}", ps)
            output *= p
            hs = jnp.einsum("il,klm->ikm", h, MT)
            h = jnp.einsum("ikm,ik->im", hs, input)

        return output


@partial(jax.vmap, in_axes=(0, None, None))
def prob(sigmas, L, p):
    size = L + (L % 2)
    minuses = jnp.array(jnp.argwhere(sigmas == -1, size=size, fill_value=jnp.nan))[:, 0]
    result = minuses[1::2] - minuses[::2]
    l = jnp.nansum(result)
    term1 = (p**l) * ((1 - p) ** (L - l))
    term2 = (p ** (L - l)) * ((1 - p) ** l)

    return jax.lax.cond(
        jnp.sum(sigmas == -1) % 2 == 0,
        lambda _: term1 + term2,
        lambda _: 0.0,
        operand=None,
    )


if __name__ == "__main__":
    import netket as nk

    g = nk.graph.Hypercube(length=4, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
    model = MPDO(hilbert=hilbert)
    state = nk.vqs.MCState(sampler, model=model, n_samples=100)

    x = hilbert.all_states()
    # x = jnp.ones((1, hilbert.size))

    p = 0.1

    gammaisN = jnp.array([1 - p, 1 - p, p, p])
    gammaN = jnp.array([1, 0, 0, 1])
    pars = flax.core.copy(state.parameters, {})

    pars["gammas"] = jnp.tile(gammaisN, (hilbert.size, 1))
    pars["gammas"] = pars["gammas"].at[-1].set(gammaN)

    M00 = jnp.array([[1 - p, 0], [0, p]])
    M11 = jnp.array([[0, p], [1 - p, 0]])
    zero = jnp.zeros((2, 2))

    MT00 = jnp.block([[M00, zero], [zero, M00]])
    MT11 = jnp.block([[M11, zero], [zero, M11]])

    pars["MT"] = jnp.array([MT00, MT11])
    pars["vL"] = jnp.array([1, 0, 0, 1])

    state.parameters = pars
    print(50 * "#")
    print(state.log_value(x))
    print(prob(x, hilbert.size, p))
