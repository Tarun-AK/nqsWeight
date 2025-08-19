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
from tqdm import tqdm


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
    machine_pow = 1

    @nn.compact
    def conditionals_log_psi(self, inputs):
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

        output = []
        for index in range(inputs.shape[1]):
            # jax.debug.print("gammas {}", gammas[index])
            # jax.debug.print("ps {} spin {} onehot {}", ps, inputs[:, index], input)
            input = jax.nn.one_hot((-inputs[:, index] + 1) / 2, num_classes=2)
            hs = jnp.einsum("il,klm->ikm", h, MT)
            h = jnp.einsum("ikm,ik->im", hs, input)
            ps = jnp.abs(jnp.einsum("ikm,m->ik", hs, gammas[index]))
            ps = ps / (jnp.sum(ps, axis=-1, keepdims=True) + 1e-54)
            # jax.debug.print("ps {}", ps)
            # p = jnp.einsum("ik,ik->i", ps, input)
            output.append(ps)

        output = jnp.transpose(jnp.array(output), (1, 0, 2))

        return jnp.log(output + 1e-90)


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
    import copy
    from collections import OrderedDict

    import matplotlib.pyplot as plt
    import netket as nk
    import numpy as np

    colors = ["red", "blue", "green", "orange", "yellow"]
    markers = ["o", "s", "^", "d", "*"]
    plt.style.use("~/plotStyle.mplstyle")

    it = 0
    for L in [6, 8, 10, 12, 14]:
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = MPDO(hilbert=hilbert)
        state = nk.vqs.MCState(sampler, model=model, n_samples=1000)

        x = hilbert.all_states()

        epsilon = 0.1
        for sigma in [0.1]:
            for p in tqdm(jnp.arange(0.00001, 1.05, 0.05)):
                # for p in [0.5]:
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

                ##Noise
                M00e = ((1 - 2 * epsilon) * M00) + (2 * epsilon * M11)
                M11e = (1 - 2 * epsilon) * M11 + 2 * epsilon * M00
                MT00e = jnp.block([[M00e, zero], [zero, M00e]])
                MT11e = jnp.block([[M11e, zero], [zero, M11e]])
                print(MT00.shape, MT00e.shape)
                noisePars = copy.deepcopy(pars)
                noisePars["MT"] = jnp.array([MT00e, MT11e])
                noisePars["vL"] = jnp.array([1, 0, 0, 1])
                noisyState = copy.deepcopy(state)
                noisyState.parameters = noisePars
                ##Noise

                ############################################################################### Stiffness

                # referenceVars = copy.deepcopy(state.variables)
                # res = []
                # for i in range(1000):
                #     state.variables = referenceVars
                #     samples = state.sample()
                #     # samples = np.random.choice([-1, 1], size=(10000, hilbert.size))
                #     if samples.ndim == 3:
                #         samples = samples[0]
                #     logPs1 = state.log_value(samples)
                #
                #     state.variables = jax.tree.map(
                #         lambda x: x + np.random.normal(0, sigma, size=x.shape),
                #         referenceVars,
                #     )
                #
                #     # MT = np.array(state.variables["params"]["MT"])
                #     # print(MT)
                #     # MT[MT == 0] = np.random.normal(0, 0.01)
                #     # print(MT)
                #     # state.variables["params"]["MT"] = MT
                #
                #     logPs2 = state.log_value(samples)
                #     res.append(np.nanmean((logPs1 - logPs2)))
                #     # res.append(np.nanmean((np.exp(logPs1) - np.exp(logPs2)) ** 2))
                #
                # plt.scatter(
                #     p,
                #     np.nanmean(res) / sigma,
                #     edgecolor="black",
                #     color=colors[it],
                #     marker=markers[it],
                #     label=rf"$\sigma^2 = {sigma}$",
                # )
                ############################################################################### chi-sq stab
                Nop = lambda x: np.exp(noisyState.log_value(x))
                x = np.array(hilbert.all_states())
                assert x.shape[-1] == hilbert.size
                print(hilbert.size)
                xLastFlipped = copy.deepcopy(x)
                xLastFlipped[:, -1] *= -1
                prefx = 0.5 * (Nop(x) + Nop(xLastFlipped))
                chiSqD = np.sum((Nop(x) - prefx) ** 2 / prefx)
                plt.scatter(
                    # L,
                    p,
                    chiSqD,
                    edgecolor="black",
                    color=colors[it],
                    marker=markers[it],
                    label=rf"$L={L}$",
                )
                ###############################################################################
            it += 1
    plt.yscale("log")
    # plt.xlabel(r"$p_{err}$")
    plt.xlabel("L")
    plt.title(rf"$\epsilon={epsilon}$")
    # plt.ylabel(
    #     r"$\mathbb{E}_{\rho \sim \mathcal{N}} \left[\mathrm{MSE}(p_{\theta}, p_{\theta + \rho})\right]$"
    # )
    # plt.ylabel(
    #     r"$ (1/\sigma_{\mathcal{N}}^2) \cdot\mathbb{E}_{\rho \sim \mathcal{N}} \left[\mathrm{D_{KL}}(p_{\theta} | p_{\theta + \rho})\right] $"
    # )
    plt.ylabel(
        r"$\chi^2\left(\mathcal{N}_{\epsilon}(p_{\mathrm{true}}) || \mathcal{N}_{\epsilon}(p_{\mathrm{ref}}) \right)$"
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
