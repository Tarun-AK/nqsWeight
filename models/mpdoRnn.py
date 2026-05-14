from collections.abc import Iterable
from functools import partial
from typing import Optional, Union

import flax
import jax
import netket as nk
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
    bias_init: NNInitFunc = zeros

    @nn.compact
    def __call__(self, inputs, cell_mem, hidden):
        MT = self.param("MT", zeros, (2, 4, 4), jnp.float64)
        MTs = jnp.einsum(inputs, MT, "ij,jkl->ikl")
        output = jnp.einsum(cell_mem, MTs, "ik,ikl->il")
        return output, output


class MPDO(AbstractARNN):
    machine_pow = 1
    p: float = 0.5
    epsilon: float = 0.0
    extra_hidden: int = 0
    hilbert = nk.hilbert.Spin(s=1 / 2, N=64)

    @nn.compact
    def conditionals_log_psi(self, inputs):
        print(self.hilbert.size)
        core_dim = 4
        extra = int(self.extra_hidden)
        hidden_dim = core_dim + extra
        p = float(self.p)
        eps = float(self.epsilon)

        def gammas_init(rng, shape, dtype=jnp.float64):
            gammaisN = jnp.array([1 - p, 1 - p, p, p], dtype=dtype)
            gammaN = jnp.array([1, 0, 0, 1], dtype=dtype)
            gammas = jnp.tile(gammaisN, (self.hilbert.size, 1))
            gammas = gammas.at[-1].set(gammaN)
            if extra > 0:
                gammas = jnp.concatenate(
                    [gammas, jnp.zeros((self.hilbert.size, extra), dtype=dtype)], axis=1
                )
            return gammas

        def MT_init(rng, shape, dtype=jnp.float64):
            M00 = jnp.array([[1 - p, 0], [0, p]], dtype=dtype)
            M11 = jnp.array([[0, p], [1 - p, 0]], dtype=dtype)
            zero = jnp.zeros((2, 2), dtype=dtype)

            if eps != 0.0:
                M00 = (1 - 2 * eps) * M00 + 2 * eps * M11
                M11 = (1 - 2 * eps) * M11 + 2 * eps * M00

            MT00 = jnp.block([[M00, zero], [zero, M00]])
            MT11 = jnp.block([[M11, zero], [zero, M11]])

            def pad4(mat4):
                out = jnp.zeros((hidden_dim, hidden_dim), dtype=dtype)
                out = out.at[:4, :4].set(mat4)
                return out

            return jnp.stack([pad4(MT00), pad4(MT11)], axis=0)

        def vL_init(rng, shape, dtype=jnp.float64):
            vL = jnp.array([1, 0, 0, 1], dtype=dtype)
            if extra > 0:
                vL = jnp.concatenate([vL, jnp.zeros((extra,), dtype=dtype)], axis=0)
            return vL

        gammas = self.param(
            "gammas", gammas_init, (self.hilbert.size, hidden_dim), jnp.float64
        )
        MT = self.param("MT", MT_init, (2, hidden_dim, hidden_dim), jnp.float64)
        vL = self.param("vL", vL_init, (hidden_dim,), jnp.float64)

        h = jnp.tile(jnp.array(vL), (inputs.shape[0], 1))
        hs = jnp.stack([h, h], axis=1)

        output = []
        for index in range(inputs.shape[1]):
            input = jax.nn.one_hot((-inputs[:, index] + 1) / 2, num_classes=2)
            hs = jnp.einsum("il,klm->ikm", h, MT)
            h = jnp.einsum("ikm,ik->im", hs, input)
            h = h * 0.5 * (jnp.sign(h + 5) - jnp.sign(h - 5))
            ps = jnp.abs(jnp.einsum("ikm,m->ik", hs, gammas[index]))
            ps = ps / (jnp.sum(ps, axis=-1, keepdims=True) + 1e-54)
            ps = jnp.abs(ps)
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
    for L in [8]:
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)

        for perr in [0.1, 0.02, 0.5]:
            model = MPDO(hilbert=hilbert, p=perr, extra_hidden=10)
            state = nk.vqs.MCState(sampler, model=model, n_samples=1000)
            x = hilbert.all_states()
            print(np.sum((np.exp(state.log_value(x)) - prob(x, L, perr)) ** 2))
        # epsilon = 0.1
        # for sigma in [0.1]:
        #     for p in tqdm(jnp.arange(0.00001, 1.05, 0.05)):
        #         gammaisN = jnp.array([1 - p, 1 - p, p, p])
        #         gammaN = jnp.array([1, 0, 0, 1])
        #         pars = flax.core.copy(state.parameters, {})
        #
        #         pars["gammas"] = jnp.tile(gammaisN, (hilbert.size, 1))
        #         pars["gammas"] = pars["gammas"].at[-1].set(gammaN)
        #
        #         M00 = jnp.array([[1 - p, 0], [0, p]])
        #         M11 = jnp.array([[0, p], [1 - p, 0]])
        #         zero = jnp.zeros((2, 2))
        #
        #         MT00 = jnp.block([[M00, zero], [zero, M00]])
        #         MT11 = jnp.block([[M11, zero], [zero, M11]])
        #         pars["MT"] = jnp.array([MT00, MT11])
        #         pars["vL"] = jnp.array([1, 0, 0, 1])
        #         state.parameters = pars
        #
        #         M00e = ((1 - 2 * epsilon) * M00) + (2 * epsilon * M11)
        #         M11e = (1 - 2 * epsilon) * M11 + 2 * epsilon * M00
        #         MT00e = jnp.block([[M00e, zero], [zero, M00e]])
        #         MT11e = jnp.block([[M11e, zero], [zero, M11e]])
        #         print(MT00.shape, MT00e.shape)
        #         noisePars = copy.deepcopy(pars)
        #         noisePars["MT"] = jnp.array([MT00e, MT11e])
        #         noisePars["vL"] = jnp.array([1, 0, 0, 1])
        #         noisyState = copy.deepcopy(state)
        #         noisyState.parameters = noisePars
        #
        #         Nop = lambda x: np.exp(noisyState.log_value(x))
        #         x = np.array(hilbert.all_states())
        #         assert x.shape[-1] == hilbert.size
        #         print(hilbert.size)
        #         xLastFlipped = copy.deepcopy(x)
        #         xLastFlipped[:, -1] *= -1
        #         prefx = 0.5 * (Nop(x) + Nop(xLastFlipped))
        #         chiSqD = np.sum((Nop(x) - prefx) ** 2 / prefx)
        #         plt.scatter(
        #             p,
        #             chiSqD,
        #             edgecolor="black",
        #             color=colors[it],
        #             marker=markers[it],
        #             label=rf"$L={L}$",
        #         )
        #     it += 1
    plt.yscale("log")
    plt.xlabel("L")
    plt.ylabel(
        r"$\chi^2\left(\mathcal{N}_{\epsilon}(p_{\mathrm{true}}) || \mathcal{N}_{\epsilon}(p_{\mathrm{ref}}) \right)$"
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
