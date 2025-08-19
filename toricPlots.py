import json
import pickle
from collections import Counter, OrderedDict
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd
import ray
from netket.experimental.models.fast_rnn import FastLSTMNet
from netket.operator.spin import sigmax, sigmaz
from optax import adamw
from scipy.special import logsumexp
from tqdm import tqdm

from models.lstm import FastLSTMNet as modLSTMNet
from models.vanilla import VanillaRNN
from regularizedQSR import QSR as rQSR


def H(dataset):
    configs = [tuple(row) for row in dataset]
    counts = Counter(configs)
    total = len(dataset)
    probs = np.array([c / total for c in counts.values()])
    return -np.sum(probs * np.log(probs)) / np.log(2)


shape = lambda tree: jax.tree.map(lambda leaf: leaf.shape, tree)

plt.style.use("~/plotStyle.mplstyle")


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


def compute_syndromes(single_qubit, log_probability):
    """
    Given a 2D numpy array `single_qubit` of qubit values (e.g. ±1),
    this function computes and returns the list of syndromes for both
    bulk and boundary regions of the array.
    """
    L_target = single_qubit.shape[0]
    syndrome_list = []

    # Compute the bulk syndromes
    for i in range(0, L_target - 1):
        for j in range(0, L_target - 1):
            if (i + j) % 2 == 0:
                s_1 = single_qubit[i, j]
                s_2 = single_qubit[i + 1, j]
                s_3 = single_qubit[i, j + 1]
                s_4 = single_qubit[i + 1, j + 1]
                syndrome = s_1 * s_2 * s_3 * s_4
                syndrome_list.append(syndrome)

    # Left boundary syndromes
    j = 0
    for i in range(1, L_target - 1, 2):
        s_1 = single_qubit[i, j]
        s_2 = single_qubit[i + 1, j]
        syndrome = s_1 * s_2
        syndrome_list.append(syndrome)

    # Right boundary syndromes
    j = L_target - 1
    for i in range(0, L_target - 1, 2):
        s_1 = single_qubit[i, j]
        s_2 = single_qubit[i + 1, j]
        syndrome = s_1 * s_2
        syndrome_list.append(syndrome)

    new_log_probability = log_probability + (L_target * L_target + 1) / 2
    # print(len(syndrome_list))
    return syndrome_list, new_log_probability


def KL(sigmas, statePath, copy):
    g = nk.graph.Hypercube(length=len(sigmas[0]), n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    if ansatz == "RBM":
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)
        model = nk.models.RBM(alpha=alpha, param_dtype=float)

    elif "anilla" in ansatz:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = VanillaRNN(layers=3, features=dh, hilbert=hilbert, graph=g)

    elif "RNN" in ansatz:
        if "2D" in ansatz:
            g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
    elif "modLSTM" in ansatz:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = modLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
    elif ansatz == "Product":
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=2, features=5, hilbert=hilbert, graph=g)

    state = nk.vqs.MCState(sampler, model=model, n_samples=256)
    path = statePath + f"_copy={copy}_L={L}"
    vars = nk.experimental.vqs.variables_from_file(
        filename=path, variables=state.variables
    )
    state.variables = vars
    addition = 0
    if ansatz == "RBM":
        samples = state.sample(n_discard_per_chain=500)
        logPsRbm = 2 * state.log_value(samples)
        addition = logsumexp(logPsRbm) - np.log(len(samples))
    logPTheta = 2 * state.log_value(sigmas)
    assert logPs.shape == logPTheta.shape
    kl = (1 / len(logPs)) * np.sum(logPs - logPTheta) + addition
    print(logPTheta)
    return kl


def JS(sigmas, statePath, copy, p):
    g = nk.graph.Hypercube(length=len(sigmas[0]), n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    if ansatz == "RBM":
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)
        model = nk.models.RBM(alpha=alpha, param_dtype=float)
    elif "anilla" in ansatz:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = VanillaRNN(layers=3, features=dh, hilbert=hilbert, graph=g)

    elif "RNN" in ansatz:
        if "2D" in ansatz:
            g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)

    elif "modLSTM" in ansatz:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = modLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)

    elif ansatz == "Product":
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=2, features=5, hilbert=hilbert, graph=g)

    state = nk.vqs.MCState(sampler, model=model, n_samples=10000)
    path = statePath + f"_copy={copy}_L={L}"
    vars = nk.experimental.vqs.variables_from_file(
        filename=path, variables=state.variables
    )
    state.variables = vars

    pSamples = sigmas
    qSamples = state.sample()[0]
    print(qSamples.shape)

    P = lambda x: prob(x, L, p)
    Q = lambda x: jnp.exp(2 * state.log_value(x))
    M = lambda x: 0.5 * (P(x) + Q(x))
    js = 0.5 * jnp.mean(np.log(P(pSamples)) - np.log(M(pSamples))) + 0.5 * (
        np.log(Q(qSamples)) - np.log(M(qSamples))
    )
    return js


klRemote = ray.remote(KL)


def trainingMetric(measurements, statePath, copy):
    Ns = 100000
    indices = np.random.randint(len(measurements), size=(Ns,))
    samples = measurements[indices]

    g = nk.graph.Hypercube(length=len(samples[0]), n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    if ansatz == "RBM":
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)
        model = nk.models.RBM(alpha=alpha, param_dtype=float)

    elif "anilla" in ansatz:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = VanillaRNN(layers=3, features=dh, hilbert=hilbert, graph=g)

    elif "RNN" in ansatz:
        if "2D" in ansatz:
            g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
    elif "modLSTM" in ansatz:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = modLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
    elif ansatz == "Product":
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=2, features=5, hilbert=hilbert, graph=g)

    state = nk.vqs.MCState(sampler, model=model, n_samples=Ns)
    path = statePath + f"_copy={copy}_L={L}"
    vars = nk.experimental.vqs.variables_from_file(
        filename=path, variables=state.variables
    )
    state.variables = vars

    ############### Gradient Stuff
    # grads = nk.jax.jacobian(state.model.apply, state.parameters, samples, mode="real")
    # batchedGrads = jax.tree.map(
    #     lambda leaf: jnp.mean(leaf.reshape(-1, 100, *leaf.shape[1:]), axis=1), grads
    # )
    # meanGrads = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), grads)
    #
    # var = jax.tree.map(
    #     lambda leaf1, leaf2: jnp.average(jnp.abs(leaf1 - leaf2) ** 2, axis=0),
    #     grads,
    #     meanGrads,
    # )
    # VarFlat, _ = jax.flatten_util.ravel_pytree(var)
    # return jnp.mean(np.abs(VarFlat))
    ###############

    ###############SNR
    # VarFlat, _ = jax.flatten_util.ravel_pytree(var)
    # meanFlat, _ = jax.flatten_util.ravel_pytree(meanGrads)
    # out = meanFlat**2 / VarFlat
    # # plt.hist(meanFlat**2, bins=50, histtype="step")
    # # plt.xlim(0, 3)
    # # plt.ylim(1, 1e3)
    # # plt.yscale("log")
    # # plt.show()
    # return jnp.median(out)
    ###############

    # #############FIM
    # Oks = jax.tree.map(lambda leaf: leaf.reshape(leaf.shape[0], -1), grads)
    # OksFlat = jnp.hstack(jax.tree.leaves(Oks))
    # Ss = jnp.einsum("ij,ik->jk", OksFlat, OksFlat) / Ns
    # out = np.linalg.eigvalsh(Ss)
    # # out = jnp.nansum(jnp.log(out[out > 0]))
    # return jnp.sum(out)
    # #############

    ############## Stiffness
    # res = []
    # for reference in range(10):
    #     path = statePath + f"_copy={reference}_L={L}"
    #     for i in range(50):
    #         vars = nk.experimental.vqs.variables_from_file(
    #             filename=path, variables=state.variables
    #         )
    #         state.variables = vars
    #         # samples = state.sample()
    #         samples = np.random.choice([-1, 1], size=(Ns, hilbert.size))
    #         if samples.ndim == 3:
    #             samples = samples[0]
    #         logPs1 = state.log_value(samples)
    #
    #         state.variables = jax.tree.map(
    #             lambda x: x + np.random.normal(0, 0.005), vars
    #         )
    #         logPs2 = state.log_value(samples)
    #         res.append(jnp.mean((logPs2 - logPs1) ** 2))
    #     return jnp.mean(jnp.array(res))
    ##############

    ############## KL after moves
    # res = []
    # for i in range(100):
    #     state.variables = vars
    #     samples = state.sample()
    #     if samples.ndim == 3:
    #         samples = samples[0]
    #     logPs1 = state.log_value(samples)
    #
    #     indices = np.random.randint(len(measurements), size=(128,))
    #     inputs = measurements[indices]
    #     grads = nk.jax.jacobian(
    #         state.model.apply, state.parameters, samples, mode="real"
    #     )
    #     grads = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), grads)
    #     grads = {"params": grads}
    #     state.variables = jax.tree.map(lambda x, y: x + 0.001 * y, vars, grads)
    #
    #     logPs2 = state.log_value(samples)
    #     res.append(jnp.mean((logPs1 - logPs2)))
    # return jnp.mean(jnp.array(res))
    ##############

    ############## Parity

    samples = state.sample()
    out = jnp.prod(samples, axis=-1)
    return np.mean(out)
    ##############

    ############## KL between copies
    # Mat = np.zeros((10, 10))
    # for reference in range(10):
    #     path = statePath + f"_copy={reference}"
    #     vars = nk.experimental.vqs.variables_from_file(
    #         filename=path, variables=state.variables
    #     )
    #     state.variables = vars
    #     # samples = state.sample()[0]
    #     logPsReference = 2 * state.log_value(samples)
    #
    #     for copy in range(reference + 1, 10):
    #         # if reference == copy:
    #         #     continue
    #         path = statePath + f"_copy={copy}"
    #         vars = nk.experimental.vqs.variables_from_file(
    #             filename=path, variables=state.variables
    #         )
    #         state.variables = vars
    #         logPs = 2 * state.log_value(samples)
    #         Mat[reference][copy] = np.mean(np.abs(logPsReference - logPs))
    # return np.min(Mat[Mat != 0])
    ##############

    ############## JS between copies
    # Mat = np.zeros((10, 10))
    # for reference in range(10):
    #     path = statePath + f"_copy={reference}"
    #     vars = nk.experimental.vqs.variables_from_file(
    #         filename=path, variables=state.variables
    #     )
    #     state.variables = vars
    #     samples = state.sample()[0]
    #     logPsReference = 2 * state.log_value(samples)
    #
    #     for copy in range(reference + 1, 10):
    #         path = statePath + f"_copy={copy}"
    #         vars = nk.experimental.vqs.variables_from_file(
    #             filename=path, variables=state.variables
    #         )
    #         state.variables = vars
    #         logPs = 2 * state.log_value(samples)
    #         logM = logsumexp([logPs, logPsReference], axis=0) - np.log(2)
    #         Mat[reference][copy] = np.mean(logPsReference - logM)
    # return np.mean(Mat[Mat != 0])
    ############## Dist from uniform
    # path = statePath
    # vars = nk.experimental.vqs.variables_from_file(
    #     filename=path, variables=state.variables
    # )
    # state.variables = vars
    # samples = state.sample()[0]
    # logPs = 2 * state.log_value(samples)
    # return jnp.mean(logPs) + hilbert.size * np.log(2)
    ##############

    ##############  'Useful' gradient
    # state = nk.vqs.MCState(sampler, model=model, n_samples=Ns)
    # uniformSamples = np.random.choice([-1, 1], size=(Ns, hilbert.size))
    #
    # # refSamples = deepcopy(samples)
    # # refSamples[:, -1] = uniformSamples[:, -1]
    # refSamples = uniformSamples
    #
    # trueGrads = nk.jax.jacobian(
    #     state.model.apply, state.parameters, samples, mode="real"
    # )
    # junkGrads = nk.jax.jacobian(
    #     state.model.apply, state.parameters, refSamples, mode="real"
    # )
    # absDiff = jax.tree.map(lambda x, y: jnp.abs(x - y), trueGrads, junkGrads)
    #
    # absDiffFlat, _ = jax.flatten_util.ravel_pytree(absDiff)
    # return jnp.mean(absDiffFlat)
    ##############


if __name__ == "__main__":
    colors = ["red", "blue", "green", "orange", "yellow"]
    markers = ["o", "s", "^", "d", "*"]
    HType = "XXpZ"
    d = 1
    dh = 10
    it = 0
    for L in [32]:
        for ansatz in [
            # "RNN"
            "modLSTM",
            # "modLSTM_double",
            # "modLSTM_triple",
            # "modLSTM_oct",
        ]:
            pDict = {
                "Toric": np.arange(0.00, 0.2, 0.01),
                "ToricPBC": np.arange(0.00, 0.2, 0.01),
                "IsingX": np.arange(0.05, 2, 0.05),
                "XXpZ": np.arange(0.05, 2, 0.05),
                "IsingZ": np.arange(0.05, 2, 0.05),
                "GHZ": np.arange(0.0, 1.0, 0.05),
                "ToricCoherent": np.arange(0, 26),
                "ToricCoherentSQ": np.arange(0, 26),
            }
            ps = pDict[HType]
            for p in tqdm(ps):
                pname = f"{int(p)}" if "ToricCoherent" in HType else f"{p:.3f}"
                # letter = "h" if "Ising" in HType else "p"
                letter = "h"
                delim = "\t"
                # delim = "\t" if "Ising" in HType else " "
                if "ToricCoherent" not in HType:
                    measurements = pd.read_csv(
                        f"traindata/{HType.lower()}/d={d}/measurements_L={L}_{letter}={pname}.csv",
                        delimiter=delim,
                        header=None,
                    )
                    measurements = measurements.dropna(axis=1, how="all")
                    measurements = measurements.to_numpy()
                    print((np.prod(measurements, axis=-1)))
                    if HType == "GHZ":
                        logPs = np.log(prob(measurements, L, p) + 1e-20)
                    else:
                        ps = pd.read_csv(
                            f"traindata/{HType.lower()}/d={d}/ps_L={L}_{letter}={pname}.csv",
                            delimiter=delim,
                            header=None,
                        )
                        # ps = ps.dropna(axis=1, how="all")
                        ps = ps.to_numpy()
                        logPs = np.log(ps)[:, 0]
                        print(logPs.shape)

                else:
                    tempFunc = lambda x, y: (
                        compute_syndromes(x, y)
                        if HType == "ToricCoherent"
                        else (x.flatten(), y)
                    )
                    filename = f"traindata/toriccoherent/L{L}.pkl"
                    with open(filename, "rb") as f:
                        etaDict = pickle.load(f)
                    logPs = []
                    measurements = []
                    for i in range(len(etaDict[p])):
                        measurement, logP = tempFunc(
                            etaDict[p][i]["sample"], etaDict[p][i]["logp"]
                        )
                        measurements.append(measurement)
                        logPs.append(logP)
                    measurements = np.array(measurements)
                    logPs = np.log(2) * np.array(logPs)[:, 0, 0]

                cont = []
                func = KL
                for copy in range(5, 10):
                    # dh = 6 + 2 * copy
                    metric = func(
                        measurements,
                        f"logs/{HType.lower()}/d={d}/{ansatz}_param={pname}",
                        copy,
                        # p,
                    )
                    cont.append(metric)
                    # plt.scatter(
                    #     p,
                    #     kl,
                    #     s=40,
                    #     color=colors[copy],
                    #     label=rf"$d_h = {dh}$",
                    #     edgecolor="black",
                    # )
                    #
                metric, err = np.min(cont), np.sqrt(np.var(cont))
                eta = (np.pi * p) / 100
                x = p if "ToricCoherent" not in HType else eta
                plt.scatter(
                    x,
                    metric,
                    marker=markers[it],
                    # yerr=err,
                    color=colors[it],
                    edgecolor="black",
                    label=f"L={L}",
                    # capsize=5,
                )
            it += 1
    plt.ylabel(r"$D_{KL}(p_{true}| p_{rnn})$")
    # plt.ylabel(r"$\left| \nabla L_{true} - \nabla L_{ref}\right|$")
    # plt.ylabel(r"$1-\expval{\prod_i^N \sigma_i}$")
    # plt.xlabel(r"$p_{err}$")
    plt.xlabel(r"$h$")
    # plt.xlabel(r"$\eta$")

    plt.title(HType)
    # plt.yscale("log")
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize="small")
    plt.show()
