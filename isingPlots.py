import json
from collections import OrderedDict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd
from netket.experimental.models.fast_rnn import FastLSTMNet
from netket.operator.spin import sigmax, sigmaz

sx = lambda a, b: (1 / 2) * sigmax(a, b)
sz = lambda a, b: (1 / 2) * sigmaz(a, b)

plt.style.use("~/plotStyle.mplstyle")

nSpins = 32


def gradVariance(measurements, statePath):
    Ns = 10000
    indices = np.random.randint(len(measurements), size=(Ns,))
    samples = measurements[indices]
    # samples = measurements

    g = nk.graph.Hypercube(length=nSpins, n_dim=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=nSpins)
    if ansatz == "RNN":
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=3, features=10, hilbert=hilbert, graph=g)
    elif ansatz == "RBM":
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)
        model = nk.models.RBM(alpha=1.5, param_dtype=float)

    state = nk.vqs.MCState(sampler, model=model, n_samples=256)

    vars = nk.experimental.vqs.variables_from_file(
        filename=statePath, variables=state.variables
    )
    state.variables = vars
    grads = nk.jax.jacobian(state.model.apply, state.parameters, samples, mode="real")

    Oks = jax.tree.map(lambda leaf: leaf.reshape(leaf.shape[0], -1), grads)
    OksFlat = jnp.hstack(jax.tree.leaves(Oks))
    Ss = jnp.einsum("ij,ik->jk", OksFlat, OksFlat) / Ns
    out = np.linalg.eigvalsh(Ss)
    out = jnp.sum(jnp.log(out[out > 0]))
    return out


def KL(sigmas, statePath, numSamples):
    g = nk.graph.Hypercube(length=sigmas.shape[-1], n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=sigmas.shape[-1])
    sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
    model = FastLSTMNet(layers=3, features=10, hilbert=hilbert, graph=g)
    state = nk.vqs.MCState(sampler, model=model, n_samples=256)

    vars = nk.experimental.vqs.variables_from_file(
        filename=statePath, variables=state.variables
    )
    state.variables = vars

    logPTheta = 2 * state.log_value(sigmas)
    return (1 / numSamples) * np.sum(logPs[:, 0] - logPTheta)


def energy(HType, statePath, numSamples):
    g = nk.graph.Hypercube(length=sigmas.shape[-1], n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=sigmas.shape[-1])

    if "RNN" in statePath:
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        model = FastLSTMNet(layers=3, features=10, hilbert=hilbert, graph=g)
    else:
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)
        model = nk.models.RBM(alpha=1.5, param_dtype=float)

    state = nk.vqs.MCState(sampler, model=model, n_samples=numSamples)
    vars = nk.experimental.vqs.variables_from_file(
        filename=statePath, variables=state.variables
    )
    state.variables = vars

    if HType == "XX+Z":
        H = sum(
            [
                -h * sz(hilbert, i)
                - 4.0 * sx(hilbert, i) * sx(hilbert, (i + 1) % hilbert.size)
                for i in range(hilbert.size)
            ]
        )
    elif HType == "ZZ+X":
        H = sum(
            [
                -h * sx(hilbert, i)
                - 4.0 * sz(hilbert, i) * sz(hilbert, (i + 1) % hilbert.size)
                for i in range(hilbert.size)
            ]
        )

    return state.expect(H).mean.real


gs_energies = [
    -32.0200030291461,
    -32.50192088865394,
    -34.033420730875584,
    -36.68735195432693,
    -40.7600324728673,
    -46.71244261642596,
    -53.50163901612963,
    -60.520439029212625,
    -67.93527314163262,
]

hs = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# hs = [0.1, 0.5, 4.0]
for ansatz in ["RNN"]:
    it = 0
    for h in hs:
        for HType in ["XX+Z"]:
            if h == 4.0 and HType == "ZZ+X":
                continue
            # data = json.load(open(f"logs/{HType}_{ansatz}_h={h}.log", "r"))
            # iters = data["Energy"]["iters"]
            sigmas = pd.read_csv(
                f"traindata/isingz/old/isingTrainingSet_{HType}_h={h}.csv",
                delimiter="\t",
                header=None,
            ).to_numpy()
            ps = pd.read_csv(
                f"traindata/isingz/old/ps_{HType}_h={h}.csv",
                delimiter="\t",
                header=None,
            ).to_numpy()
            logPs = np.log(ps)
            print(np.prod(sigmas, axis=-1))
            # metric = np.array(data["Energy"]["Mean"])
            # plt.plot(
            #     iters,
            #     metric,
            #     label=f"{h}",
            #     # color="red" if HType == "XX+Z" else "blue",
            # )
            # plt.scatter(
            #     iters,
            #     metric,
            #     edgecolor="black",
            #     # color="red" if HType == "XX+Z" else "blue",
            # )
            # metric = KL(sigmas, f"logs/{HType}_RNN_h={h}.mpack", 50000)
            # # metric = energy(HType, f"logs/{HType}_{ansatz}_h={h}.mpack", 5000)
            # # metric = gradVariance(sigmas, f"logs/{HType}_{ansatz}_h={h}.mpack")
            # print(h, metric)
            # plt.scatter(
            #     h,
            #     metric,
            #     edgecolor="black",
            #     color="red" if ansatz == "RNN" else "blue",
            #     # color="red" if HType == "XX+Z" else "blue",
            #     label=HType,
            # )
            #
            # it += 1
# plt.plot(
#     data["KL"]["iters"],
#     -40.76 * np.ones(len(metric)),
#     color="black",
#     linestyle="--",
#     linewidth=2,
#     label="Exact",
# )

# plt.yscale("log")
# plt.xlabel("Training Iterations")
plt.xlabel("h")
plt.ylabel(r"$\log(\mathrm(Det)[F])$")
plt.tight_layout()
# plt.ylabel(r"$D_{KL}(p_{mps}| p_{rnn})$")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
