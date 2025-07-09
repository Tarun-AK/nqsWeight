import itertools
import json
import os
import pickle

import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd
import ray
from netket.experimental import QSR
from netket.operator.spin import sigmax, sigmaz
from optax import adabelief, adam, adamw, sgd

# from netket.experimental.models.fast_rnn import FastGRUNet1D, FastLSTMNet
from models.lstm import FastLSTMNet
from regularizedQSR import QSR as rQSR

plt.style.use("~/plotStyle.mplstyle")

# ray.init(address=os.environ["ip_head"])

L = 10
d = 1

sx = lambda a, b: (1 / 2) * sigmax(a, b)
sz = lambda a, b: (1 / 2) * sigmaz(a, b)


def callback(step, log_data, driver):
    s = np.ones((1, L))
    logp = 2 * driver.state.log_value(s)
    log_data["NLL"] = logp
    return True


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


# @ray.remote(num_cpus=8)
def runToric(param, copy, load=False):
    dh = 4
    for HType in ["GHZ"]:

        if "ToricCoherent" not in HType:
            sigmas = pd.read_csv(
                f"traindata/{HType.lower()}/d={d}/measurements_L={25}_p={param}.csv",
                delimiter=" ",
                header=None,
            )
            sigmas = sigmas.dropna(axis=1, how="all")
            sigmas = sigmas.to_numpy()
            sigmas = sigmas[:, :L]
        else:
            param = int(eval(param))
            filename = f"/gpfs/anegari/Surface_code_CMI/Generating_sample_tarun/data/338993/L_chunks/L{L}.pkl"
            with open(filename, "rb") as f:
                etaDict = pickle.load(f)

            tempFunc = lambda x, y: (
                compute_syndromes(x, y)
                if HType == "ToricCoherent"
                else (x.flatten(), y)
            )
            sigmas = np.array(
                [
                    tempFunc(etaDict[param][i]["sample"], etaDict[param][i]["logp"])[0]
                    for i in range(len(etaDict[param]))
                ]
            )
            print(sigmas.shape)
            # assert sigmas.shape == (50000, L**d)

        Us = len(sigmas) * [len(sigmas[0]) * "I"]
        g = nk.graph.Hypercube(length=len(sigmas[0]), n_dim=1, pbc=True)
        # g = nk.graph.Hypercube(length=L, n_dim=d, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        # sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)

        ######################################################################################################
        # num_layers = 2  # 4
        # d_model = 10  # 128 #128
        # dff = 8  # 128 # 512
        # num_heads = 2  # 8
        #
        # model = Transformer(
        #     hilbert=hilbert,
        #     num_layers=num_layers,
        #     num_heads=num_heads,
        #     d_model=d_model,
        #     dff=dff,
        #     autoreg=True,
        # )
        model = FastLSTMNet(layers=3, features=dh, hilbert=hilbert, graph=g)
        # model = nk.models.RBM(alpha=1.5, param_dtype=float)
        # model = SymmetrizedRBM(alpha=1.5)
        ######################################################################################################
        vstate = nk.vqs.MCState(sampler, model=model, n_samples=256)

        if load:
            path = (
                f"logs/{HType.lower()}/d={d}/RNN_triple_param={param}_copy={copy}_L={L}"
            )
            vars = nk.experimental.vqs.variables_from_file(
                filename=path, variables=vstate.variables
            )
            vstate.variables = vars

        opt = adabelief(5e-5)
        print(vstate.n_parameters)

        qsr = rQSR(
            training_data=(sigmas, Us),
            training_batch_size=128,
            optimizer=opt,
            variational_state=vstate,
        )
        qsr.run(
            out=f"logs/{HType.lower()}/d={d}/Test_param={param}_copy={copy}_L={L}",
            n_iter=50000,
            callback=callback,
        )
        print(qsr.state.parameters)


if __name__ == "__main__":
    ps = [f"{p:.3f}" for p in [0.0]]
    copies = np.arange(1)
    iterator = itertools.product(ps, copies)

    ids = [runToric(p, copy, load=False) for p, copy in iterator]

    data = json.load(open(f"logs/GHZ/d={d}/Test_param=0.000_copy={0}_L={L}.log"))
    NLL = data["NLL"]["value"]
    plt.plot(-np.array(NLL))
    plt.show()
    print(NLL[-1])
