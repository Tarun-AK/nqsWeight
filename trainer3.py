import itertools
import os
import pickle

import jax
import netket as nk
import numpy as np
import pandas as pd
import ray

# from netket.experimental.models import FastLSTMNet
from netket.operator.spin import sigmax, sigmaz
from optax import adabelief, adam, adamw, sgd

from models.lstm import FastLSTMNet
from models.mpdoRnn import MPDO
from regularizedQSR import QSR as rQSR

ray.init(address=os.environ["ip_head"])

L = 10
d = 1

sx = lambda a, b: (1 / 2) * sigmax(a, b)
sz = lambda a, b: (1 / 2) * sigmaz(a, b)

import numpy as np


def rangeSampler(L, r="long"):
    np.random.seed(10)
    samples = np.random.choice([-1, 1], size=(100000, L))
    if r == "long":
        samples[:, L - 2] = samples[:, 1]
    if r == "short":
        samples[:, L // 2 + 1] = samples[:, L // 2]

    return samples


def group(configs, L):
    num_samples = configs.shape[0]
    pairs = configs.reshape(num_samples, L * L, 2)
    idx = (pairs[..., 0] == -1).astype(int) * 2 + (pairs[..., 1] == -1).astype(int)
    mapping = np.array([-3, -1, 1, 3])
    return mapping[idx]


def reorder_serialization(new_array, L, d=2):
    N_samples, N = new_array.shape
    assert N == L * L * d, f"Array length {N} does not match L*L*d={L*L*d}"
    arr = new_array.reshape(N_samples, d, L, L)
    arr = arr.transpose(0, 2, 3, 1)
    old_array = arr.reshape(N_samples, L * L * d)
    return old_array


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


def callback(step, log_data, driver):
    if step % 500 == 0:
        # log_data["nll"] = driver.loss_val
        samples = driver.state.sample()[0]
        log_data["parity"] = np.mean(np.prod(samples, axis=-1), axis=0)
    return True


@ray.remote(num_cpus=4)
def runToric(param, copy, load=False):
    dh = 10
    for HType in ["GHZ"]:
        if "Range" in HType:
            sigmas = rangeSampler(L, r="Short")
        elif "ToricCoherent" not in HType:
            letter = "h" if "Ising" in HType else "p"
            delim = "\t" if "Ising" in HType else " "
            sigmas = pd.read_csv(
                f"traindata/{HType.lower()}/d={d}/measurements_L={L}_{letter}={param}.csv",
                delimiter=delim,
                header=None,
            )
            if "Toric" in HType:
                sigmas = sigmas.dropna(axis=1, how="all")
                sigmas = sigmas.to_numpy()
            else:
                sigmas = sigmas.to_numpy()[:, :L]
            if "PQ" in HType:
                sigmas = reorder_serialization(sigmas, L)
                sigmas = group(sigmas, L)
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

        Us = len(sigmas) * [len(sigmas[0]) * "I"]
        # g = nk.graph.Hypercube(length=len(sigmas[0]), n_dim=1, pbc=True)
        g = nk.graph.Hypercube(length=L, n_dim=d, pbc=True)
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
        # model = MPDO(hilbert=hilbert, p=0.5, extra_hidden=dh)
        # model = nk.models.RBM(alpha=1.5, param_dtype=float)
        # model = SymmetrizedRBM(alpha=1.5)
        ######################################################################################################
        vstate = nk.vqs.MCState(sampler, model=model, n_samples=1028)

        if load:
            path = f"logs/{HType.lower()}/d={d}/modLSTM_adabelief_dh={dh}_param={param}_copy={copy}_L={L}"
            vars = nk.experimental.vqs.variables_from_file(
                filename=path, variables=vstate.variables
            )
            vstate.variables = vars
        opt = adabelief(5e-5)
        qsr = rQSR(
            training_data=(sigmas, Us),
            training_batch_size=128,
            optimizer=opt,
            variational_state=vstate,
        )
        qsr.run(
            out=f"logs/{HType.lower()}/d={d}/modLSTM_adabelief_dh={dh}_param={param}_copy={copy}_L={L}",
            n_iter=500000,
            callback=callback,
        )


if __name__ == "__main__":
    ps = [f"{p:.3f}" for p in [0.45, 0.46, 0.47, 0.48, 0.49]]
    copies = np.arange(0, 10)
    iterator = itertools.product(ps, copies)

    ids = [runToric.remote(p, copy, load=False) for p, copy in iterator]
    ray.get(ids)

    # runs = [runToric(p, copy, load=False) for p, copy in iterator]
