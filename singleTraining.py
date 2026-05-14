import argparse
import pickle

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pandas as pd
from optax import adabelief
from tqdm import trange

# Assuming your MultiTrainer is in multiTrainer.py
from multiTrainer import MultiTrainer


def group(configs, L):
    num_samples = configs.shape[0]
    pairs = configs.reshape(num_samples, L * L, 2)
    idx = (pairs[..., 0] == -1).astype(int) * 2 + (pairs[..., 1] == -1).astype(int)
    mapping = np.array([-3, -1, 1, 3])
    return mapping[idx]


def reorder_serialization(new_array, L, d=2):
    N_samples, N = new_array.shape
    arr = new_array.reshape(N_samples, d, L, L)
    arr = arr.transpose(0, 2, 3, 1)
    return arr.reshape(N_samples, L * L * d)


def run_single_param_training(param):
    d = 2
    L = 8
    HType = "ToricOpenPQUV3"
    name = "Conv2D_nl=11"

    # 1. Load Data for the specific parameter
    letter = "h" if "Ising" in HType else "p"
    delim = "\t" if "Ising" in HType else " "
    path = f"traindata/{HType.lower()}/d={d}/measurements_L={L}_{letter}={param}.csv"

    sigmas = pd.read_csv(path, delimiter=delim, header=None)
    sigmas = sigmas.dropna(axis=1, how="all").to_numpy()
    sigmas = reorder_serialization(sigmas, L)
    sigmas = group(sigmas, L)

    # Wrap in a list because MultiTrainer expects a list of datasets
    datasets = [sigmas]

    # 2. Setup Hilbert and Model
    g = nk.graph.Hypercube(length=len(sigmas[0]), n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=1.5, N=g.n_nodes)

    model = nk.models.FastARNNConv2D(
        layers=11, features=20, kernel_size=(L // 2, L // 2), hilbert=hilbert
    )

    # 3. Initialize MultiTrainer for this single process
    # We set models_per_dataset=1 since this process only handles this one param
    trainer = MultiTrainer(
        datasets=jnp.array(datasets),
        models_per_dataset=10,
        model=model,
        batch_size=128,
        sample_input=jnp.ones((1, hilbert.size)),
        optimizer=adabelief(5e-5),
        out=f"logs/{HType.lower()}/d={d}/{name}_p={param}_L={L}.pkl",
        save_freq=400000,
        base_key=0,
    )

    # 4. Training Loop
    rng = jax.random.PRNGKey(42)
    losses = []

    for i in trange(400000):
        rng, sub = jax.random.split(rng)
        loss = trainer.run_step(sub)

        if i % 1000 == 0:
            print(f"Step {i} ", flush=True)

    # 5. Save Final Logs
    with open(f"logs/{HType.lower()}/d={d}/{name}_p={param}_L={L}.log", "wb") as f:
        pickle.dump(losses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, required=True)
    args = parser.parse_args()
    run_single_param_training(args.param)
