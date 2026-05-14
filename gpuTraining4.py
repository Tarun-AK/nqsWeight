import pickle

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pandas as pd
from models.conv import Conv2D
from models.mpdoRnn import MPDO
from models.resnet import ResidualConv2D
from models.transformer import Transformer
from multiTrainer import MultiTrainer
from netket.experimental.models import FastLSTMNet
from optax import adabelief
from tqdm import trange


def generate_spin_configs(mode):
    n_samples = 100000
    n_sites = 36

    if mode == "local1d":
        configs = np.random.choice([-1, 1], size=(n_samples, n_sites))
        constrained_indices = [0, 1, 2]
        constraint_values = np.random.choice([-1, 1], size=n_samples)
        for idx in constrained_indices:
            configs[:, idx] = constraint_values

    elif mode == "nonlocal1d":
        configs = np.random.choice([-1, 1], size=(n_samples, n_sites))
        constrained_indices = [0, 17, 34]
        constraint_values = np.random.choice([-1, 1], size=n_samples)
        for idx in constrained_indices:
            configs[:, idx] = constraint_values

    elif mode == "local2d":
        L = 6
        configs = np.random.choice([-1, 1], size=(n_samples, L, L))
        constrained_positions = [(0, 0), (0, 1), (0, 2)]
        constraint_values = np.random.choice([-1, 1], size=n_samples)
        for pos in constrained_positions:
            configs[:, pos[0], pos[1]] = constraint_values
        configs = configs.reshape(n_samples, n_sites)

    elif mode == "nonlocal2d":
        L = 6
        configs = np.random.choice([-1, 1], size=(n_samples, L, L))
        constrained_positions = [(0, 0), (3, 3), (5, 4)]
        constraint_values = np.random.choice([-1, 1], size=n_samples)
        for pos in constrained_positions:
            configs[:, pos[0], pos[1]] = constraint_values
        configs = configs.reshape(n_samples, n_sites)

    return configs


def count_params(model, input_example):
    params = model.init(jax.random.PRNGKey(0), input_example)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


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


def runTraining(params):
    d = 1
    hilbert_spin = 0.5
    for L in [64]:
        for HType in ["GHZ"]:
            datasets = []
            for param in params:
                if "local" not in HType:
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
                    if "PQ" in HType:
                        sigmas = reorder_serialization(sigmas, L)
                        sigmas = group(sigmas, L)
                        hilbert_spin = 1.5
                else:
                    sigmas = generate_spin_configs(mode=HType)
                datasets.append(sigmas)

            g = nk.graph.Hypercube(length=len(sigmas[0]), n_dim=1, pbc=True)
            hilbert = nk.hilbert.Spin(s=hilbert_spin, N=g.n_nodes)

            ######################################################################################################
            # num_layers = 2  # 4
            # d_model = 24  # 128 #128
            # dff = 16  # 128 # 512
            # num_heads = 4  # 8
            # model = Transformer(
            #     hilbert=hilbert,
            #     num_layers=num_layers,
            #     num_heads=num_heads,
            #     d_model=d_model,
            #     dff=dff,
            #     autoreg=True,
            # )
            # model = Conv2D(
            #     layers=9,
            #     features=20,
            #     kernel_size=(L // 2, L // 2),
            #     hilbert=hilbert,
            #     L=L,
            #     W=L,
            #     channels=1,
            # )
            # model = nk.models.FastARNNConv2D(
            #     layers=7, features=32, kernel_size=(L // 2, L // 2), hilbert=hilbert
            # )

            model = MPDO(hilbert=hilbert, p=0.5, extra_hidden=10)
            # model = ResidualConv2D(
            #     layers=4,  # 8 residual blocks
            #     features=24,  # 64 feature maps (adjust to 96 or 128 for more capacity)
            #     kernel_size=(
            #         3,
            #         5,
            #     ),  # (height, width) - asymmetric for better receptive field
            #     hilbert=hilbert,
            #     L=L,
            #     W=L,
            #     channels=1,
            #     use_dropout=True,  # Enable if you see overfitting
            #     dropout_rate=0.2,  # Standard rate when dropout is enabled
            # )
            # model = FastLSTMNet(layers=3, features=24, hilbert=hilbert, graph=g)
            print(count_params(model, np.ones((1, len(sigmas[0])))))
            ######################################################################################################
            name = "MPDO"
            trainer = MultiTrainer(
                datasets=jnp.array(datasets),
                models_per_dataset=5,
                model=model,
                batch_size=128,
                sample_input=jnp.ones((1, hilbert.size)),
                optimizer=adabelief(5e-5),
                # in_path=f"logs/{HType.lower()}/d={d}/{name}_oct_L={L}.pkl",
                out=f"logs/{HType.lower()}/d={d}/{name}_oct_L={L}.pkl",
                save_freq=10000,
                base_key=1,
            )
            rng = jax.random.PRNGKey(42)
            losses = []
            for _ in trange(400000):
                rng, sub = jax.random.split(rng)
                losses.append(trainer.run_step(sub))

            with open(
                f"logs/{HType.lower()}/d={d}/{name}_oct_L={L}.log",
                "wb",
            ) as f:
                pickle.dump(losses, f)


if __name__ == "__main__":
    ps = [f"{p:.3f}" for p in np.arange(0.0, 1.0, 0.05)]
    runTraining(ps)
