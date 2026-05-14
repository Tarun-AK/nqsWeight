import pickle
import sys

import jax.numpy as jnp
import jax.tree_util as tree


def combine_pkls(pkl1_path, pkl2_path, outpath):
    # Load both parameter sets
    with open(pkl1_path, "rb") as f:
        params1 = pickle.load(f)

    with open(pkl2_path, "rb") as f:
        params2 = pickle.load(f)

    # Concatenate parameters along the first axis (model dimension)
    combined_params = tree.tree_map(
        lambda p1, p2: jnp.concatenate([p1, p2], axis=0), params1, params2
    )

    # Save combined parameters with the name of the first pkl
    with open(outpath, "wb") as f:
        pickle.dump(combined_params, f)

    print(f"Combined {pkl1_path} and {pkl2_path}")
    print(f"Saved combined parameters to {outpath}")


if __name__ == "__main__":
    HType = "ToricOpenPQUV3"
    d = 2
    name = "ResidualConv2D_nl=2_dh=24_dp=0.2"
    L = 4

    out = f"logs/{HType.lower()}/d={d}/{name}_oct_L={L}.pkl"
    out2 = f"logs/{HType.lower()}/d={d}/{name}_42_oct_L={L}.pkl"
    final = f"logs/{HType.lower()}/d={d}/{name}_combined_oct_L={L}.pkl"

    combine_pkls(out, out2, final)
