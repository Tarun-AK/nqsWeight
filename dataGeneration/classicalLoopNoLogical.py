import os

import numpy as np
import pandas as pd
import ray
from ncon import ncon
from tqdm import tqdm

# -----------------------------
# Local tensors and helpers
# -----------------------------
ray.init(address=os.environ["ip_head"], ignore_reinit_error=True)


def z2_vertex_tensor():
    """G(iN,iE,iS,iW)=1 iff (iN+iE+iS+iW) mod 2 == 0; else 0."""
    G = np.zeros((2, 2, 2, 2), dtype=np.int64)
    for iN in range(2):
        for iE in range(2):
            for iS in range(2):
                for iW in range(2):
                    G[iN, iE, iS, iW] = 1 if ((iN + iE + iS + iW) % 2 == 0) else 0
    return G


def _halfedge_labels(Lx, Ly, start=1):
    """Unique positive label for each vertex leg (0=N,1=E,2=S,3=W)."""
    lab = start
    leg = {}
    for x in range(Lx):
        for y in range(Ly):
            for d in range(4):
                leg[(x, y, d)] = lab
                lab += 1
    return leg, lab


def _edge_iter(Lx, Ly):
    """Yield edges in the fixed order used for s_bits: all horizontals, then all verticals."""
    for x in range(Lx):
        for y in range(Ly):
            yield ("h", x, y)
    for x in range(Lx):
        for y in range(Ly):
            yield ("v", x, y)


def _closed_form_Z(Lx, Ly):
    """Trace (partition function) for the torus grid."""
    return 2 ** (Lx * Ly + 1)


# -----------------------------
# Main: probability of a bitstring after bit flips
# -----------------------------
def probability_bitstring_ncon(Lx, Ly, p, s_bits):
    """
    Return P(s) = <s| rho_p |s>, where rho_p is the uniform closed-loop mixture
    on the Lx x Ly torus with independent bit flips at rate p on every edge.
    """
    s = np.asarray(s_bits, dtype=int).ravel()
    E = 2 * Lx * Ly
    if s.size != E:
        raise ValueError(f"s_bits must have length {E} for Lx={Lx}, Ly={Ly}.")

    G = z2_vertex_tensor()
    tensors, connects = [], []

    # Place all vertex tensors with unique half-edge labels
    leg, last_label = _halfedge_labels(Lx, Ly, start=1)
    for x in range(Lx):
        for y in range(Ly):
            iN = leg[(x, y, 0)]
            iE = leg[(x, y, 1)]
            iS = leg[(x, y, 2)]
            iW = leg[(x, y, 3)]
            tensors.append(G)
            connects.append([iN, iE, iS, iW])

    # Insert a 2x2 diagonal weight on each undirected edge, conditioned on s_e
    k = 0
    for kind, x, y in _edge_iter(Lx, Ly):
        bit = s[k]
        k += 1
        w0 = (1 - p) if bit == 0 else p
        w1 = (1 - p) if bit == 1 else p
        W = np.diag([w0, w1]).astype(float)

        if kind == "h":
            a = leg[(x, y, 1)]  # E of (x,y)
            b = leg[((x + 1) % Lx, y, 3)]  # W of (x+1,y)
        else:  # "v"
            a = leg[(x, y, 2)]  # S of (x,y)
            b = leg[(x, (y + 1) % Ly, 0)]  # N of (x,y+1)

        tensors.append(W)
        connects.append([a, b])

    # Contract to get the unnormalized score N(s)
    N_s = ncon(tensors, connects)

    # Normalize by Z (trace is unchanged by TP noise)
    Z = _closed_form_Z(Lx, Ly)
    return N_s / Z


# -----------------------------
# Nested Ray parallelization
# -----------------------------


@ray.remote
def process_measurement(Lx, Ly, p, measurement):
    """Compute probability for a single measurement."""
    s_bits = (-(measurement - 1) / 2).astype(int)
    return probability_bitstring_ncon(Lx, Ly, p, s_bits)


@ray.remote(num_cpus=4)
def process_p(Lx, Ly, L, p):
    """Process all measurements for a given p in parallel using nested Ray tasks."""
    print(f"Starting p={p:.3f}")
    measurements = pd.read_csv(
        f"../testdata/storicpqu/d=2/measurements_L={L}_p={p:.3f}.csv",
        delimiter=" ",
        header=None,
    )
    measurements = measurements.dropna(axis=1, how="all").to_numpy()

    # Inner-level parallelism: one Ray task per measurement
    inner_tasks = [
        process_measurement.remote(Lx, Ly, p, measurement)
        for measurement in measurements
    ]
    ps = ray.get(inner_tasks)

    data = pd.DataFrame(ps)
    outpath = f"../testdata/storicpqu/d=2/ps_L={L}_p={p:.3f}.csv"
    data.to_csv(outpath, index=False, header=False)
    print(f"Finished p={p:.3f} -> {outpath}")
    return p


if __name__ == "__main__":
    Lx, Ly, L = 6, 6, 6
    ps = np.arange(0, 0.2, 0.01)

    tasks = [process_p.remote(Lx, Ly, L, float(p)) for p in ps]
    done = ray.get(tasks)

    print(f"All tasks finished: {done}")
    ray.shutdown()
