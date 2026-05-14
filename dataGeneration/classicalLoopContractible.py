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
    G = np.zeros((2, 2, 2, 2), dtype=np.int64)
    for iN in range(2):
        for iE in range(2):
            for iS in range(2):
                for iW in range(2):
                    G[iN, iE, iS, iW] = 1 if ((iN + iE + iS + iW) % 2 == 0) else 0
    return G


def _halfedge_labels(Lx, Ly, start=1):
    lab = start
    leg = {}
    for x in range(Lx):
        for y in range(Ly):
            for d in range(4):
                leg[(x, y, d)] = lab
                lab += 1
    return leg, lab


def _edge_iter(Lx, Ly):
    for x in range(Lx):
        for y in range(Ly):
            yield ("h", x, y)
    for x in range(Lx):
        for y in range(Ly):
            yield ("v", x, y)


def _closed_form_Z_all(Lx, Ly):
    return 2 ** (Lx * Ly + 1)


def _closed_form_Z_contractible(Lx, Ly):
    return 2 ** (Lx * Ly - 1)


def _F_tensor():
    F = np.zeros((2, 2, 2), dtype=np.int64)
    for a in range(2):
        for i in range(2):
            F[a, a ^ i, i] = 1
    return F


_v0 = np.array([1, 0], dtype=np.int64)


def _append_even_parity_mpo_on_indices(tensors, connects, idx_list, next_label):
    F = _F_tensor()
    next_label += 1
    left = next_label
    tensors.append(_v0)
    connects.append([left])
    cur = left
    for i_idx in idx_list:
        next_label += 1
        nxt = next_label
        tensors.append(F)
        connects.append([cur, nxt, i_idx])
        cur = nxt
    tensors.append(_v0)
    connects.append([cur])
    return next_label


def _copy3():
    C3 = np.zeros((2, 2, 2), dtype=np.int64)
    for b in (0, 1):
        C3[b, b, b] = 1
    return C3


# -----------------------------
# Main: probability of a bitstring after bit flips
# -----------------------------


def probability_bitstring_ncon(Lx, Ly, p, s_bits, contractible_only=True):
    s = np.asarray(s_bits, dtype=int).ravel()
    E = 2 * Lx * Ly
    if s.size != E:
        raise ValueError(f"s_bits must have length {E} for Lx={Lx}, Ly={Ly}.")

    G = z2_vertex_tensor()
    tensors, connects = [], []

    leg, next_label = _halfedge_labels(Lx, Ly, start=1)
    for x in range(Lx):
        for y in range(Ly):
            iN = leg[(x, y, 0)]
            iE = leg[(x, y, 1)]
            iS = leg[(x, y, 2)]
            iW = leg[(x, y, 3)]
            tensors.append(G)
            connects.append([iN, iE, iS, iW])

    weight_pos = {}
    k = 0
    for kind, x, y in _edge_iter(Lx, Ly):
        bit = s[k]
        k += 1
        w0 = (1 - p) if bit == 0 else p
        w1 = (1 - p) if bit == 1 else p
        W = np.diag([w0, w1]).astype(float)

        if kind == "h":
            a = leg[(x, y, 1)]
            b = leg[((x + 1) % Lx, y, 3)]
        else:
            a = leg[(x, y, 2)]
            b = leg[(x, (y + 1) % Ly, 0)]

        tensors.append(W)
        connects.append([a, b])
        weight_pos[a] = len(tensors) - 1

    if contractible_only:
        C3 = _copy3()
        wrap_x_legs = [leg[(Lx - 1, y, 1)] for y in range(Ly)]
        wrap_y_legs = [leg[(x, Ly - 1, 2)] for x in range(Lx)]
        jF_x, jF_y = [], []

        for i in wrap_x_legs:
            next_label += 1
            jW = next_label
            next_label += 1
            jF = next_label
            tensors.append(C3)
            connects.append([i, jW, jF])
            widx = weight_pos[i]
            connects[widx][0] = jW
            jF_x.append(jF)

        for i in wrap_y_legs:
            next_label += 1
            jW = next_label
            next_label += 1
            jF = next_label
            tensors.append(C3)
            connects.append([i, jW, jF])
            widx = weight_pos[i]
            connects[widx][0] = jW
            jF_y.append(jF)

        next_label = _append_even_parity_mpo_on_indices(
            tensors, connects, jF_x, next_label
        )
        next_label = _append_even_parity_mpo_on_indices(
            tensors, connects, jF_y, next_label
        )

    N_s = ncon(tensors, connects)
    Z = (
        _closed_form_Z_contractible(Lx, Ly)
        if contractible_only
        else _closed_form_Z_all(Lx, Ly)
    )
    return N_s / Z


# -----------------------------
# Nested Ray parallelization
# -----------------------------


@ray.remote
def process_measurement(Lx, Ly, p, measurement):
    s_bits = ((measurement + 1) / 2).astype(int)
    return probability_bitstring_ncon(Lx, Ly, p, s_bits)


@ray.remote(num_cpus=60)
def process_p(Lx, Ly, L, p):
    print(f"Starting p={p:.3f}")
    measurements = pd.read_csv(
        f"../testdata/toricpq/d=2/measurements_L={L}_p={p:.3f}.csv",
        delimiter=" ",
        header=None,
    )
    measurements = measurements.dropna(axis=1, how="all").to_numpy()

    # Inner-level parallelism over measurements
    inner_tasks = [process_measurement.remote(Lx, Ly, p, m) for m in measurements]
    ps = ray.get(inner_tasks)

    data = pd.DataFrame(ps)
    outpath = f"../testdata/toricpq/d=2/ps_L={L}_p={p:.3f}.csv"
    data.to_csv(outpath, index=False, header=False)
    print(f"Finished p={p:.3f} -> {outpath}")
    return p


if __name__ == "__main__":
    Lx, Ly, L = 6, 6, 6
    ps = np.arange(0, 0.2, 0.01)
    tasks = [process_p.remote(Lx, Ly, L, float(p)) for p in ps]
    ray.get(tasks)
    ray.shutdown()
