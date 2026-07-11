import os

import numpy as np
import pandas as pd
import ray
from ncon import ncon
from tqdm import tqdm

# ray.init(address=os.environ["ip_head"], ignore_reinit_error=True)


def z2_vertex_tensor():
    """G[iN,iE,iS,iW] = 1 iff parity(iN+iE+iS+iW) is even; else 0."""
    idx = np.indices((2, 2, 2, 2))
    return ((idx.sum(axis=0) % 2) == 0).astype(np.int8)


# --- NEW: parity MPO pieces (bond dim 2) ---
def _F_tensor():
    """F[a,b;i]=1 if b == a XOR i; else 0. Shape (2,2,2)."""
    F = np.zeros((2, 2, 2), dtype=np.int8)
    for a in range(2):
        for i in range(2):
            F[a, a ^ i, i] = 1
    return F


_v0 = np.array([1, 0], dtype=np.int8)  # selects even parity


def _append_even_parity_mpo_on_indices(tensors, connects, idx_list, next_label):
    """
    Append MPO: v0 -- F(i1) -- F(i2) -- ... -- F(im) -- v0
    Enforces XOR(i1..im) = 0.
    """
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
        connects.append([cur, nxt, int(i_idx)])
        cur = nxt

    tensors.append(_v0)
    connects.append([cur])
    return next_label


def probability_bitstring_rectangle_projected_even_top_parity(Lx, Ly, p, s_bits):
    """
    OPEN boundaries (same as your current open code), but additionally project to the
    EVEN top-boundary vertical parity sector.

    s_bits ordering (COLUMN-MAJOR):
      E = (Lx+1)*Ly + (Ly+1)*Lx
      1) Horizontals: for x_edge=0..Lx, y=0..Ly-1
      2) Verticals  : for x=0..Lx-1, y_edge=0..Ly
    """
    del_indices = (Ly + 1) * np.arange(Lx)
    s_bits = np.delete(s_bits, del_indices)
    s_bits = np.concatenate(
        [
            np.zeros(Ly),
            s_bits,
        ]
    )
    s = np.asarray(s_bits, dtype=np.int8).ravel()

    G = z2_vertex_tensor()
    labels = np.arange(1, 1 + 4 * Lx * Ly, dtype=int).reshape(Lx, Ly, 4)

    tensors = []
    connects = []

    # Vertex tensors (COLUMN-MAJOR: x then y)
    for x in range(Lx):
        for y in range(Ly):
            tensors.append(G)
            connects.append(labels[x, y].tolist())  # [N,E,S,W]

    # Internal-edge weights (also used as 2-leg "forwarders")
    W0 = np.diag([1.0 - p, p])  # observed 0
    W1 = np.diag([p, 1.0 - p])  # observed 1
    W_for_bit = (W0, W1)

    # Boundary tensors (open-specific)
    B_fixed0 = np.array([1.0, 0.0], dtype=float)
    B0 = np.array([1.0 - p, p], dtype=float)
    B1 = np.array([p, 1.0 - p], dtype=float)
    B_for_bit = (B0, B1)

    # NEW: labels for the MPO legs
    next_label = int(labels.max())
    top_parity_legs = []  # collect f_x for the top boundary projector

    k = 0

    # Horizontals with boundary slots (COLUMN-MAJOR) -- unchanged
    for x_edge in range(Lx + 1):
        for y in range(Ly):
            bit = int(s[k])
            k += 1

            if x_edge == 0:
                if bit != 0:
                    return 0.0
                a = labels[0, y, 3]  # W of (0,y)
                tensors.append(B_fixed0)
                connects.append([int(a)])

            elif x_edge == Lx:
                if bit != 0:
                    return 0.0
                a = labels[Lx - 1, y, 1]  # E of (Lx-1,y)
                tensors.append(B_fixed0)
                connects.append([int(a)])

            else:
                a = labels[x_edge - 1, y, 1]  # E of (x_edge-1,y)
                b = labels[x_edge, y, 3]  # W of (x_edge,y)
                tensors.append(W_for_bit[bit])
                connects.append([int(a), int(b)])

    # Verticals with boundary slots (COLUMN-MAJOR)
    for x in range(Lx):
        for y_edge in range(Ly + 1):
            bit = int(s[k])
            k += 1

            if y_edge == 0:
                # bottom boundary unchanged (1-leg noisy termination)
                a = labels[x, 0, 0]  # N of (x,0)
                tensors.append(B_for_bit[bit])
                connects.append([int(a)])

            elif y_edge == Ly:
                # NEW: top boundary becomes 2-leg diagonal "forwarder" to parity MPO
                a = labels[x, Ly - 1, 2]  # S of (x,Ly-1)
                next_label += 1
                f = next_label  # exposed leg to MPO
                tensors.append(W_for_bit[bit])  # diag([w0,w1]) implements w_i * δ_{i,f}
                connects.append([int(a), int(f)])
                top_parity_legs.append(f)

            else:
                # bulk vertical unchanged
                a = labels[x, y_edge - 1, 2]  # S of (x,y_edge-1)
                b = labels[x, y_edge, 0]  # N of (x,y_edge)
                tensors.append(W_for_bit[bit])
                connects.append([int(a), int(b)])

    # NEW: enforce even parity on the top boundary exposed legs
    next_label = _append_even_parity_mpo_on_indices(
        tensors, connects, top_parity_legs, next_label
    )

    N_s = ncon(tensors, connects)

    # NEW: projected Z is half the unprojected open Z (one independent Z2 constraint)
    Z = 2 ** (Lx * Ly + Lx - Ly - 1)
    return float(N_s) / Z


@ray.remote
def process_measurement(Lx, Ly, p, measurement):
    s_bits = (-(measurement - 1) / 2).astype(int)
    return probability_bitstring_rectangle_projected_even_top_parity(Lx, Ly, p, s_bits)


@ray.remote(num_cpus=4)
def process_p(Lx, Ly, L, p):
    measurements = pd.read_csv(
        f"../testdata/toricopenpqv3/d=2/measurements_L={L}_p={p:.3f}.csv",
        delimiter=" ",
        header=None,
    )
    measurements = measurements.dropna(axis=1, how="all").to_numpy()

    # Inner-level parallelism over measurements with progress bar
    inner_tasks = [process_measurement.remote(Lx, Ly, p, m) for m in measurements]

    # Process with progress bar for this p value
    ps = []
    desc = f"p={p:.3f}"
    for task in tqdm(inner_tasks, desc=desc, leave=False):
        ps.append(ray.get(task))

    data = pd.DataFrame(ps)
    outpath = f"../testdata/toricopenpqv3/d=2/ps_L={L}_p={p:.3f}.csv"
    data.to_csv(outpath, index=False, header=False)
    return p


if __name__ == "__main__":
    Lx, Ly, L = 4, 3, 4
    ps = np.arange(0, 0.2, 0.01)

    # Launch all p-value tasks
    tasks = [process_p.remote(Lx, Ly, L, float(p)) for p in ps]

    # Collect results with outer progress bar
    results = []
    for task in tqdm(tasks, desc="Overall progress", total=len(ps)):
        results.append(ray.get(task))

    print(f"\nCompleted all {len(results)} p-values")
    ray.shutdown()
