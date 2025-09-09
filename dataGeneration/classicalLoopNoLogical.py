import numpy as np
import pandas as pd
from ncon import ncon
from tqdm import tqdm


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
    # horizontals: edge (x,y) -- (x+1,y)
    for x in range(Lx):
        for y in range(Ly):
            yield ("h", x, y)
    # verticals: edge (x,y) -- (x,y+1)
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

    Args:
      Lx, Ly : lattice dimensions (integers)
      p      : bit-flip probability in [0,1]
      s_bits : iterable of length E=2*Lx*Ly with 0/1 for each edge, ordered as:
               all horizontal edges row-major, then all vertical edges row-major.
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
    k = 0  # index in s_bits
    for kind, x, y in _edge_iter(Lx, Ly):
        bit = s[k]
        k += 1
        # weights: prefer matching the hidden loop bit to observed bit s_e
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


if __name__ == "__main__":
    Lx, Ly, L = 4, 4, 4
    E = 2 * Lx * Ly

    for p in np.arange(0, 0.2, 0.01):
        measurements = pd.read_csv(
            f"../traindata/toricpqu/d=2/measurements_L={L}_p={p:.3f}.csv",
            delimiter=" ",
            header=None,
        )
        measurements = measurements.dropna(axis=1, how="all")
        measurements = measurements.to_numpy()
        ps = []
        for measurement in tqdm(measurements):
            s_bits = ((measurement + 1) / 2).astype(int)
            P = probability_bitstring_ncon(Lx, Ly, p, s_bits)
            ps.append(P)

        data = pd.DataFrame(ps)
        data.to_csv(
            f"../traindata/toricpqu/d=2/ps_L={L}_p={p:.3f}.csv",
            index=False,
            header=False,
        )
