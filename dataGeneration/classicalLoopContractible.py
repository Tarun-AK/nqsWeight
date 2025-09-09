import numpy as np
import pandas as pd
from ncon import ncon
from tqdm import tqdm

# -----------------------------
# Local tensors and helpers
# -----------------------------

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
                leg[(x,y,d)] = lab
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

def _closed_form_Z_all(Lx, Ly):
    """Trace for all winding sectors (torus)."""
    return 2 ** (Lx * Ly + 1)

def _closed_form_Z_contractible(Lx, Ly):
    """Trace restricted to the contractible sector (no winding)."""
    return 2 ** (Lx * Ly - 1)

# ---- MPO pieces (bond dim 2) ----
def _F_tensor():
    """F[a,b;i]=1 if b==a XOR i; else 0. Shape (2,2,2)."""
    F = np.zeros((2,2,2), dtype=np.int64)
    for a in range(2):
        for i in range(2):
            F[a, a ^ i, i] = 1
    return F

_v0 = np.array([1,0], dtype=np.int64)  # boundary vector selects even parity and change of parity happens here!

def _append_even_parity_mpo_on_indices(tensors, connects, idx_list, next_label):
    """
    Append MPO: v0 -- F(i1) -- F(i2) -- ... -- F(im) -- v0
    Each F has legs [bond_in, bond_out, i_k], with i_k from idx_list.
    Returns updated next_label.
    """
    F = _F_tensor()
    # left boundary
    next_label += 1
    left = next_label
    tensors.append(_v0); connects.append([left])
    cur = left
    for i_idx in idx_list:
        next_label += 1
        nxt = next_label
        tensors.append(F); connects.append([cur, nxt, i_idx])
        cur = nxt
    # right boundary
    tensors.append(_v0); connects.append([cur])
    return next_label


# ---- NEW: 3-leg copy tensor to split a leg cleanly ----
def _copy3():
    """COPY3[i, jW, jF] = 1 if i==jW==jF else 0."""
    C3 = np.zeros((2,2,2), dtype=np.int64)
    for b in (0,1):
        C3[b,b,b] = 1
    return C3

# -----------------------------
# Main: probability of a bitstring after bit flips
# -----------------------------


def probability_bitstring_ncon(Lx, Ly, p, s_bits, contractible_only=True):
    """
    Return P(s) = <s| rho_p |s>, where rho_p is the uniform closed-loop mixture
    on the Lx x Ly torus with independent bit flips at rate p on every edge.
    If contractible_only=True, project to the sector with no non-contractible loops.
    """
    s = np.asarray(s_bits, dtype=int).ravel()
    E = 2 * Lx * Ly
    if s.size != E:
        raise ValueError(f"s_bits must have length {E} for Lx={Lx}, Ly={Ly}.")

    G = z2_vertex_tensor()
    tensors, connects = [], []

    # Place all vertex tensors with unique half-edge labels
    leg, next_label = _halfedge_labels(Lx, Ly, start=1)
    for x in range(Lx):
        for y in range(Ly):
            iN = leg[(x,y,0)]
            iE = leg[(x,y,1)]
            iS = leg[(x,y,2)]
            iW = leg[(x,y,3)]
            tensors.append(G)
            connects.append([iN, iE, iS, iW])

    # Insert a 2x2 diagonal weight on each undirected edge, conditioned on s_e
    # Record where each weight (W) is, keyed by the 'a' (left/top) half-edge label.
    weight_pos = {}  # map: halfedge_label_used_as_a -> index of that W in tensors/connects
    k = 0  # index in s_bits
    for kind, x, y in _edge_iter(Lx, Ly):
        bit = s[k]; k += 1
        # weights: prefer matching the hidden loop bit to observed bit s_e
        w0 = (1 - p) if bit == 0 else p
        w1 = (1 - p) if bit == 1 else p
        W = np.diag([w0, w1]).astype(float)

        if kind == "h":
            a = leg[(x, y, 1)]                     # E of (x,y)
            b = leg[((x + 1) % Lx, y, 3)]          # W of (x+1,y)
        else:  # "v"
            a = leg[(x, y, 2)]                     # S of (x,y)
            b = leg[(x, (y + 1) % Ly, 0)]          # N of (x,y+1)

        tensors.append(W)
        connects.append([a, b])
        weight_pos[a] = len(tensors) - 1  # remember where this W sits

    # ---- FIX LOGICAL: project to contractible sector using parity MPOs ----
    if contractible_only:
        C3 = _copy3()

        # (i) Choose the wrap legs we will constrain:
        #  - horizontal wrap edges: use E-leg of (Lx-1, y)
        #  - vertical   wrap edges: use S-leg of (x, Ly-1)
        wrap_x_legs = [leg[(Lx - 1, y, 1)] for y in range(Ly)]
        wrap_y_legs = [leg[(x, Ly - 1, 2)] for x in range(Lx)]

        # For each selected leg i, split it with COPY3 into (i, jW, jF),
        # retarget the existing W to use jW (instead of i),
        # and collect jF to feed into the MPO along that cut.
        jF_x, jF_y = [], []

        for i in wrap_x_legs:
            next_label += 1; jW = next_label
            next_label += 1; jF = next_label
            # COPY3[i, jW, jF]
            tensors.append(C3); connects.append([i, jW, jF])
            # Retarget W at this edge so that it uses jW instead of i
            widx = weight_pos[i]
            assert connects[widx][0] == i, "Internal wiring assumption failed for horizontal wrap."
            connects[widx][0] = jW
            jF_x.append(jF)

        for i in wrap_y_legs:
            next_label += 1; jW = next_label
            next_label += 1; jF = next_label
            tensors.append(C3); connects.append([i, jW, jF])
            widx = weight_pos[i]
            assert connects[widx][0] == i, "Internal wiring assumption failed for vertical wrap."
            connects[widx][0] = jW
            jF_y.append(jF)

        # Now place the two even-parity MPOs on the jF indices (one per cycle)
        next_label = _append_even_parity_mpo_on_indices(tensors, connects, jF_x, next_label)
        next_label = _append_even_parity_mpo_on_indices(tensors, connects, jF_y, next_label)
    # ----------------------------------------------------------------------

    # Contract to get the unnormalized score N(s)
    N_s = ncon(tensors, connects)

    # Normalize by the appropriate Z (trace unchanged by TP noise; projector reduces sectors)
    Z = _closed_form_Z_contractible(Lx, Ly) if contractible_only else _closed_form_Z_all(Lx, Ly)
    return N_s / Z

if __name__ == "__main__":
    Lx, Ly, L = 4, 4, 4
    E = 2 * Lx * Ly

    for p in np.arange(0, 0.2, 0.01):
        measurements = pd.read_csv(
            f"../traindata/toricpq/d=2/measurements_L={L}_p={p:.3f}.csv",
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
            f"../traindata/toricpq/d=2/ps_L={L}_p={p:.3f}.csv",
            index=False,
            header=False,
        )

