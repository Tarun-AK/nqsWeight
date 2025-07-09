import csv

import numpy as np
import opt_einsum as oe


def build_edge_weights(p):
    """Return array w[s] = p**s*(1-p)**(1-s) for s=0,1."""
    return np.array([(1 - p), p], dtype=np.float64)


def transfer_matrix_row(Lx, p, a_row):
    """
    Build the transfer operator M for one horizontal strip of plaquettes at fixed y,
    conditioned on the anyon row a_row (length Lx of 0/1).
    M has indices: (h_left, h_right, h_left', h_right') where
      - h_left, h_right are incoming horizontal-edge bits on this row
      - h_left', h_right' are outgoing horizontal-edge bits for the next row
    We’ll represent M as a dense array of shape (2^(Lx+1), 2^(Lx+1)) by flattening the horizontal
    strings into integer labels.  This is O(2^{2(Lx+1)}) in memory, so it’s OK up to Lx~8–10.
    """
    w = build_edge_weights(p)
    # Precompute vertical-edge weights for each site
    # vertical edges appear twice (above & below), so one strip sees edges above and below
    # but we’ll absorb one factor per row; the boundary rows must be handled separately.
    size = 2 ** (Lx + 1)
    M = np.zeros((size, size), dtype=np.float64)
    for in_label in range(size):
        h_in = np.array(list(np.binary_repr(in_label, width=Lx + 1)), int)
        for out_label in range(size):
            h_out = np.array(list(np.binary_repr(out_label, width=Lx + 1)), int)
            weight = 1.0
            # weight from horizontal edges (incoming & outgoing)
            weight *= np.prod(w[h_in]) * np.prod(w[h_out])
            # now for each plaquette x compute parity constraint
            for x in range(Lx):
                s_sum = (
                    h_in[x]  # top-left
                    + h_in[x + 1]  # top-right
                    + h_out[x]  # bottom-left
                    + h_out[x + 1]  # bottom-right
                )
                # enforce parity = a_row[x]
                if (s_sum % 2) != a_row[x]:
                    weight = 0.0
                    break
            M[in_label, out_label] = weight
    return M


def compute_anyon_probability(anyon_config, p, normalize=True):
    """
    anyon_config: 2D array of shape (Ly, Lx) with 0/1 entries.
    p: edge-flip probability.
    normalize: if True, divides by the full Z(p) so you get a true probability.
    """
    anyon = np.asarray(anyon_config, dtype=int)
    Ly, Lx = anyon.shape

    # Build transfer matrices for each row
    Ts = [transfer_matrix_row(Lx, p, anyon[y]) for y in range(Ly)]

    # Initial boundary vector: there's no horizontal edges above row 0,
    # so we start with the weight of a single horizontal row of edges (size=2^(Lx+1)):
    w = build_edge_weights(p)
    size = 2 ** (Lx + 1)
    v = np.zeros(size, dtype=np.float64)
    for label in range(size):
        h = np.array(list(np.binary_repr(label, width=Lx + 1)), int)
        # this is the top boundary horizontal-edges weight
        v[label] = np.prod(w[h])
    # Propagate down through the Ly strips
    for T in Ts:
        v = T.dot(v)

    # After the last row, we must absorb the bottom horizontal boundary weights again
    # but since we included both h_in and h_out in each T, that is already done.
    weight = np.sum(v)

    if normalize:
        # To get Z(p), set all a_row to zero (no anyons) and sum the same way,
        # OR build transfer matrices with sum over a_row=0 only.
        zero_Ts = [transfer_matrix_row(Lx, p, np.zeros(Lx, int)) for _ in range(Ly)]
        v0 = np.copy(v)  # re-initialize
        v0 = np.zeros(size, dtype=np.float64)
        for label in range(size):
            h = np.array(list(np.binary_repr(label, width=Lx + 1)), int)
            v0[label] = np.prod(w[h])
        for T0 in zero_Ts:
            v0 = T0.dot(v0)
        Z = np.sum(v0)
        return weight / Z

    return weight


# Example usage:
if __name__ == "__main__":
    for i in range(1, 20):
        out_probs = []
        pfloat = 0.01 * i
        p = f"{pfloat:.3f}"
        # Read the CSV file
        matrix = np.genfromtxt(
            f"traindata/toric/d=2/measurements_L=6_p={p}.csv", delimiter=" "
        )
        for j in range(100000):
            samp = matrix[j, :]
            samp = samp.reshape(6, 6)
            samp = -(samp - 1) / 2
            prob = compute_anyon_probability(samp, pfloat)
            print(prob)
            out_probs.append(prob)
        # Write the output probabilities to a new CSV file
        with open(f"traindata/toric/d=2/ps_L=6_p={p}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for prob in out_probs:
                writer.writerow([prob])
