from collections import Counter, OrderedDict
from itertools import combinations, product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import quimb as qu
import quimb.tensor as qnt


def _dual_couplings_from_primal(H: nx.Graph, qubit_attr: str = "qubit") -> dict:
    coords = list(H.nodes())
    Lx = max(i for i, _ in coords) + 1
    Ly = max(j for _, j in coords) + 1
    assert Lx == Ly, "Assumes square LxL for simplicity."
    L = Lx

    def orient(u, v):
        return (u, v) if H.has_edge(u, v) else (v, u)

    j_dual = {}
    for i in range(L):
        for j in range(L):
            # right neighbor on dual: (i,j) -- (i+1,j)
            u = ((i + 1) % L, j)
            v = ((i + 1) % L, (j + 1) % L)  # primal vertical edge at column i+1
            e = orient(u, v)
            eta = H.edges[e][qubit_attr]
            j_dual[((i, j), ((i + 1) % L, j))] = eta  # sign only; scale later

            # up neighbor on dual: (i,j) -- (i,j+1)
            u = (i, (j + 1) % L)
            v = ((i + 1) % L, (j + 1) % L)  # primal horizontal edge at row j+1
            e = orient(u, v)
            eta = H.edges[e][qubit_attr]
            j_dual[((i, j), (i, (j + 1) % L))] = eta
    return j_dual


def assign_physical_qubits_all_sectors(
    G: nx.Graph, syndrome_attr: str = "syndrome", qubit_attr: str = "qubit"
) -> dict:

    H0 = assign_physical_qubits(G, syndrome_attr, qubit_attr)

    coords = list(G.nodes())
    Lx = max(i for i, _ in coords) + 1
    Ly = max(j for _, j in coords) + 1

    cycle_x = [((i, int(Lx / 2)), ((i + 1) % Lx, int(Lx / 2))) for i in range(Lx)]
    cycle_y = [((int(Lx / 2), j), (int(Lx / 2), (j + 1) % Ly)) for j in range(Ly)]

    def orient(u, v):
        return (u, v) if G.has_edge(u, v) else (v, u)

    cycle_x = [orient(u, v) for u, v in cycle_x]
    cycle_y = [orient(u, v) for u, v in cycle_y]

    sectors = {}
    for bx, by in product([0, 1], repeat=2):
        H = H0.copy()
        if bx:
            for u, v in cycle_x:
                H.edges[u, v][qubit_attr] *= -1
        if by:
            for u, v in cycle_y:
                H.edges[u, v][qubit_attr] *= -1
        sectors[(bx, by)] = H
    return sectors


def assign_physical_qubits(
    G: nx.Graph, syndrome_attr: str = "syndrome", qubit_attr: str = "qubit"
) -> nx.Graph:

    H = G.copy()

    odd_vs = [v for v, data in H.nodes(data=True) if data.get(syndrome_attr) == -1]
    error_edges = set()

    while odd_vs:
        u = odd_vs.pop(0)
        v = odd_vs.pop(0)
        path = nx.shortest_path(H, u, v)
        error_edges.update(zip(path, path[1:]))

    # print(error_edges)

    for u, v in H.edges():
        if (u, v) in error_edges or (v, u) in error_edges:
            H.edges[u, v][qubit_attr] = -1
        else:
            H.edges[u, v][qubit_attr] = +1

    return H


def TN(syndromes, p):
    L = syndromes.shape[0]
    G = nx.grid_2d_graph(L, L, periodic=True)

    for i in range(L):
        for j in range(L):
            G.nodes[(i, j)]["syndrome"] = syndromes[i, j]

    sectors = assign_physical_qubits_all_sectors(
        G,
    )
    J0 = 0.5 * np.log((1 - p) / p)
    out = []
    for bx, by in product([0, 1], repeat=2):
        H = sectors[(bx, by)]
        eta_dual = _dual_couplings_from_primal(H, qubit_attr="qubit")
        J = {edge: J0 * eta for edge, eta in eta_dual.items()}
        # J = {(u, v): J0 * data["qubit"] for u, v, data in H.edges(data=True)}

        TN = qnt.tensor_builder.TN2D_classical_ising_partition_function(
            L, L, beta=1, j=J, h=0, cyclic=True
        )
        Z_raw = TN ^ ...
        prefactor = 0.5 * (p * (1 - p)) ** (L * L)

        out.append(prefactor * Z_raw)
    return out


if __name__ == "__main__":

    import netket as nk
    import pandas as pd
    from tqdm import tqdm

    plt.style.use("~/plotStyle.mplstyle")
    colorDict = {4: "red", 5: "blue", 6: "green", 7: "yellow", 9: "orange", 11: "cyan"}
    for perr in np.arange(0.0, 0.2, 0.01):
        for L in [5]:
            hilbert = nk.hilbert.Spin(s=1 / 2, N=L * L)
            all_arrays = []
            ps = []
            measurements = pd.read_csv(
                f"traindata/toricpbc/d=2/measurements_L={L}_p={perr:.3f}.csv",
                delimiter=" ",
                header=None,
            )
            measurements = measurements.dropna(axis=1, how="all")
            measurements = measurements.to_numpy()
            metric = []
            for combo in tqdm(measurements):
                array = np.array(combo).reshape((L, L))
                all_arrays.append(array)
                if len(np.argwhere(array == -1)) % 2 == 0:
                    Zs = TN(array, perr)
                    # metric.append(np.log(Zs[0] / Zs[1]))
                    metric.append(np.sum(Zs))
            pd.DataFrame(metric).to_csv(
                f"traindata/toricpbc/d=2/ps_L={L}_p={perr:.3f}.csv",
                index=False,
                header=False,
            )
            # plt.scatter(
            #     perr,
            #     np.mean(metric),
            #     color=colorDict[L],
            #     edgecolor="black",
            #     label=f"L={L}",
            # )

    # plt.ylabel(r"$\expval{\log(\frac{Z_{RBIM}^{0,0}}{Z_{RBIM}^{0, 1}})}$")
    # plt.xlabel(r"$p_{err}$")
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), fontsize="small")
    # plt.show()
