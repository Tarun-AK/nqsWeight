import copy
from functools import reduce

import numpy as np


def delta_ijkl(n):
    delta_ij = np.eye(n)
    delta_kl = np.eye(n)
    return np.einsum("ij,kl->ijkl", delta_ij, delta_kl)


def TN(syndromes, p):
    dijkl = delta_ijkl(2)
    L = syndromes.shape[-1]
    qubitConfs = []
    for syndrome in syndromes:
        errors = np.argwhere(syndrome == -1)
        qubits = np.ones((*(syndrome.shape), 2))
        assert errors.shape[0] % 2 == 0

        lefts = errors[: int(len(errors) / 2)]
        rights = errors[int(len(errors) / 2) :]

        for left, right in zip(lefts, rights):
            if right[0] - left[0] < 0:
                right, left = left, right

            print(qubits.shape)
            qubits[left[0] : right[0], left[1]][0] *= -1

            if right[1] - left[1] > 0:
                qubits[right[0], left[1] : right[1]][1] *= -1
            else:
                qubits[right[0], right[1] : left[1]][1] *= -1

        qubitConfs.append(qubits)

    qubitConfs1 = np.array(qubitConfs)
    qubitConfs2 = copy.deepcopy(qubitConfs1)
    qubitConfs2[:, int(L / 2), :, 0] *= -1
    qubitConfs3 = copy.deepcopy(qubitConfs1)
    qubitConfs3[:, :, int(L / 2), 1] *= -1
    qubitConfs4 = copy.deepcopy(qubitConfs2)
    qubitConfs4[:, :, int(L / 2), 1] *= -1

    exponent = (np.log((1 - p) / p)) * (1 + np.array([[1, -1], [-1, 1]])) / 2

    Zs = []
    for qubitC in [qubitConfs1, qubitConfs2, qubitConfs3, qubitConfs4]:
        Os = np.exp(np.einsum("blkf,ij->blkfij", qubitC, exponent))
        TMs = []
        for l in range(L):
            TMs.append(reduce(np.kron, Os[:, l, :, 0].transpose(1, 0, 2, 3)))
            row2 = np.einsum("bkij,jmno->bkimno", Os[:, l, :, 1], dijkl)

            def reduction(A, B):
                out = np.reshape(
                    np.einsum("bimno,bopqr->bimnpqr", A, B),
                    (A.shape[0], 2, 2 * A.shape[2], 2 * A.shape[2], 2),
                )
                return out

            TMs.append(
                np.einsum(
                    "bijki->bjk", reduce(reduction, row2.transpose(1, 0, 2, 3, 4, 5))
                )
            )

        Zs.append(
            np.einsum(
                "bii->b", reduce(lambda A, B: np.einsum("bij,bjk->bik", A, B), TMs)
            )
        )
    print(Zs)
    return p


if __name__ == "__main__":
    import itertools

    import netket as nk
    import tqdm

    L = 3
    hilbert = nk.hilbert.Spin(s=1 / 2, N=L * L)
    all_arrays = []
    ps = []
    for combo in tqdm.tqdm(hilbert.all_states()):
        array = np.array(combo).reshape((1, L, L))
        all_arrays.append(array)
        if len(np.argwhere(array == -1)) % 2 == 0:
            print(array)
            p = TN(array, 0.00001)
            ps.append(p)
        else:
            ps.append(0.0)

    ps = ps / np.sum(ps)
    for i in range(2 ** (L * L)):
        if ps[i] > 0.1:
            print(i, all_arrays[i])
