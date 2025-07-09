from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np

plt.style.use("~/plotStyle.mplstyle")


def fwht(f):
    """
    Perform the Walsh-Hadamard Transform on a Boolean function f.

    :param f: A list or numpy array of function outputs for all spin configurations.
    :return: The Walsh-Hadamard transformed array.
    """
    n = int(np.log2(len(f)))
    assert len(f) == 2**n, "Length of f must be a power of 2"

    h = np.array(f, dtype=float)
    for i in range(n):
        step = 2**i
        for j in range(0, len(f), 2 * step):
            for k in range(step):
                x = h[j + k]
                y = h[j + k + step]
                h[j + k] = x + y
                h[j + k + step] = x - y
    return h


def stability(logp, rho):
    N = int(np.log2(len(logp)))
    coeffs = np.array(fwht(logp.copy()))  # * (1 / 2**N)
    degrees = np.array([bin(i).count("1") for i in range(len(coeffs))])
    return np.dot(rho**degrees, coeffs**2)  # / np.sum(coeffs**2)


def max_fourier_walsh_coefficients(logp):
    N = int(np.log2(len(logp)))
    coeffs = np.array(fwht(logp.copy())) * (1 / 2**N)
    degrees = np.array([bin(i).count("1") for i in range(len(coeffs))])
    max_coeffs = {}
    for d in range(0, N + 1):
        max_coeffs[d] = np.nanmax(coeffs[degrees == d])  # / (
        # np.sqrt(float(np.sum(coeffs**2)))
        # )  # plt.hist(np.abs(coeffs)[degrees == d])
        # plt.title(d)
        # plt.show()
    return max_coeffs


@partial(jax.vmap, in_axes=(0, None, None))
def prob(sigmas, L, p):
    size = L + (L % 2)
    minuses = jnp.array(jnp.argwhere(sigmas == -1, size=size, fill_value=jnp.nan))[:, 0]
    result = minuses[1::2] - minuses[::2]
    l = jnp.nansum(result)
    term1 = (p**l) * ((1 - p) ** (L - l))
    term2 = (p ** (L - l)) * ((1 - p) ** l)

    return jax.lax.cond(
        jnp.sum(sigmas == -1) % 2 == 0,
        lambda _: term1 + term2,
        lambda _: 0.0,
        operand=None,
    )


L = 9
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

allStates = hilbert.all_states()
np.savetxt("traindata/toric/d=2/exactMs_L=3.csv", allStates, fmt="%d", delimiter=" ")
for p in np.linspace(0.0, 1.0, 20):
    allps = prob(allStates, L, p)
    if np.sum(allps) != 1:
        print(np.sum(allps))

    # logp = np.log(allps + 1e-20)

    max_coeffs = max_fourier_walsh_coefficients(allps)
    plt.scatter(
        p, stability(allps, 0.95), edgecolor="black", color="red", label=f"p={p}"
    )

plt.ylabel(r"$\mathrm{Stab}_{0.95}(\log{p})$")
plt.xlabel(r"$p_{err}$")
plt.title("Normalized")
plt.show()
