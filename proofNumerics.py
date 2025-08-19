import netket as nk
import numpy as np

N = 4
J = 2.0
h = 1.0

g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

ha = nk.operator.LocalOperator(hi)

for i in range(N):
    j = (i + 1) % N
    ha += (
        -J * 0.5 * nk.operator.spin.sigmaz(hi, i) * 0.5 * nk.operator.spin.sigmaz(hi, j)
    )

for i in range(N):
    ha += -h * 0.5 * nk.operator.spin.sigmax(hi, i)

E0, psi0 = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

print("Ground state energy:", E0)
print("Ground state wavefunction:", psi0)

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
# Build full rotation as tensor product
Ux = H
for _ in range(N - 1):
    Ux = np.kron(Ux, H)

# Ground state in x-basis
psi0_x = Ux @ psi0
print("Ground state in x-basis:", psi0_x)
