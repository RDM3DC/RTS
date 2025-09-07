
import numpy as np

# Pauli matrices
I2 = np.array([[1,0],[0,1]], dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
Sp = np.array([[0,1],[0,0]], dtype=complex)  # sigma+
Sm = np.array([[0,0],[1,0]], dtype=complex)  # sigma-

def kronN(ops):
    M = np.array([[1]], dtype=complex)
    for op in ops:
        M = np.kron(M, op)
    return M

def op_at(site, op, N):
    return kronN([op if i==site else I2 for i in range(N)])

def build_static_terms(N):
    # Precompute site ops
    Zs  = [op_at(i, Z,  N) for i in range(N)]
    Sps = [op_at(i, Sp, N) for i in range(N)]
    Sms = [op_at(i, Sm, N) for i in range(N)]
    # Link XY operators
    H_links = []
    for i in range(N-1):
        Hxy = Sps[i] @ Sms[i+1] + Sms[i] @ Sps[i+1]
        H_links.append(Hxy)
    return Zs, H_links

def hamiltonian(N, Zs, H_links, omegas, gs):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H += 0.5 * omegas[i] * Zs[i]
    for i in range(N-1):
        H += gs[i] * H_links[i]
    return H

def normalize(psi):
    return psi/np.linalg.norm(psi)

def evolve_step(psi, H, dt):
    # Hermitian H: use eigendecomp to exponentiate
    w, V = np.linalg.eigh(H)
    U = V @ np.diag(np.exp(-1j*w*dt)) @ V.conj().T
    return U @ psi

def link_current_expectation(psi, Sps, Sms):
    # I_link ~ |<σ+_i σ-_i+1>| + |<σ-_i σ+_i+1>|  (coherence magnitude surrogate)
    N = len(Sps)
    vals = []
    for i in range(N-1):
        a = (psi.conj().T @ (Sps[i] @ Sms[i+1]) @ psi).item()
        b = (psi.conj().T @ (Sms[i] @ Sps[i+1]) @ psi).item()
        vals.append(abs(a) + abs(b))
    return np.array(vals)

def populations(psi, N):
    # Return site excitation probabilities in the single-excitation subspace approx:
    # p_i = <(I+Z_i)/2>
    ps = []
    for i in range(N):
        Zi = op_at(i, Z, N)
        ps.append(0.5*(1 + (psi.conj().T @ Zi @ psi).real.item()))
    return np.array(ps)
