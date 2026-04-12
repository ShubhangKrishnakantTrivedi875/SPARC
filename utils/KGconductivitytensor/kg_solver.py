"""
kg_solver.py
────────────
Core physics routines for the Kubo-Greenwood conductivity calculation.

Provides
────────
  build_gradient_info(params)
      Build k-independent sparse FD gradient ingredients (MSPARC translation).

  compute_kubo_greenwood(psi, header, eign, occ, kpts, kpt_wts,
                         I_indices, grad_info, params, Omega, eta)
      Return σ_{αβ}(ω) [Ha, atomic units], shape (3, 3, n_omega).

Formula
───────
  σ_{αβ}(ω) = (2π/V) Σ_k w_k Σ_{n,m}
                [(f_n−f_m)/(E_n−E_m)] M^α_{nm} [M^β_{nm}]*
                / (E_n−E_m − ω − iη)

  M^α_{nm}(k) = dV Σ_r ψ_n*(r,k) [∂_α ψ_m](r,k)   (FD gradient)

The gradient matrix uses the exact same central-difference stencil
as M-SPARC (gradIndicesValues.m + blochGradient.m, cell_typ = 1).

Authors: (your names)
"""

import time
import numpy as np
from scipy.sparse import csr_matrix


# ════════════════════════════════════════════════════════════════════════════
#  FD weights  (MSPARC fract() formula)
# ════════════════════════════════════════════════════════════════════════════

def _fd_weights_D1(FDn):
    """
    Central FD weights w1[1..FDn] for the first derivative.

    MSPARC formula (gradIndicesValues.m, lines for w1):
        w1[p] = (-1)^{p+1} × fract(FDn, p) / p

        fract(n, k) = [∏_{i=n-k+1}^{n} i] / [∏_{i=n+1}^{n+k} i]

    The stencil approximates  f'(x) ≈ Σ_{p=1}^{FDn} w1[p] (f(x+p·h) − f(x−p·h)) / h
    """
    w1 = np.zeros(FDn + 1)
    for p in range(1, FDn + 1):
        numerator   = float(np.prod(np.arange(FDn - p + 1, FDn + 1)))
        denominator = float(np.prod(np.arange(FDn + 1,     FDn + p + 1)))
        w1[p] = ((-1) ** (p + 1)) * (numerator / denominator) / p
    return w1


# ════════════════════════════════════════════════════════════════════════════
#  Gradient matrix ingredients  (MSPARC gradIndicesValues.m)
# ════════════════════════════════════════════════════════════════════════════

def build_gradient_info(params):
    """
    Build the k-independent COO arrays for the FD gradient operator in
    x, y, and z directions.

    Direct Python translation of M-SPARC's gradIndicesValues.m
    (S.cell_typ < 3 branch, i.e. orthogonal / non-cyclic cells).

    Index convention
    ─────────────────
    Grid points ordered x-fastest (Fortran column-major), matching SPARC:
        linear_index = ix + iy*Nx + iz*Nx*Ny     (0-based)

    Stencil entry for direction x, shift ±p at row (ix, iy, iz):
        value  = ±w1[p] / dx
        column = ((ix ± p) mod Nx) + iy*Nx + iz*Nx*Ny

    Boundary markers (periodic BC only)
    ─────────────────────────────────────
    isOutl[e] = True  → stencil crosses the LEFT  boundary  (phase e^{−ikL})
    isOutr[e] = True  → stencil crosses the RIGHT boundary  (phase e^{+ikL})

    Parameters
    ──────────
    params : dict  from file_reader.read_out_file

    Returns
    ───────
    dict with keys 'x', 'y', 'z'.  Each value is a dict:
        I      (nnz,)  int    row indices
        J      (nnz,)  int    wrapped column indices
        V      (nnz,)  float  stencil values ±w1[p]/h  (k-independent, real)
        isOutl (nnz,)  bool   left-boundary wrap markers
        isOutr (nnz,)  bool   right-boundary wrap markers
        BC     int            0 = periodic, 1 = Dirichlet
        L      float          lattice length [Bohr]  (for Bloch phase 2πkL)
        N      int            total grid points Nx×Ny×Nz
    """
    t0 = time.time()
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    N           = Nx * Ny * Nz
    FDn         = params['FDn']
    w1          = _fd_weights_D1(FDn)

    print(f"\n── build_gradient_info ─────────────────────────────────────")
    print(f"   Grid ({Nx}×{Ny}×{Nz}), N={N},  FDn={FDn}")

    # ── flattened grid index arrays (x varies fastest) ───────────────────
    # meshgrid with indexing='ij': ii loops over x, jj over y, kk over z
    ii_3d, jj_3d, kk_3d = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij'
    )
    ii  = ii_3d.ravel()    # (N,)  x-index of each grid point
    jj  = jj_3d.ravel()    # (N,)  y-index
    kk  = kk_3d.ravel()    # (N,)  z-index
    row = kk * (Nx * Ny) + jj * Nx + ii   # linear row index (N,)

    # helpers: linear column index given shifted coordinate in each direction
    def col_x(s): return kk * (Nx * Ny) + jj * Nx + s
    def col_y(s): return kk * (Nx * Ny) + s  * Nx + ii
    def col_z(s): return s  * (Nx * Ny) + jj * Nx + ii

    direction_cfg = {
        'x': (params['BCx'], params['dx'], Nx, params['Lx'], ii, col_x),
        'y': (params['BCy'], params['dy'], Ny, params['Ly'], jj, col_y),
        'z': (params['BCz'], params['dz'], Nz, params['Lz'], kk, col_z),
    }

    grad_info = {}

    for dir_name, (BC, h, Ndir, L, base_idx, make_col) in direction_cfg.items():
        h_inv = 1.0 / h
        I_lst, J_lst, V_lst       = [], [], []
        outl_lst, outr_lst        = [], []

        for p in range(1, FDn + 1):
            coeff = w1[p] * h_inv      # magnitude of stencil coefficient

            idx_p = base_idx + p       # raw shifted index (may go out of domain)
            idx_m = base_idx - p

            outr = (idx_p >= Ndir)     # crosses right boundary
            outl = (idx_m <  0)        # crosses left  boundary

            if BC == 1:
                # ── Dirichlet: drop out-of-domain stencil entries ──
                keep_p = ~outr
                I_lst.append(row[keep_p])
                J_lst.append(make_col(idx_p[keep_p]))
                V_lst.append(np.full(keep_p.sum(), +coeff))
                outl_lst.append(np.zeros(keep_p.sum(), dtype=bool))
                outr_lst.append(np.zeros(keep_p.sum(), dtype=bool))

                keep_m = ~outl
                I_lst.append(row[keep_m])
                J_lst.append(make_col(idx_m[keep_m]))
                V_lst.append(np.full(keep_m.sum(), -coeff))
                outl_lst.append(np.zeros(keep_m.sum(), dtype=bool))
                outr_lst.append(np.zeros(keep_m.sum(), dtype=bool))

            else:
                # ── Periodic: wrap indices, record boundary crossings ──
                idx_pw = idx_p % Ndir   # Python % is always ≥0 → same as MATLAB mod
                idx_mw = idx_m % Ndir

                I_lst.append(row);                J_lst.append(make_col(idx_pw))
                V_lst.append(np.full(N, +coeff)); outl_lst.append(np.zeros(N, bool))
                outr_lst.append(outr.copy())

                I_lst.append(row);                J_lst.append(make_col(idx_mw))
                V_lst.append(np.full(N, -coeff)); outl_lst.append(outl.copy())
                outr_lst.append(np.zeros(N, bool))

        G_I    = np.concatenate(I_lst)
        G_J    = np.concatenate(J_lst)
        G_V    = np.concatenate(V_lst).astype(np.float64)
        G_outl = np.concatenate(outl_lst)
        G_outr = np.concatenate(outr_lst)

        grad_info[dir_name] = dict(
            I=G_I, J=G_J, V=G_V,
            isOutl=G_outl, isOutr=G_outr,
            BC=BC, L=L, N=N,
        )
        print(f"   {dir_name}: nnz={len(G_I):>8d}  BC={BC}  L={L:.4f} Bohr")

    elapsed = time.time() - t0
    print(f"   Gradient info built in {elapsed:.3f}s")
    return grad_info


# ════════════════════════════════════════════════════════════════════════════
#  Bloch gradient matrix  (MSPARC blochGradient.m)
# ════════════════════════════════════════════════════════════════════════════

def _bloch_gradient_matrix(g, k_frac):
    """
    Assemble the k-dependent sparse gradient matrix for ONE Cartesian direction.

    Translation of M-SPARC blochGradient.m (cell_typ < 3 branch).

    Bloch boundary condition:
        ψ(r + L ê) = e^{+i k·L} ψ(r),   k·L = k_frac × 2π

    Stencil entries that wrap across boundaries pick up the phase:
        LEFT  boundary crossing → multiply by  e^{−i k_frac × 2π}
        RIGHT boundary crossing → multiply by  e^{+i k_frac × 2π}

    Parameters
    ──────────
    g       : dict  one entry from build_gradient_info (one direction)
    k_frac  : float fractional k-coordinate for this direction
                    (e.g. kpts[ki, 0] for x)

    Returns
    ───────
    sparse (N, N) complex matrix
    """
    V = g['V'].astype(complex)      # copy, promote to complex

    if g['BC'] == 0 and k_frac != 0.0:
        phase_l = np.exp(-1j * k_frac * 2.0 * np.pi)   # e^{-ikL}
        phase_r = np.exp(+1j * k_frac * 2.0 * np.pi)   # e^{+ikL}
        V[g['isOutl']] *= phase_l
        V[g['isOutr']] *= phase_r

    N = g['N']
    return csr_matrix((V, (g['I'], g['J'])), shape=(N, N))


# ════════════════════════════════════════════════════════════════════════════
#  Kubo-Greenwood conductivity tensor
# ════════════════════════════════════════════════════════════════════════════

def compute_kubo_greenwood(psi, header, eign, occ, kpts, kpt_wts,
                           I_indices, grad_info, params, Omega, eta):
    """
    Compute the frequency-dependent electrical conductivity tensor σ_{αβ}(ω).

    Algorithm (per k-point)
    ────────────────────────
    1.  Sort psi columns to match ascending eigenvalues (using I_indices).
    2.  Build k-dependent Bloch gradient matrices G_α(k)  (α ∈ {x,y,z}).
    3.  Momentum matrix elements (local part):
            M^α[n,m] = dV × ψ_n†(k) G_α(k) ψ_m(k)       shape (nband, nband)
    4.  Occupation-weighted matrix:
            A^α[n,m] = M^α[n,m] × (f_n − f_m) / (E_n − E_m)  (diagonal = 0)
    5.  Kubo-Greenwood sum:
            σ_{αβ}(ω) += w_k Σ_{n,m} A^α[n,m] [M^β[n,m]]* / (E_n−E_m−ω−iη)

    Global prefactor: 2π / V   [atomic units; factor of 2 for spin degeneracy]

    Parameters
    ──────────
    psi       : (Nd, nband, nkpt)  complex/real wavefunctions from read_psi_file
    header    : dict from read_psi_file
    eign      : (nband, nkpt)  [Ha]  sorted eigenvalues
    occ       : (nband, nkpt)  occupation numbers
    kpts      : (nkpt,  3)    fractional k-coordinates
    kpt_wts   : (nkpt,)       k-point weights
    I_indices : (nband, nkpt) argsort permutation (from read_eigen_file)
    grad_info : dict from build_gradient_info
    params    : dict from read_out_file
    Omega     : (n_omega,) [Ha] frequency array
    eta       : float [Ha] Lorentzian broadening

    Returns
    ───────
    sigma : (3, 3, n_omega)  complex128  conductivity tensor [a.u.]
    timing: dict  per-section timing info
    """
    t0      = time.time()
    nband   = header['nband']
    nkpt    = header['nkpt']
    dV      = params['dV']
    volume  = params['volume']
    n_omega = len(Omega)
    dirs    = ['x', 'y', 'z']

    print(f"\n── compute_kubo_greenwood ──────────────────────────────────")
    print(f"   nkpt={nkpt}  nband={nband}  n_omega={n_omega}  η={eta:.4e} Ha")
    print(f"   dV={dV:.4e} Bohr³  V={volume:.4f} Bohr³")

    # ── 1. Sort psi to match ascending eigenvalues ────────────────────────
    print(f"   Sorting psi columns to match eigenvalue order …")
    psi_sorted = np.zeros_like(psi)
    for k in range(nkpt):
        perm = I_indices[:, k].astype(int)
        psi_sorted[:, :, k] = psi[:, perm, k]
    print(f"   psi_sorted shape: {psi_sorted.shape}")

    sigma = np.zeros((3, 3, n_omega), dtype=complex)

    t_grad_total   = 0.0
    t_matel_total  = 0.0
    t_kgsum_total  = 0.0

    for ki in range(nkpt):
        t_k    = time.time()
        w_k    = float(kpt_wts[ki])
        psi_k  = psi_sorted[:, :, ki]      # (Nd, nband)  complex or real
        e_k    = eign[:, ki]               # (nband,)  Ha
        f_k    = occ[:, ki]                # (nband,)

        print(f"\n   k = {ki+1}/{nkpt}  kpt=({kpts[ki,0]:+.4f},{kpts[ki,1]:+.4f},"
              f"{kpts[ki,2]:+.4f})  w={w_k:.5f}")

        # ── Energy/occupation difference matrices  (nband × nband) ────────
        dE = e_k[:, None] - e_k[None, :]          # E_n − E_m
        df = f_k[:, None] - f_k[None, :]          # f_n − f_m

        # (f_n−f_m)/(E_n−E_m): diagonal is 0/0; adding eye avoids the 0/0
        # (diagonal of df is 0 so the diagonal of the ratio is always 0)
        df_over_dE = df / (dE + np.eye(nband))

        # ── 2. Bloch gradient matrices ────────────────────────────────────
        t_g = time.time()
        G = {}
        for ai, d in enumerate(dirs):
            G[d] = _bloch_gradient_matrix(grad_info[d], float(kpts[ki, ai]))
        t_grad_total += time.time() - t_g
        print(f"     Bloch gradient matrices assembled")

        # ── 3. Momentum matrix elements  M^α[n,m] = dV ψ_n† G_α ψ_m ─────
        t_m = time.time()
        M = {}
        for d in dirs:
            Gpsi = G[d] @ psi_k            # (Nd, nband)
            M[d] = dV * (psi_k.conj().T @ Gpsi)   # (nband, nband)  complex
        t_matel_total += time.time() - t_m
        print(f"     |M_x|² max = {np.max(np.abs(M['x'])**2):.4e}  "
              f"|M_y|² max = {np.max(np.abs(M['y'])**2):.4e}  "
              f"|M_z|² max = {np.max(np.abs(M['z'])**2):.4e}")

        # ── 4. Occupation-weighted matrix ─────────────────────────────────
        WM = {d: M[d] * df_over_dE for d in dirs}

        # ── 5. Kubo-Greenwood frequency sum ───────────────────────────────
        t_kg = time.time()
        for wi, omega in enumerate(Omega):
            denom = 1.0 / (dE - omega - 1j * eta)    # (nband, nband)
            for ai, da in enumerate(dirs):
                for bi, db in enumerate(dirs):
                    sigma[ai, bi, wi] += (
                        w_k * np.sum(WM[da] * M[db].conj() * denom)
                    )
        t_kgsum_total += time.time() - t_kg

        elapsed_k = time.time() - t_k
        print(f"     k-point done  ({elapsed_k:.2f}s)")

    # Global prefactor: 2π/V  (spin degeneracy ×2 included)
    sigma *= (2.0 * np.pi / volume)

    total_elapsed = time.time() - t0
    timing = dict(
        total       = total_elapsed,
        gradient    = t_grad_total,
        matrix_elem = t_matel_total,
        kg_sum      = t_kgsum_total,
    )

    print(f"\n   KG tensor computed  (total = {total_elapsed:.2f}s)")
    print(f"   Breakdown:  gradient={t_grad_total:.2f}s  "
          f"matrix_elem={t_matel_total:.2f}s  kg_sum={t_kgsum_total:.2f}s")
    return sigma, timing
