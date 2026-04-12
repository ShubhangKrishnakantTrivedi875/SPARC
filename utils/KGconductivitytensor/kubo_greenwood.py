#!/usr/bin/env python3
"""
kubo_greenwood.py
─────────────────
Frequency-dependent electrical conductivity tensor via the Kubo-Greenwood
formula from SPARC output files.  Restricted to orthogonal unit cells.

Formula (atomic units):
  σ_{αβ}(ω) = (2π/V) Σ_{k} w_k Σ_{n,m} [(f_n-f_m)/(E_n-E_m)]
               × M^α_{nm} conj(M^β_{nm}) / (E_n - E_m - ω - iη)

  M^α_{nm} = dV × Σ_r ψ_n*(r) [∂_α ψ_m](r)    (gradient via FD stencil)

The gradient stencil is a direct Python translation of MSPARC's
gradIndicesValues.m + blochGradient.m (cell_typ = 1, orthogonal cell).

Usage
─────
  python kubo_greenwood.py \\
      --out    Al.out    \\
      --psi    Al.psi    \\
      --eigen  Al.eigen  \\
      --outdir KG_output \\
      --omega_start 0.0  --omega_end 1.0  --n_omega 100 \\
      --eta 0.01

Author: (your name)
"""

import os
import re
import sys
import time
import argparse
import numpy as np
from scipy.sparse import csr_matrix

# ── locate calculate_pdos.py in the same directory ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from calculate_pdos import read_psi, PDOSCalculator


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Kubo-Greenwood conductivity from SPARC output files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out",         required=True,           help="SPARC .out  file path")
    p.add_argument("--psi",         required=True,           help="SPARC .psi  file path")
    p.add_argument("--eigen",       required=True,           help="SPARC .eigen file path")
    p.add_argument("--outdir",      default="KG_output",     help="Output directory")
    p.add_argument("--omega_start", type=float, default=0.0, help="Start frequency (Ha)")
    p.add_argument("--omega_end",   type=float, default=1.0, help="End   frequency (Ha)")
    p.add_argument("--n_omega",     type=int,   default=50,  help="Number of frequency points")
    p.add_argument("--eta",         type=float, default=0.01,help="Lorentzian broadening η (Ha)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1+2 ── Read .out file and validate
# ════════════════════════════════════════════════════════════════════════════

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"   # regex for any real number


def _last_numbers(fpath, keyword):
    """Return list[float] from the LAST line in fpath that contains keyword."""
    result = []
    with open(fpath) as f:
        for line in f:
            if keyword in line:
                result = [float(x) for x in re.findall(_NUM, line)]
    return result


def _numbers_after_keyword(fpath, keyword, n_lines=3):
    """Return flat list[float] from the n_lines following the LAST occurrence
    of keyword in fpath."""
    with open(fpath) as f:
        lines = f.readlines()
    result = []
    for idx, line in enumerate(lines):
        if keyword in line:
            block = []
            for i in range(1, n_lines + 1):
                if idx + i < len(lines):
                    block += [float(x) for x in re.findall(_NUM, lines[idx + i])]
            result = block       # keep overwriting → last occurrence wins
    return result


def read_out_file(fpath):
    """
    Parse SPARC .out file and return a params dict.

    Checks performed
    ─────────────────
      1. Lattice vectors computed from LATVEC × scale match the printed
         'Lattice vectors (Bohr)' block.
      2. Unit-cell volume is consistent.
      3. Cell is orthogonal (mandatory – raises ValueError otherwise).

    Returned dict keys (selection)
    ───────────────────────────────
      latvec, Lx/Ly/Lz, Nx/Ny/Nz, dx/dy/dz, dV, volume,
      FDn, FD_ORDER, BCx/BCy/BCz, nstates
    """
    t0 = time.time()
    print(f"\n[1/5]  Reading .out file: {fpath}")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"  .out file not found: {fpath}")

    # ── raw reads ────────────────────────────────────────────────────────
    latvec_scale = _last_numbers(fpath, "LATVEC_SCALE:")
    cell         = _last_numbers(fpath, "CELL:")
    fd_grid      = [int(x) for x in _last_numbers(fpath, "FD_GRID:")]
    fd_order     = int(_last_numbers(fpath, "FD_ORDER:")[0])
    bc           = [int(x) for x in _last_numbers(fpath, "BC:")]
    nstates      = int(_last_numbers(fpath, "NSTATES:")[0])

    # LATVEC rows are unit-direction vectors (normalised)
    latvec_dir_flat = _numbers_after_keyword(fpath, "LATVEC:", 3)
    if len(latvec_dir_flat) != 9:
        raise ValueError("Could not parse LATVEC from .out file (expected 9 numbers).")
    latvec_dir = np.array(latvec_dir_flat).reshape(3, 3)   # row i = direction of a_i

    # Cross-check quantities
    latvec_bohr_flat = _numbers_after_keyword(fpath, "Lattice vectors (Bohr):", 3)
    volume_file      = _last_numbers(fpath, "Volume")

    # ── cell scale ───────────────────────────────────────────────────────
    if latvec_scale:
        scale = np.array(latvec_scale)          # LATVEC_SCALE gives lengths directly
    elif cell:
        scale = np.array(cell)
    else:
        raise ValueError("Neither LATVEC_SCALE nor CELL found in .out file.")

    # Full lattice vectors: row i = scale_i * direction_i
    latvec = (latvec_dir.T * scale).T           # shape (3, 3)

    # ── CHECK 1: lattice vectors ──────────────────────────────────────────
    if len(latvec_bohr_flat) == 9:
        latvec_bohr = np.array(latvec_bohr_flat).reshape(3, 3)
        diff = np.max(np.abs(latvec - latvec_bohr))
        if diff > 1e-4:
            raise ValueError(
                f"Lattice-vector mismatch (max |Δ| = {diff:.2e}). "
                "Check LATVEC × LATVEC_SCALE vs 'Lattice vectors (Bohr)' in .out."
            )

    # ── CHECK 2: volume ────────────────────────────────────────────────────
    vol = float(abs(np.linalg.det(latvec)))
    if volume_file:
        vf = volume_file[0]
        if abs(vol - vf) / vf > 1e-4:
            raise ValueError(
                f"Volume mismatch: computed {vol:.6f}, file {vf:.6f} Bohr³."
            )

    # ── CHECK 3: orthogonality (mandatory) ────────────────────────────────
    metric  = latvec @ latvec.T          # g_{ij} = a_i · a_j
    off_dia = [metric[0,1], metric[0,2], metric[1,2]]
    if any(abs(v) > 1e-6 for v in off_dia):
        raise ValueError(
            "Non-orthogonal cell detected "
            f"(off-diagonal metric elements: {off_dia}).\n"
            "This code supports orthogonal cells only."
        )

    # ── Derived quantities ────────────────────────────────────────────────
    Lx = float(np.sqrt(metric[0, 0]))
    Ly = float(np.sqrt(metric[1, 1]))
    Lz = float(np.sqrt(metric[2, 2]))

    BCx, BCy, BCz = bc[0], bc[1], bc[2]        # 0 = periodic, 1 = Dirichlet

    # Number of grid points (Dirichlet adds one node per direction)
    Nx = fd_grid[0] + BCx
    Ny = fd_grid[1] + BCy
    Nz = fd_grid[2] + BCz

    # Mesh sizes (distance between adjacent nodes)
    dx = Lx / fd_grid[0]
    dy = Ly / fd_grid[1]
    dz = Lz / fd_grid[2]
    dV = dx * dy * dz

    params = dict(
        latvec=latvec,
        Lx=Lx, Ly=Ly, Lz=Lz,
        Nx=Nx, Ny=Ny, Nz=Nz, N=Nx*Ny*Nz,
        dx=dx, dy=dy, dz=dz, dV=dV,
        volume=vol,
        FD_ORDER=fd_order, FDn=fd_order // 2,
        BCx=BCx, BCy=BCy, BCz=BCz,
        nstates=nstates,
    )

    print(f"       Lattice  : {Lx:.4f} × {Ly:.4f} × {Lz:.4f}  Bohr")
    print(f"       Grid     : {Nx} × {Ny} × {Nz}   (FD_ORDER = {fd_order})")
    print(f"       BC       : ({BCx}, {BCy}, {BCz})   0 = periodic, 1 = Dirichlet")
    print(f"       dV       : {dV:.4e} Bohr³    Volume = {vol:.4f} Bohr³")
    print(f"       NSTATES  : {nstates}")
    print(f"       ✓  .out file OK  ({time.time()-t0:.2f}s)")
    return params


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 ── Read .psi file (wavefunctions)
# ════════════════════════════════════════════════════════════════════════════

def read_psi_file(fpath, params):
    """
    Parse binary SPARC .psi file.

    Returns
    ───────
    psi     : ndarray  shape (Nd, nband, nkpt), complex128 or float64
    header  : dict     keys: Nx,Ny,Nz,Nd,dx,dy,dz,dV,Nspinor_eig,
                             isGamma,nspin,nkpt,nband
    """
    t0 = time.time()
    print(f"\n[2/5]  Reading .psi file : {fpath}")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"  .psi file not found: {fpath}")

    psi, header, _ = read_psi(fpath, verbose=False)
    # psi.shape = (Nd * Nspinor_eig, nband, nkpt)

    # Grid-size consistency check
    for key in ("Nx", "Ny", "Nz"):
        if params[key] != header[key]:
            raise ValueError(
                f"  Grid mismatch in {key}: .out = {params[key]}, "
                f".psi header = {header[key]}"
            )

    print(f"       shape     : {psi.shape}  "
          f"(Nd={header['Nd']}, nband={header['nband']}, nkpt={header['nkpt']})")
    print(f"       isGamma   : {header['isGamma']}   "
          f"Nspinor_eig : {header['Nspinor_eig']}")
    print(f"       ✓  .psi file OK  ({time.time()-t0:.2f}s)")
    return psi, header


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 ── Read .eigen file (eigenvalues, occupations, k-points)
# ════════════════════════════════════════════════════════════════════════════

def read_eigen_file(fpath, nband, nkpt):
    """
    Parse SPARC .eigen file using PDOSCalculator's reader (no full init needed).

    Returns
    ───────
    eign      : (nband, nkpt)  float64  Ha, ascending within each k
    occ       : (nband, nkpt)  float64  occupation numbers
    kpts      : (nkpt,  3)     float64  fractional k-coordinates
    kpt_wts   : (nkpt,)        float64  k-point weights (sum ≈ 1)
    I_indices : (nband, nkpt)  int      argsort permutation used to sort bands
    """
    t0 = time.time()
    print(f"\n[3/5]  Reading .eigen file : {fpath}")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"  .eigen file not found: {fpath}")

    # Minimal stub – bypasses PDOSCalculator.__init__
    stub          = object.__new__(PDOSCalculator)
    stub.band_num = nband
    stub.kpt_num  = nkpt
    stub.Ha2eV    = 27.21140795
    stub.read_sparc_eigen_file_parameters(fpath)

    eign      = stub.eign           # (nband, nkpt)  Ha
    occ       = stub.occ            # (nband, nkpt)
    kpts      = stub.kpts_store     # (nkpt,  3)  fractional
    kpt_wts   = stub.kpt_wts_store  # (nkpt,)
    I_indices = stub.I_indices      # (nband, nkpt)  argsort indices

    Ha2eV = 27.21140795
    print(f"       nkpt = {nkpt},  nband = {nband}")
    print(f"       k-weight sum = {kpt_wts.sum():.6f}")
    print(f"       E range : [{eign.min()*Ha2eV:.3f}, {eign.max()*Ha2eV:.3f}] eV")
    for ki in range(nkpt):
        k3 = kpts[ki]
        print(f"         k#{ki+1:>3d} = ({k3[0]:+.4f}, {k3[1]:+.4f}, {k3[2]:+.4f})"
              f"  w = {kpt_wts[ki]:.6f}")
    print(f"       ✓  .eigen file OK  ({time.time()-t0:.2f}s)")
    return eign, occ, kpts, kpt_wts, I_indices


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 ── Finite-difference gradient (translated from MSPARC)
# ════════════════════════════════════════════════════════════════════════════

def _fd_weights_D1(FDn):
    """
    First-order central FD weights w1[1..FDn].

    Formula (identical to MSPARC):
        w1[p] = (-1)^{p+1} × fract(FDn, p) / p
        fract(n, k) = ∏_{i=n-k+1}^{n} i  /  ∏_{i=n+1}^{n+k} i
    """
    w1 = np.zeros(FDn + 1)
    for p in range(1, FDn + 1):
        nr = float(np.prod(np.arange(FDn - p + 1, FDn + 1)))
        dr = float(np.prod(np.arange(FDn + 1,     FDn + p + 1)))
        w1[p] = ((-1)**(p + 1)) * (nr / dr) / p
    return w1


def build_gradient_info(params):
    """
    Build the k-independent sparse-matrix ingredients for the gradient
    operator in x, y, z directions.

    Translation of MSPARC's gradIndicesValues.m (S.cell_typ < 3 branch).
    All indices are 0-based (MATLAB's 1-based are shifted by –1 throughout).

    For PERIODIC directions:
      • Column indices are wrapped: II_wrapped = II_raw % Ndir
      • isOutl[e] = True when stencil crosses the LEFT  boundary (→ Bloch phase e^{-ikL})
      • isOutr[e] = True when stencil crosses the RIGHT boundary (→ Bloch phase e^{+ikL})

    For DIRICHLET directions:
      • Out-of-domain entries are simply dropped (no Bloch phase).

    Returns
    ───────
    dict with keys 'x', 'y', 'z', each holding:
        I      (nnz,) int    row indices
        J      (nnz,) int    column indices (wrapped for periodic)
        V      (nnz,) float  stencil values  ±w1[p]/h  (real, phase-free)
        isOutl (nnz,) bool   left-boundary wrap markers
        isOutr (nnz,) bool   right-boundary wrap markers
        BC     int            0 = periodic, 1 = Dirichlet
        L      float          lattice length [Bohr] (needed for Bloch phase)
        N      int            total grid points Nx×Ny×Nz
    """
    t0 = time.time()
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    N   = Nx * Ny * Nz
    FDn = params['FDn']
    w1  = _fd_weights_D1(FDn)

    print(f"\n[4/5]  Building gradient matrices (FDn = {FDn})")

    # ── Grid coordinates (0-based, x varies fastest — SPARC Fortran order) ──
    ii_3d, jj_3d, kk_3d = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij'
    )
    ii  = ii_3d.ravel()                             # (N,)
    jj  = jj_3d.ravel()
    kk  = kk_3d.ravel()
    row = kk * (Nx * Ny) + jj * Nx + ii            # linear row index

    # Helper: linear column index given the (possibly wrapped) shifted coordinate
    # One closure per direction, capturing the fixed ii/jj/kk arrays.
    def col_x(shifted_ii): return kk * (Nx * Ny) + jj * Nx + shifted_ii
    def col_y(shifted_jj): return kk * (Nx * Ny) + shifted_jj * Nx + ii
    def col_z(shifted_kk): return shifted_kk * (Nx * Ny) + jj * Nx + ii

    direction_cfg = {
        'x': (params['BCx'], params['dx'], Nx, params['Lx'], ii,  col_x),
        'y': (params['BCy'], params['dy'], Ny, params['Ly'], jj,  col_y),
        'z': (params['BCz'], params['dz'], Nz, params['Lz'], kk,  col_z),
    }

    grad_info = {}

    for dir_name, (BC, h, Ndir, L, base_idx, make_col) in direction_cfg.items():
        h_inv = 1.0 / h
        I_lst, J_lst, V_lst = [], [], []
        outl_lst, outr_lst  = [], []

        for p in range(1, FDn + 1):
            coeff    = w1[p] * h_inv            # ±w1[p]/h

            idx_p    = base_idx + p             # raw positive-shifted index (N,)
            idx_m    = base_idx - p             # raw negative-shifted index (N,)

            outr     = idx_p >= Ndir            # exits right boundary
            outl     = idx_m <  0               # exits left  boundary

            if BC == 1:                         # ── Dirichlet: discard ghost nodes ──
                # positive stencil
                keep = ~outr
                I_lst.append(row[keep])
                J_lst.append(make_col(idx_p)[keep])
                V_lst.append(np.full(keep.sum(), +coeff))
                outl_lst.append(np.zeros(keep.sum(), dtype=bool))
                outr_lst.append(np.zeros(keep.sum(), dtype=bool))
                # negative stencil
                keep = ~outl
                I_lst.append(row[keep])
                J_lst.append(make_col(idx_m)[keep])
                V_lst.append(np.full(keep.sum(), -coeff))
                outl_lst.append(np.zeros(keep.sum(), dtype=bool))
                outr_lst.append(np.zeros(keep.sum(), dtype=bool))

            else:                               # ── Periodic: wrap + mark boundaries ──
                # Python's % is always ≥ 0, equivalent to MATLAB's mod(idx+(N-1),N)+1 − 1
                idx_p_w = idx_p % Ndir
                idx_m_w = idx_m % Ndir

                # positive stencil (+p)
                I_lst.append(row)
                J_lst.append(make_col(idx_p_w))
                V_lst.append(np.full(N, +coeff))
                outl_lst.append(np.zeros(N, dtype=bool))   # +shift only exits RIGHT
                outr_lst.append(outr.copy())

                # negative stencil (−p)
                I_lst.append(row)
                J_lst.append(make_col(idx_m_w))
                V_lst.append(np.full(N, -coeff))
                outl_lst.append(outl.copy())               # −shift only exits LEFT
                outr_lst.append(np.zeros(N, dtype=bool))

        G_I    = np.concatenate(I_lst)
        G_J    = np.concatenate(J_lst)
        G_V    = np.concatenate(V_lst).astype(float)
        G_outl = np.concatenate(outl_lst)
        G_outr = np.concatenate(outr_lst)

        grad_info[dir_name] = dict(
            I=G_I, J=G_J, V=G_V,
            isOutl=G_outl, isOutr=G_outr,
            BC=BC, L=L, N=N,
        )
        print(f"       {dir_name}-direction : nnz = {len(G_I):>8d}  "
              f"BC = {BC}  L = {L:.4f} Bohr")

    print(f"       ✓  gradient matrices built  ({time.time()-t0:.2f}s)")
    return grad_info


def _bloch_gradient_matrix(g, k_frac):
    """
    Return the k-dependent sparse gradient matrix for ONE direction.

    Translation of MSPARC's blochGradient.m.

    Parameters
    ──────────
    g      : one entry of the dict from build_gradient_info
    k_frac : fractional k-point coordinate for THIS direction (scalar)
             e.g. kpts[ki, 0] for the x-direction

    Physics
    ───────
    Bloch BC:  ψ(r + L ê) = e^{+i k L} ψ(r)  ⟹
      • Terms wrapping across the LEFT  boundary see ψ at r–L → multiply by e^{−i k L}
      • Terms wrapping across the RIGHT boundary see ψ at r+L → multiply by e^{+i k L}
    With k_frac = k/(2π/L): k·L = k_frac × 2π.
    """
    V = g['V'].astype(complex)                   # copy

    if g['BC'] == 0 and k_frac != 0.0:           # periodic + non-zero k
        phase_l = np.exp(-1j * k_frac * 2.0 * np.pi)   # left  boundary
        phase_r = np.exp(+1j * k_frac * 2.0 * np.pi)   # right boundary
        V[g['isOutl']] *= phase_l
        V[g['isOutr']] *= phase_r

    N = g['N']
    return csr_matrix((V, (g['I'], g['J'])), shape=(N, N))


# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 ── Kubo-Greenwood conductivity tensor
# ════════════════════════════════════════════════════════════════════════════

def compute_kubo_greenwood(psi, header, eign, occ, kpts, kpt_wts, I_indices,
                           grad_info, params, Omega, eta):
    """
    Compute σ_{αβ}(ω) for α,β ∈ {x,y,z} at each frequency in Omega.

    Algorithm (per k-point)
    ───────────────────────
    1. Build Bloch gradient matrices G_α(k).
    2. Momentum matrix:  M^α[n,m] = dV × ψ_n†(k) · G_α(k) · ψ_m(k)
    3. Weighted matrix:  A^α[n,m] = M^α[n,m] × (f_n−f_m)/(E_n−E_m)  (diagonal = 0)
    4. σ_{αβ}(ω) += w_k × Σ_{n,m} A^α[n,m] × conj(M^β[n,m]) / (E_n−E_m−ω−iη)

    Global prefactor: 2π/V  (atomic units; factor of 2 for spin degeneracy).

    Returns
    ───────
    sigma : ndarray  shape (3, 3, n_omega)  complex128
    """
    t0      = time.time()
    nband   = header['nband']
    nkpt    = header['nkpt']
    dV      = params['dV']
    volume  = params['volume']
    n_omega = len(Omega)
    dirs    = ['x', 'y', 'z']

    print(f"\n[5/5]  Computing Kubo-Greenwood conductivity tensor")
    print(f"       nkpt={nkpt}  nband={nband}  n_omega={n_omega}  η={eta:.4f} Ha")

    # Sort wavefunctions to match ascending eigenvalues
    psi_sorted = np.zeros_like(psi)                     # (Nd, nband, nkpt)
    for k in range(nkpt):
        perm = I_indices[:, k].astype(int)
        psi_sorted[:, :, k] = psi[:, perm, k]          # psi_sorted[:,n,k] ↔ eign[n,k]

    sigma      = np.zeros((3, 3, n_omega), dtype=complex)
    t_kpt_sum  = 0.0

    for ki in range(nkpt):
        t_k   = time.time()
        w_k   = float(kpt_wts[ki])
        psi_k = psi_sorted[:, :, ki]                   # (Nd, nband)
        e_k   = eign[:, ki]                             # (nband,)  Ha
        f_k   = occ[:, ki]                              # (nband,)

        # Energy and occupation difference matrices  (nband, nband)
        dE         = e_k[:, None] - e_k[None, :]       # E_n − E_m
        df         = f_k[:, None] - f_k[None, :]       # f_n − f_m

        # (f_n−f_m)/(E_n−E_m): add eye to avoid 0/0 on diagonal (diagonal = 0 anyway)
        df_over_dE = df / (dE + np.eye(nband))

        # Bloch gradient matrices for this k-point
        G = {d: _bloch_gradient_matrix(grad_info[d], float(kpts[ki, ai]))
             for ai, d in enumerate(dirs)}              # each (Nd, Nd) sparse

        # Momentum matrix elements  M^α[n,m] = dV × ψ_n†(k) G_α(k) ψ_m(k)
        # (the factor −i of p̂ = −i∂ is absorbed into the prefactor below)
        M = {}
        for d in dirs:
            Gpsi   = G[d] @ psi_k                      # (Nd, nband)
            M[d]   = dV * (psi_k.conj().T @ Gpsi)      # (nband, nband)

        # Weighted momentum matrix  A^α[n,m] = M^α[n,m] × (f_n−f_m)/(E_n−E_m)
        WM = {d: M[d] * df_over_dE for d in dirs}

        # Accumulate over frequencies
        for wi, omega in enumerate(Omega):
            # Lorentzian resolvent  (nband, nband)
            denom = 1.0 / (dE - omega - 1j * eta)

            for ai, da in enumerate(dirs):
                for bi, db in enumerate(dirs):
                    # σ_{αβ} += w_k Σ_{n,m} A^α_{nm} conj(M^β_{nm}) / (E_n−E_m−ω−iη)
                    sigma[ai, bi, wi] += w_k * np.sum(WM[da] * M[db].conj() * denom)

        t_kpt_sum += time.time() - t_k
        avg = t_kpt_sum / (ki + 1)
        eta_str = (f"ETA {avg*(nkpt-ki-1):.1f}s" if ki < nkpt - 1 else "done")
        print(f"         k = {ki+1:>3d}/{nkpt}  {eta_str}", end='\r', flush=True)

    print()  # newline after progress bar

    # Global prefactor  2π/V  (atomic units; spin degeneracy ×2 included)
    sigma *= (2.0 * np.pi / volume)

    print(f"       ✓  KG tensor computed  ({time.time()-t0:.2f}s)")
    return sigma


# ════════════════════════════════════════════════════════════════════════════
#  Output
# ════════════════════════════════════════════════════════════════════════════

def save_results(sigma, Omega, out_dir):
    """Save σ_{αβ}(ω) as plain-text files (one per tensor component)."""
    os.makedirs(out_dir, exist_ok=True)
    Ha2eV = 27.21140795

    components = [
        (0,0,'xx'), (0,1,'xy'), (0,2,'xz'),
        (1,1,'yy'), (1,2,'yz'),
        (2,2,'zz'),
    ]
    header_line = "# omega_Ha   omega_eV   Re(sigma)   Im(sigma)   [atomic units]"

    for (ai, bi, lab) in components:
        fname = os.path.join(out_dir, f"sigma_{lab}.dat")
        data  = np.column_stack([
            Omega,
            Omega * Ha2eV,
            sigma[ai, bi, :].real,
            sigma[ai, bi, :].imag,
        ])
        np.savetxt(fname, data, header=header_line, fmt="%.10e", comments='')

    # Isotropic average  σ_avg = (σ_xx + σ_yy + σ_zz) / 3
    sigma_avg = (sigma[0,0] + sigma[1,1] + sigma[2,2]) / 3.0
    fname = os.path.join(out_dir, "sigma_avg.dat")
    data  = np.column_stack([
        Omega, Omega*Ha2eV, sigma_avg.real, sigma_avg.imag,
    ])
    np.savetxt(fname, data,
               header="# omega_Ha   omega_eV   Re(sigma_avg)   Im(sigma_avg)",
               fmt="%.10e", comments='')

    print(f"\n  Results written to  {out_dir}/")
    print(f"  Files: sigma_xx/xy/xz/yy/yz/zz.dat  +  sigma_avg.dat")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 65)
    print("  Kubo-Greenwood Electrical Conductivity")
    print("  (SPARC real-space, orthogonal cell, MSPARC FD gradient)")
    print("=" * 65)
    print(f"  .out   : {args.out}")
    print(f"  .psi   : {args.psi}")
    print(f"  .eigen : {args.eigen}")
    print(f"  outdir : {args.outdir}")
    print(f"  ω range: {args.omega_start:.4f} → {args.omega_end:.4f} Ha"
          f"  ({args.n_omega} pts)")
    print(f"  η      : {args.eta:.4f} Ha")

    Omega = np.linspace(args.omega_start, args.omega_end, args.n_omega)

    # ── read ──────────────────────────────────────────────────────────────
    params          = read_out_file(args.out)
    psi, header     = read_psi_file(args.psi, params)
    eign, occ, kpts, kpt_wts, I_indices = read_eigen_file(
        args.eigen, nband=header['nband'], nkpt=header['nkpt']
    )

    # ── gradient matrices (k-independent part) ────────────────────────────
    grad_info = build_gradient_info(params)

    # ── Kubo-Greenwood ─────────────────────────────────────────────────────
    sigma = compute_kubo_greenwood(
        psi, header,
        eign, occ, kpts, kpt_wts, I_indices,
        grad_info, params, Omega, eta=args.eta,
    )

    # ── save ───────────────────────────────────────────────────────────────
    save_results(sigma, Omega, args.outdir)
    print("\n  All done.\n")


if __name__ == "__main__":
    main()
