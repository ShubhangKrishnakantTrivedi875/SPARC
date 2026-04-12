"""
file_reader.py
──────────────
Standalone readers for SPARC post-processing files.

Provides
────────
  read_out_file(fpath)
      → params dict  (lattice, grid, BC, FD, …)

  read_psi_file(fpath, params)
      → psi ndarray (Nd, nband, nkpt), header dict

  read_eigen_file(fpath, nband, nkpt)
      → eign, occ, kpts, kpt_wts, I_indices

All functions are fully self-contained; no dependency on calculate_pdos.py
or any SPARC Python package.

Authors: (your names)
"""

import os
import re
import time
import struct
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ════════════════════════════════════════════════════════════════════════════

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"   # matches any real number


def _last_numbers_on_keyword_line(fpath, keyword):
    """Return list[float] from the LAST line in fpath that contains keyword."""
    result = []
    with open(fpath) as fh:
        for line in fh:
            if keyword in line:
                result = [float(x) for x in re.findall(_NUM, line)]
    return result


def _numbers_below_keyword(fpath, keyword, n_lines=3):
    """
    Return flat list[float] from the n_lines that follow the LAST occurrence
    of keyword in fpath.
    """
    with open(fpath) as fh:
        lines = fh.readlines()
    result = []
    for idx, line in enumerate(lines):
        if keyword in line:
            block = []
            for i in range(1, n_lines + 1):
                if idx + i < len(lines):
                    block += [float(x) for x in re.findall(_NUM, lines[idx + i])]
            result = block     # overwrite → last occurrence wins
    return result


# ════════════════════════════════════════════════════════════════════════════
#  .out reader
# ════════════════════════════════════════════════════════════════════════════

def read_out_file(fpath):
    """
    Parse a SPARC .out file and return a params dict.

    Extracts
    ─────────
      LATVEC_SCALE / CELL, LATVEC, FD_GRID, FD_ORDER, BC, KPOINT_GRID,
      KPOINT_SHIFT, SPIN_TYP, NSTATES, 'Lattice vectors (Bohr)', Volume.

    Validation checks (raises ValueError on failure)
    ──────────────────────────────────────────────────
      1. Lattice vectors computed from direction × scale match the
         printed 'Lattice vectors (Bohr)' block.
      2. Unit-cell volume matches the determinant.
      3. Cell is orthogonal (mandatory for KG).

    Returns
    ────────
    dict with keys:
        latvec        (3,3) row = lattice vector a_i [Bohr]
        Lx, Ly, Lz   float  [Bohr]
        Nx, Ny, Nz   int    grid points (including Dirichlet boundary node)
        dx, dy, dz   float  mesh spacing [Bohr]
        dV           float  volume element [Bohr³]
        volume       float  unit-cell volume [Bohr³]
        FD_ORDER     int
        FDn          int    = FD_ORDER // 2
        BCx, BCy, BCz  int  0 = periodic, 1 = Dirichlet
        kpt_grid     [3]   int   k-point mesh
        kpt_shift    [3]   float k-point shift
        spin_typ     int
        nstates      int
        num_intervals [3]  int   raw FD_GRID values (no BC correction)
    """
    t0 = time.time()
    print(f"\n── read_out_file ──────────────────────────────────────────")
    print(f"   file : {fpath}")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"   .out file not found: {fpath}")

    # ── raw reads ────────────────────────────────────────────────────────
    latvec_scale = _last_numbers_on_keyword_line(fpath, "LATVEC_SCALE:")
    cell         = _last_numbers_on_keyword_line(fpath, "CELL:")
    fd_grid      = [int(x) for x in _last_numbers_on_keyword_line(fpath, "FD_GRID:")]
    fd_order_raw = _last_numbers_on_keyword_line(fpath, "FD_ORDER:")
    bc_raw       = _last_numbers_on_keyword_line(fpath, "BC:")
    kpt_grid_raw = _last_numbers_on_keyword_line(fpath, "KPOINT_GRID:")
    kpt_shift_raw= _last_numbers_on_keyword_line(fpath, "KPOINT_SHIFT:")
    spin_typ_raw = _last_numbers_on_keyword_line(fpath, "SPIN_TYP:")
    nstates_raw  = _last_numbers_on_keyword_line(fpath, "NSTATES:")

    # Lattice vector directions (normalised row vectors)
    latvec_dir_flat = _numbers_below_keyword(fpath, "LATVEC:", 3)
    # Printed Lattice vectors (Bohr) for cross-check
    latvec_bohr_flat = _numbers_below_keyword(fpath, "Lattice vectors (Bohr):", 3)
    volume_file = _last_numbers_on_keyword_line(fpath, "Volume")

    # ── basic parsing ────────────────────────────────────────────────────
    if len(latvec_dir_flat) != 9:
        raise ValueError("Could not parse LATVEC from .out file (expected 9 numbers).")
    latvec_dir = np.array(latvec_dir_flat).reshape(3, 3)   # row i = unit direction of a_i

    if latvec_scale:
        scale = np.array(latvec_scale)
    elif cell:
        scale = np.array(cell)
    else:
        raise ValueError("Neither LATVEC_SCALE nor CELL found in .out file.")

    # Full lattice vectors: row i = scale_i × direction_i
    latvec = (latvec_dir.T * scale).T      # shape (3, 3)

    print(f"   Lattice vectors [Bohr]:")
    for i, (row, s) in enumerate(zip(latvec, ["a₁","a₂","a₃"])):
        print(f"     {s} = [{row[0]:+12.6f}  {row[1]:+12.6f}  {row[2]:+12.6f}]")

    # ── CHECK 1: lattice vectors ──────────────────────────────────────────
    if len(latvec_bohr_flat) == 9:
        latvec_bohr = np.array(latvec_bohr_flat).reshape(3, 3)
        diff = np.max(np.abs(latvec - latvec_bohr))
        if diff > 1e-4:
            raise ValueError(
                f"Lattice-vector mismatch (max |Δ| = {diff:.2e}).\n"
                "  Check: LATVEC × LATVEC_SCALE  vs  'Lattice vectors (Bohr)'."
            )
        print(f"   ✓  Lattice vectors self-consistent  (max |Δ| = {diff:.2e})")
    else:
        print("   ⚠  'Lattice vectors (Bohr)' block not found – skipping cross-check.")

    # ── CHECK 2: volume ────────────────────────────────────────────────────
    vol = float(abs(np.linalg.det(latvec)))
    if volume_file:
        vf = volume_file[0]
        rel = abs(vol - vf) / max(vf, 1e-30)
        if rel > 1e-4:
            raise ValueError(
                f"Volume mismatch: computed {vol:.6f}, file {vf:.6f} Bohr³ "
                f"(rel. diff = {rel:.2e})."
            )
        print(f"   ✓  Volume = {vol:.4f} Bohr³  (file: {vf:.4f})")
    else:
        print(f"   ✓  Volume = {vol:.4f} Bohr³  (computed from det(latvec))")

    # ── CHECK 3: orthogonality ────────────────────────────────────────────
    metric   = latvec @ latvec.T
    off_diag = [metric[0,1], metric[0,2], metric[1,2]]
    if any(abs(v) > 1e-6 for v in off_diag):
        raise ValueError(
            "Non-orthogonal cell detected "
            f"(off-diagonal metric elements: {[f'{v:.3e}' for v in off_diag]}).\n"
            "  This code supports orthogonal cells only."
        )
    print(f"   ✓  Orthogonal cell confirmed")

    # ── Derived grid quantities ───────────────────────────────────────────
    Lx = float(np.sqrt(metric[0, 0]))
    Ly = float(np.sqrt(metric[1, 1]))
    Lz = float(np.sqrt(metric[2, 2]))

    if not bc_raw:
        raise ValueError("BC keyword not found in .out file.")
    BCx, BCy, BCz = int(bc_raw[0]), int(bc_raw[1]), int(bc_raw[2])

    Nx = fd_grid[0] + BCx      # number of grid nodes (Dirichlet adds one boundary node)
    Ny = fd_grid[1] + BCy
    Nz = fd_grid[2] + BCz

    dx = Lx / fd_grid[0]       # mesh spacing = length / number-of-intervals
    dy = Ly / fd_grid[1]
    dz = Lz / fd_grid[2]
    dV = dx * dy * dz

    fd_order = int(fd_order_raw[0]) if fd_order_raw else 12
    FDn      = fd_order // 2

    kpt_grid  = [int(x) for x in kpt_grid_raw]  if kpt_grid_raw  else [1, 1, 1]
    kpt_shift = list(kpt_shift_raw)              if kpt_shift_raw else [0.0, 0.0, 0.0]
    spin_typ  = int(spin_typ_raw[0])             if spin_typ_raw  else 0
    nstates   = int(nstates_raw[0])              if nstates_raw   else 0

    params = dict(
        latvec=latvec,
        Lx=Lx, Ly=Ly, Lz=Lz,
        Nx=Nx, Ny=Ny, Nz=Nz, N=Nx*Ny*Nz,
        dx=dx, dy=dy, dz=dz, dV=dV,
        volume=vol,
        FD_ORDER=fd_order, FDn=FDn,
        BCx=BCx, BCy=BCy, BCz=BCz,
        num_intervals=[fd_grid[0], fd_grid[1], fd_grid[2]],
        kpt_grid=kpt_grid,
        kpt_shift=kpt_shift,
        spin_typ=spin_typ,
        nstates=nstates,
    )

    elapsed = time.time() - t0
    print(f"   Lattice  : {Lx:.4f} × {Ly:.4f} × {Lz:.4f}  Bohr")
    print(f"   Grid     : {Nx} × {Ny} × {Nz}   "
          f"(intervals: {fd_grid[0]} × {fd_grid[1]} × {fd_grid[2]})")
    print(f"   Mesh     : dx={dx:.4f}  dy={dy:.4f}  dz={dz:.4f}  Bohr")
    print(f"   dV       : {dV:.4e} Bohr³")
    print(f"   FD_ORDER : {fd_order}  (FDn = {FDn})")
    print(f"   BC       : ({BCx}, {BCy}, {BCz})   0=periodic  1=Dirichlet")
    print(f"   KPOINT_GRID  : {kpt_grid}")
    print(f"   KPOINT_SHIFT : {kpt_shift}")
    print(f"   SPIN_TYP : {spin_typ}   NSTATES : {nstates}")
    print(f"   .out read in {elapsed:.3f}s")
    return params


# ════════════════════════════════════════════════════════════════════════════
#  .psi reader  (translated directly from calculate_pdos.py)
# ════════════════════════════════════════════════════════════════════════════

def read_psi_file(fpath, params, endian="<"):
    """
    Read the binary SPARC wavefunction file (.psi).

    Binary layout (little-endian by default)
    ──────────────────────────────────────────
      Header: Nx Ny Nz Nd  dx dy dz dV  Nspinor_eig isGamma nspin nkpt nband
      Then for kpt in range(nkpt):
        for band in range(nband):
          spin_index kpt_index kpt_vec(3) band_indx
          for spinor in range(Nspinor_eig):
            if isGamma : Nd × float64
            else       : 2×Nd × float64  (interleaved Re/Im, column-major)

    Parameters
    ──────────
    fpath   : str   path to .psi file
    params  : dict  from read_out_file (used for consistency checks)
    endian  : str   '<' little-endian (default), '>' big-endian

    Returns
    ────────
    psi     : ndarray (Nd*Nspinor_eig, nband, nkpt)  complex128 or float64
    header  : dict    Nx,Ny,Nz,Nd,dx,dy,dz,dV,Nspinor_eig,
                      isGamma,nspin,nkpt,nband
    per_band_meta : list[dict]  spin/kpt/kpt_vec/band indices for each (kpt,band)
    """
    t0 = time.time()
    print(f"\n── read_psi_file ───────────────────────────────────────────")
    print(f"   file : {fpath}")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"   .psi file not found: {fpath}")

    def _read(fh, dtype, count=1):
        dt    = np.dtype(dtype).newbyteorder(endian)
        nbytes = dt.itemsize * count
        buf   = fh.read(nbytes)
        if len(buf) != nbytes:
            raise EOFError("Unexpected end of .psi file.")
        return np.frombuffer(buf, dtype=dt, count=count)

    with open(fpath, "rb") as fh:
        Nx          = int(_read(fh, np.int32)[0])
        Ny          = int(_read(fh, np.int32)[0])
        Nz          = int(_read(fh, np.int32)[0])
        Nd          = int(_read(fh, np.int32)[0])
        dx          = float(_read(fh, np.float64)[0])
        dy          = float(_read(fh, np.float64)[0])
        dz          = float(_read(fh, np.float64)[0])
        dV_h        = float(_read(fh, np.float64)[0])
        Nspinor_eig = int(_read(fh, np.int32)[0])
        isGamma     = int(_read(fh, np.int32)[0])
        nspin       = int(_read(fh, np.int32)[0])
        nkpt        = int(_read(fh, np.int32)[0])
        nband       = int(_read(fh, np.int32)[0])

        if Nd != Nx * Ny * Nz:
            raise ValueError(
                f"   .psi header: Nd={Nd} ≠ Nx×Ny×Nz={Nx*Ny*Nz}"
            )

        dtype_psi = np.float64 if isGamma else np.complex128
        psi = np.zeros((Nd * Nspinor_eig, nband, nkpt), dtype=dtype_psi)
        per_band_meta = []

        for kpt in range(nkpt):
            for band in range(nband):
                spin_index = int(_read(fh, np.int32)[0])
                kpt_index  = int(_read(fh, np.int32)[0])
                kpt_vec    = _read(fh, np.float64, 3).astype(float)
                band_indx  = int(_read(fh, np.int32)[0])

                per_band_meta.append(dict(
                    spin_index=spin_index,
                    kpt_index=kpt_index,
                    kpt_vec=kpt_vec,
                    band_indx=band_indx,
                ))

                for spinor in range(Nspinor_eig):
                    start = spinor * Nd
                    end   = (spinor + 1) * Nd
                    if isGamma:
                        arr = _read(fh, np.float64, Nd)
                        psi[start:end, band, kpt] = arr
                    else:
                        raw = _read(fh, np.float64, 2 * Nd)
                        raw = np.reshape(raw, (2, Nd), order='F')  # MATLAB column-major
                        psi[start:end, band, kpt] = raw[0, :] + 1j * raw[1, :]

    header = dict(
        Nx=Nx, Ny=Ny, Nz=Nz, Nd=Nd,
        dx=dx, dy=dy, dz=dz, dV=dV_h,
        Nspinor_eig=Nspinor_eig, isGamma=isGamma,
        nspin=nspin, nkpt=nkpt, nband=nband,
    )

    # ── consistency checks against .out params ───────────────────────────
    for key in ("Nx", "Ny", "Nz"):
        if params[key] != header[key]:
            raise ValueError(
                f"   Grid mismatch in {key}: "
                f".out = {params[key]},  .psi header = {header[key]}"
            )
    dV_out = params['dV']
    if abs(dV_h - dV_out) / max(dV_out, 1e-30) > 0.05:
        print(f"   ⚠  dV mismatch: .psi header = {dV_h:.4e}, "
              f".out computed = {dV_out:.4e}  (>5%)")
    else:
        print(f"   ✓  Grid & dV consistent with .out")

    elapsed = time.time() - t0
    print(f"   psi shape  : {psi.shape}  "
          f"(Nd={Nd}, nband={nband}, nkpt={nkpt})")
    print(f"   isGamma={isGamma}  Nspinor_eig={Nspinor_eig}  nspin={nspin}")
    print(f"   dV (header) = {dV_h:.4e} Bohr³")
    print(f"   .psi read in {elapsed:.3f}s  "
          f"({psi.nbytes/1024/1024:.1f} MB)")
    return psi, header, per_band_meta


# ════════════════════════════════════════════════════════════════════════════
#  .eigen reader  (standalone, no PDOSCalculator dependency)
# ════════════════════════════════════════════════════════════════════════════

def read_eigen_file(fpath, nband, nkpt):
    """
    Parse a SPARC .eigen file.

    File format (per k-point block)
    ─────────────────────────────────
      kred #N = (kx, ky, kz)
      weight = w
      n        eigval                 occ
      1        <Ha>     <occ>
      ...

    Parameters
    ──────────
    fpath  : str  path to .eigen file
    nband  : int  number of bands (from .psi header or .out NSTATES)
    nkpt   : int  number of k-points (from .psi header)

    Returns  (all sorted ascending in eigenvalue within each k)
    ────────
    eign      : (nband, nkpt)  float64   [Ha]
    occ       : (nband, nkpt)  float64   occupation numbers
    kpts      : (nkpt,  3)    float64   fractional k-coordinates
    kpt_wts   : (nkpt,)       float64   k-point weights
    I_indices : (nband, nkpt) int64     argsort permutation that sorted the bands
    """
    t0 = time.time()
    print(f"\n── read_eigen_file ─────────────────────────────────────────")
    print(f"   file  : {fpath}")
    print(f"   nband = {nband},  nkpt = {nkpt}")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"   .eigen file not found: {fpath}")

    Ha2eV = 27.21140795

    # Pre-compiled regex patterns
    kvec_re = re.compile(
        r"kred\s*#(\d+)\s*=\s*\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)"
    )
    wgt_re   = re.compile(r"weight\s*=\s*(" + _NUM + r")")
    eign_re  = re.compile(r"\s*(\d+)\s+(" + _NUM + r")\s+(" + _NUM + r")")

    with open(fpath) as fh:
        lines = fh.readlines()

    # Find all k-point header lines
    kpt_line_idx = [i for i, l in enumerate(lines) if "kred" in l]
    if len(kpt_line_idx) != nkpt:
        raise ValueError(
            f"   Expected {nkpt} k-point blocks, found {len(kpt_line_idx)} "
            f"in {fpath}."
        )

    eign_raw = np.zeros((nband, nkpt))
    occ_raw  = np.zeros((nband, nkpt))
    kpts     = np.zeros((nkpt, 3))
    kpt_wts  = np.zeros(nkpt)

    for ki in range(nkpt):
        base = kpt_line_idx[ki]

        # k-vector
        m = kvec_re.search(lines[base])
        if not m:
            raise ValueError(f"   Could not parse k-vector from line: {lines[base]!r}")
        kidx = int(m.group(1)) - 1   # 0-based
        kpts[ki] = [float(m.group(2)), float(m.group(3)), float(m.group(4))]

        # weight
        m = wgt_re.search(lines[base + 1])
        if not m:
            raise ValueError(f"   Could not parse weight for k#{ki+1}.")
        kpt_wts[ki] = float(m.group(1))

        # eigenvalues & occupations (start at base+3, skip header line base+2)
        for bi, line in enumerate(lines[base + 3 : base + 3 + nband]):
            m = eign_re.search(line)
            if not m:
                raise ValueError(
                    f"   Could not parse eigenvalue line for k#{ki+1}, "
                    f"band {bi+1}: {line!r}"
                )
            eign_raw[bi, ki] = float(m.group(2))
            occ_raw [bi, ki] = float(m.group(3))

    # Sort bands by ascending eigenvalue within each k
    I_indices = np.argsort(eign_raw, axis=0)          # (nband, nkpt)
    eign = np.take_along_axis(eign_raw, I_indices, axis=0)
    occ  = np.take_along_axis(occ_raw,  I_indices, axis=0)

    elapsed = time.time() - t0
    print(f"   k-weight sum = {kpt_wts.sum():.6f}")
    print(f"   Energy range : [{eign.min()*Ha2eV:.3f}, "
          f"{eign.max()*Ha2eV:.3f}] eV")
    for ki in range(nkpt):
        kv = kpts[ki]
        print(f"     k#{ki+1:>3d} = ({kv[0]:+.4f}, {kv[1]:+.4f}, {kv[2]:+.4f})"
              f"  w = {kpt_wts[ki]:.6f}  "
              f"Σocc = {occ[:,ki].sum():.4f}")
    print(f"   .eigen read in {elapsed:.3f}s")
    return eign, occ, kpts, kpt_wts, I_indices
