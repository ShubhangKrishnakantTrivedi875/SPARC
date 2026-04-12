"""
printing.py
───────────
All printing and file-saving for the Kubo-Greenwood calculation.

Provides
────────
  print_banner()
  print_run_config(cfg)
  print_out_params(params)
  print_eigen_summary(eign, occ, kpts, kpt_wts)
  print_psi_summary(psi, header)
  print_kg_tensor(sigma, Omega, eta)
  print_timing_summary(t_read, t_grad, t_kg)
  save_results(sigma, Omega, out_dir)

All functions write to stdout.  save_results also writes .dat files.
"""

import os
import time
import numpy as np

Ha2eV  = 27.21140795
# Unit conversion: 1 a.u. of conductivity = e²/(ħ a₀) = 4.6009×10⁶ S/m
#   → 1 a.u. = 4.6009×10⁶  (Ω·m)⁻¹  =  4.6009×10⁴  (Ω·cm)⁻¹
#   → 1 a.u. = 0.046009 (μΩ·cm)⁻¹   [since 1 (μΩ·cm)⁻¹ = 10⁸ S/m]
AU_TO_SI        = 4.6009e6    # [S/m]        per a.u. of conductivity
AU_TO_OHM_INV_M   = 4.6009e6  # [(Ω·m)⁻¹]   per a.u.
AU_TO_MUOHM_CM_INV = 4.6009e-2 # [(μΩ·cm)⁻¹] per a.u.  (= AU_TO_SI / 1e8)


# ════════════════════════════════════════════════════════════════════════════
#  Header / banner
# ════════════════════════════════════════════════════════════════════════════

def print_banner():
    w = 65
    print("=" * w)
    print("  Kubo-Greenwood Electrical Conductivity Tensor")
    print("  Real-space FD (SPARC/M-SPARC), orthogonal cell")
    print("=" * w)


def print_run_config(cfg):
    """
    Print the resolved run configuration (merged from input_params + CLI).

    cfg : dict with keys  out_file, psi_file, eigen_file, out_dir,
                          omega_start, omega_end, n_omega, eta, psi_endian
    """
    w = 65
    print("\n" + "─" * w)
    print("  Run configuration")
    print("─" * w)
    print(f"  .out  file : {cfg['out_file']}")
    print(f"  .psi  file : {cfg['psi_file']}")
    print(f"  .eigen file: {cfg['eigen_file']}")
    print(f"  output dir : {cfg['out_dir']}")
    print(f"  ω range    : {cfg['omega_start']:.4f} → {cfg['omega_end']:.4f} Ha"
          f"  ({cfg['n_omega']} pts)")
    print(f"             = {cfg['omega_start']*Ha2eV:.3f} → "
          f"{cfg['omega_end']*Ha2eV:.3f} eV")
    print(f"  η          : {cfg['eta']:.4e} Ha = {cfg['eta']*Ha2eV:.4e} eV")
    print(f"  .psi endian: {cfg['psi_endian']}")
    print("─" * w)


# ════════════════════════════════════════════════════════════════════════════
#  .out params
# ════════════════════════════════════════════════════════════════════════════

def print_out_params(params):
    w = 65
    print("\n" + "─" * w)
    print("  SPARC .out file — extracted parameters")
    print("─" * w)

    lv = params['latvec']
    print("  Lattice vectors [Bohr]:")
    labels = ["a₁", "a₂", "a₃"]
    for i in range(3):
        r = lv[i]
        print(f"    {labels[i]} = [{r[0]:+12.6f}  {r[1]:+12.6f}  {r[2]:+12.6f}]")

    print(f"  Cell lengths  : Lx={params['Lx']:.6f}  "
          f"Ly={params['Ly']:.6f}  Lz={params['Lz']:.6f}  [Bohr]")
    print(f"  Cell volume   : {params['volume']:.6f} Bohr³")
    print(f"  Grid nodes    : Nx={params['Nx']}  Ny={params['Ny']}  Nz={params['Nz']}  "
          f"(total N={params['N']})")
    ni = params['num_intervals']
    print(f"  FD intervals  : {ni[0]} × {ni[1]} × {ni[2]}")
    print(f"  Mesh spacing  : dx={params['dx']:.6f}  "
          f"dy={params['dy']:.6f}  dz={params['dz']:.6f}  [Bohr]")
    print(f"  dV            : {params['dV']:.6e} Bohr³")
    print(f"  FD_ORDER      : {params['FD_ORDER']}  (FDn = {params['FDn']})")
    _bc_str = {0: "P", 1: "D"}
    print(f"  BC            : {_bc_str[params['BCx']]} {_bc_str[params['BCy']]} "
          f"{_bc_str[params['BCz']]}   (P=periodic  D=Dirichlet)")
    print(f"  KPOINT_GRID   : {params['kpt_grid']}")
    print(f"  KPOINT_SHIFT  : {params['kpt_shift']}")
    print(f"  SPIN_TYP      : {params['spin_typ']}")
    print(f"  NSTATES       : {params['nstates']}")
    print("─" * w)


# ════════════════════════════════════════════════════════════════════════════
#  .psi summary
# ════════════════════════════════════════════════════════════════════════════

def print_psi_summary(psi, header):
    w = 65
    print("\n" + "─" * w)
    print("  Wavefunction file (.psi) — summary")
    print("─" * w)
    print(f"  psi shape     : {psi.shape}   (Nd, nband, nkpt)")
    print(f"  dtype         : {psi.dtype}")
    print(f"  isGamma       : {header['isGamma']}")
    print(f"  Nspinor_eig   : {header['Nspinor_eig']}")
    print(f"  nspin         : {header['nspin']}")
    print(f"  nkpt          : {header['nkpt']}")
    print(f"  nband         : {header['nband']}")
    print(f"  Nd (header)   : {header['Nd']}")
    print(f"  dV (header)   : {header['dV']:.6e} Bohr³")
    # Quick norms for the first k-point as a sanity check
    dV = header['dV']
    norms = [np.sum(np.abs(psi[:, n, 0])**2) * dV for n in range(min(3, header['nband']))]
    print(f"  ‖ψ_{0}‖² dV  : {norms[0]:.6f}  (should be ≈ 1)")
    if len(norms) > 1:
        print(f"  ‖ψ_{1}‖² dV  : {norms[1]:.6f}")
    print("─" * w)


# ════════════════════════════════════════════════════════════════════════════
#  .eigen summary
# ════════════════════════════════════════════════════════════════════════════

def print_eigen_summary(eign, occ, kpts, kpt_wts):
    w = 65
    print("\n" + "─" * w)
    print("  Eigenvalue file (.eigen) — summary")
    print("─" * w)
    nband, nkpt = eign.shape
    print(f"  nkpt={nkpt}  nband={nband}")
    print(f"  k-weight sum  : {kpt_wts.sum():.8f}  (should be 1.0)")
    print(f"  E range       : [{eign.min()*Ha2eV:.4f}, {eign.max()*Ha2eV:.4f}] eV")
    print()
    print(f"  {'k#':>4}  {'kx':>8}  {'ky':>8}  {'kz':>8}  "
          f"{'weight':>10}  {'Σocc':>8}  "
          f"{'Emin(eV)':>10}  {'Emax(eV)':>10}")
    print(f"  " + "-"*74)
    for ki in range(nkpt):
        kv = kpts[ki]
        print(f"  {ki+1:>4}  {kv[0]:>+8.4f}  {kv[1]:>+8.4f}  {kv[2]:>+8.4f}  "
              f"{kpt_wts[ki]:>10.6f}  {occ[:,ki].sum():>8.4f}  "
              f"{eign[0,ki]*Ha2eV:>10.4f}  {eign[-1,ki]*Ha2eV:>10.4f}")
    print("─" * w)


# ════════════════════════════════════════════════════════════════════════════
#  KG tensor — console print
# ════════════════════════════════════════════════════════════════════════════

def print_kg_tensor(sigma, Omega, eta):
    """
    Print Re(σ_{αβ}(ω)) — the physical absorptive optical conductivity —
    at a few representative frequencies, in atomic units and SI.

    Note: Re(sigma) from the resolvent formula is the physical quantity.
    """
    w = 65
    print("\n" + "─" * w)
    print("  Kubo-Greenwood conductivity tensor  Re[σ_{αβ}(ω)]")
    print(f"  η = {eta:.4e} Ha = {eta*Ha2eV:.4e} eV")
    print(f"  Re(σ) is the physical absorptive optical conductivity.")
    print("─" * w)

    labels = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    idx    = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]

    # Pick ~5 representative frequencies to display
    n_omega  = len(Omega)
    show_idx = np.unique(np.linspace(0, n_omega - 1, min(5, n_omega)).astype(int))

    header_row = "  ".join(f"Re(s_{l:2s})" for l in labels)
    print(f"\n  {'w_Ha':>10}  {'w_eV':>10}  {header_row}   [a.u.]")
    print("  " + "-" * (12 + 12 + 14 * len(labels)))
    for wi in show_idx:
        row_vals = "  ".join(f"{sigma[wi,a,b].real:>12.4e}" for (a,b) in idx)
        print(f"  {Omega[wi]:>10.4f}  {Omega[wi]*Ha2eV:>10.4f}  {row_vals}")

    # Isotropic average at a few points, with unit conversions
    sigma_avg = (sigma[:,0,0] + sigma[:,1,1] + sigma[:,2,2]) / 3.0
    print(f"\n  Isotropic Re(σ_avg) — unit conversions:")
    print(f"  {'w_Ha':>10}  {'w_eV':>10}  {'[a.u.]':>14}  {'[S/m]':>14}  {'[(μΩcm)⁻¹]':>14}")
    print("  " + "-" * 70)
    for wi in show_idx:
        s_au  = sigma_avg[wi].real
        s_si  = s_au * AU_TO_SI
        s_muo = s_au * AU_TO_MUOHM_CM_INV
        print(f"  {Omega[wi]:>10.4f}  {Omega[wi]*Ha2eV:>10.4f}  "
              f"{s_au:>14.6e}  {s_si:>14.6e}  {s_muo:>14.6e}")
    print("─" * w)


# ════════════════════════════════════════════════════════════════════════════
#  Timing summary
# ════════════════════════════════════════════════════════════════════════════

def print_timing_summary(t_read_out, t_read_psi, t_read_eigen,
                          t_grad, timing_kg, t_total):
    w = 65
    print("\n" + "─" * w)
    print("  Timing summary")
    print("─" * w)
    print(f"  read .out          : {t_read_out:.3f}s")
    print(f"  read .psi          : {t_read_psi:.3f}s")
    print(f"  read .eigen        : {t_read_eigen:.3f}s")
    print(f"  build gradient     : {t_grad:.3f}s")
    print(f"  KG tensor (total)  : {timing_kg['total']:.3f}s")
    print(f"    ↳ gradient mats  : {timing_kg['gradient']:.3f}s")
    print(f"    ↳ matrix elems   : {timing_kg['matrix_elem']:.3f}s")
    print(f"    ↳ KG frequency Σ : {timing_kg['kg_sum']:.3f}s")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL              : {t_total:.3f}s")
    print("─" * w)


# ════════════════════════════════════════════════════════════════════════════
#  Save results to disk
# ════════════════════════════════════════════════════════════════════════════

def save_results(sigma, Omega, out_dir, params, cfg):
    """
    Write all conductivity data to plain-text .dat files.

    Files written
    ─────────────
      sigma_xx.dat  sigma_xy.dat  sigma_xz.dat
      sigma_yy.dat  sigma_yz.dat  sigma_zz.dat
      sigma_trace_avg.dat   (isotropic average)
      run_info.txt    (all input params + timing)

    Format of each sigma file .dat
    ────────────────────────────
    # omega_Ha   omega_eV   Re(sigma)   Im(sigma)
    <floats in scientific notation>
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n── save_results ────────────────────────────────────────────")
    print(f"   Writing to: {out_dir}/")

    components = [
        (0,0,'xx'), (0,1,'xy'), (0,2,'xz'),
        (1,1,'yy'), (1,2,'yz'),
        (2,2,'zz'),
    ]

    hdr = ("# omega_Ha          omega_eV            "
           "Re(sigma)[a.u.]     Im(sigma)[a.u.]     "
           "Re(sigma)[S/m]      Re(sigma)[(uOhm.cm)^-1]")

    for (ai, bi, lab) in components:
        fname = os.path.join(out_dir, f"sigma_{lab}.dat")
        re_au = sigma[:,ai, bi].real
        data  = np.column_stack([
            Omega,
            Omega * Ha2eV,
            re_au,
            sigma[:,ai, bi].imag,
            re_au * AU_TO_SI,
            re_au * AU_TO_MUOHM_CM_INV,
        ])
        np.savetxt(fname, data, header=hdr, fmt="%.10e", comments='')
        print(f"   sigma_{lab}.dat written  ({len(Omega)} rows)")

    # Trace  σ_trace_avg = Tr(σ)/3
    sigma_trace_avg = (sigma[:,0,0] + sigma[:,1,1] + sigma[:,2,2]) / 3.0
    fname = os.path.join(out_dir, "sigma_trace_avg.dat")
    re_au = sigma_trace_avg.real
    data  = np.column_stack([
        Omega, Omega * Ha2eV,
        re_au, sigma_trace_avg.imag,
        re_au * AU_TO_SI,
        re_au * AU_TO_MUOHM_CM_INV,
    ])
    np.savetxt(fname, data,
               header=hdr.replace("sigma)", "sigma_trace_avg)"),
               fmt="%.10e", comments='')
    print(f"   sigma_trace_avg.dat written")

    # ── run_info.txt ──────────────────────────────────────────────────────
    info_path = os.path.join(out_dir, "run_info.txt")
    lv        = params['latvec']
    with open(info_path, 'w') as f:
        f.write("Kubo-Greenwood Run Info\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Input files\n")
        f.write(f"  .out   : {cfg['out_file']}\n")
        f.write(f"  .psi   : {cfg['psi_file']}\n")
        f.write(f"  .eigen : {cfg['eigen_file']}\n")
        f.write(f"  outdir : {cfg['out_dir']}\n\n")

        f.write("Frequency range\n")
        f.write(f"  omega_start = {cfg['omega_start']:.6f} Ha"
                f" = {cfg['omega_start']*Ha2eV:.6f} eV\n")
        f.write(f"  omega_end   = {cfg['omega_end']:.6f} Ha"
                f" = {cfg['omega_end']*Ha2eV:.6f} eV\n")
        f.write(f"  n_omega     = {cfg['n_omega']}\n")
        f.write(f"  eta         = {cfg['eta']:.6e} Ha\n\n")

        f.write("Lattice vectors [Bohr]\n")
        for i in range(3):
            r = lv[i]
            f.write(f"  a{i+1} = [{r[0]:+.8f}  {r[1]:+.8f}  {r[2]:+.8f}]\n")
        f.write(f"\n")
        f.write(f"  Lx={params['Lx']:.8f}  "
                f"Ly={params['Ly']:.8f}  Lz={params['Lz']:.8f}  [Bohr]\n")
        f.write(f"  Volume = {params['volume']:.8f} Bohr³\n\n")

        f.write("Grid\n")
        ni = params['num_intervals']
        f.write(f"  FD_GRID    : {ni[0]} × {ni[1]} × {ni[2]}\n")
        f.write(f"  Nodes      : {params['Nx']} × {params['Ny']} × {params['Nz']}\n")
        f.write(f"  FD_ORDER   : {params['FD_ORDER']}\n")
        _bc_str = {0: "P", 1: "D"}
        f.write(f"  BC         : {_bc_str[params['BCx']]} {_bc_str[params['BCy']]} "
                f"{_bc_str[params['BCz']]}\n")
        f.write(f"  KPOINT_GRID: {params['kpt_grid']}\n")
        f.write(f"  KPOINT_SHIFT: {params['kpt_shift']}\n")
    print(f"   run_info.txt written")
    print("─" * 65)
