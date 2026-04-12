#!/usr/bin/env python3
"""
main.py
───────
Entry point for the Kubo-Greenwood conductivity calculation.

Two ways to supply input
────────────────────────
1.  Edit input_params.py and run:
        python main.py

2.  Pass arguments on the command line (overrides input_params.py):
        python main.py --out Al.out --psi Al.psi --eigen Al.eigen \\
                       --outdir KG_output \\
                       --omega_start 0.0 --omega_end 1.0 --n_omega 100 \\
                       --eta 0.01

    Any CLI argument you omit will fall back to the value in input_params.py.

Authors: (your names)
"""

import os
import sys
import time
import argparse
import numpy as np

# ── make sure the sibling modules are importable ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from file_reader import read_out_file, read_psi_file, read_eigen_file
from kg_solver   import build_gradient_info, compute_kubo_greenwood
from printing    import (print_banner, print_run_config, print_out_params,
                         print_psi_summary, print_eigen_summary,
                         print_kg_tensor, print_timing_summary, save_results)


# ════════════════════════════════════════════════════════════════════════════
#  1.  Load defaults from input_params.py
# ════════════════════════════════════════════════════════════════════════════

def _load_input_params():
    """
    Import input_params.py and return its values as a dict.
    Falls back gracefully if the file is missing.
    """
    try:
        import input_params as IP
        return dict(
            out_file    = IP.OUT_FILE,
            psi_file    = IP.PSI_FILE,
            eigen_file  = IP.EIGEN_FILE,
            out_dir     = IP.OUT_DIR,
            omega_start = IP.OMEGA_START,
            omega_end   = IP.OMEGA_END,
            n_omega     = IP.N_OMEGA,
            eta         = IP.ETA,
            psi_endian  = getattr(IP, 'PSI_ENDIAN', '<'),
        )
    except ModuleNotFoundError:
        print("  [warning] input_params.py not found – "
              "all values must be supplied via CLI.")
        return {}
    except Exception as e:
        print(f"  [warning] Could not fully read input_params.py: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
#  2.  Build CLI parser  (all args are optional; defaults come from IP)
# ════════════════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Kubo-Greenwood electrical conductivity from SPARC output files.\n"
            "All flags are optional; missing values fall back to input_params.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "────────\n"
            "  python main.py                          # uses input_params.py only\n"
            "  python main.py --out Al.out --eta 0.02  # partial override\n"
            "  python main.py --out Al.out --psi Al.psi --eigen Al.eigen\n"
            "                 --outdir KG_output\n"
            "                 --omega_start 0 --omega_end 1 --n_omega 100\n"
            "                 --eta 0.01\n"
        ),
    )
    p.add_argument("--out",          type=str,   default=None,
                   help="SPARC .out file")
    p.add_argument("--psi",          type=str,   default=None,
                   help="SPARC .psi file (binary wavefunctions)")
    p.add_argument("--eigen",        type=str,   default=None,
                   help="SPARC .eigen file (eigenvalues + occupations)")
    p.add_argument("--outdir",       type=str,   default=None,
                   help="Output directory")
    p.add_argument("--omega_start",  type=float, default=None,
                   help="Start frequency [Ha]")
    p.add_argument("--omega_end",    type=float, default=None,
                   help="End   frequency [Ha]")
    p.add_argument("--n_omega",      type=int,   default=None,
                   help="Number of frequency points")
    p.add_argument("--eta",          type=float, default=None,
                   help="Lorentzian broadening η [Ha]")
    p.add_argument("--endian",       type=str,   default=None,
                   choices=['<', '>'],
                   help="Byte order of .psi file (< little, > big)")
    return p


# ════════════════════════════════════════════════════════════════════════════
#  3.  Merge defaults with CLI overrides → resolved config dict
# ════════════════════════════════════════════════════════════════════════════

def _resolve_config(ip_defaults, cli_args):
    """
    Priority: CLI > input_params.py > hard-coded fallbacks.
    Raises ValueError if any required field is still missing after merging.
    """
    # Hard-coded fallbacks for optional numeric params
    fallbacks = dict(
        out_dir     = "KG_output",
        omega_start = 0.0,
        omega_end   = 1.0,
        n_omega     = 100,
        eta         = 0.01,
        psi_endian  = "<",
    )

    cfg = {}
    # File paths (required – no safe fallback)
    for key, cli_val in [
        ("out_file",   cli_args.out),
        ("psi_file",   cli_args.psi),
        ("eigen_file", cli_args.eigen),
    ]:
        if cli_val is not None:
            cfg[key] = cli_val
        elif key in ip_defaults:
            cfg[key] = ip_defaults[key]
        else:
            raise ValueError(
                f"Required input '{key}' not found.\n"
                f"  Supply it via input_params.py  OR  --{key.replace('_file','')}"
            )

    # Optional params with fallbacks
    for key, cli_val, fb_key in [
        ("out_dir",     cli_args.outdir,      "out_dir"),
        ("omega_start", cli_args.omega_start, "omega_start"),
        ("omega_end",   cli_args.omega_end,   "omega_end"),
        ("n_omega",     cli_args.n_omega,     "n_omega"),
        ("eta",         cli_args.eta,         "eta"),
        ("psi_endian",  cli_args.endian,      "psi_endian"),
    ]:
        if cli_val is not None:
            cfg[key] = cli_val
        elif key in ip_defaults:
            cfg[key] = ip_defaults[key]
        else:
            cfg[key] = fallbacks[fb_key]

    # Sanity checks
    if cfg['omega_end'] <= cfg['omega_start']:
        raise ValueError(
            f"omega_end ({cfg['omega_end']}) must be > omega_start ({cfg['omega_start']})."
        )
    if cfg['n_omega'] < 1:
        raise ValueError(f"n_omega must be ≥ 1, got {cfg['n_omega']}.")
    if cfg['eta'] <= 0:
        raise ValueError(f"eta must be > 0, got {cfg['eta']}.")

    return cfg


# ════════════════════════════════════════════════════════════════════════════
#  4.  Main pipeline
# ════════════════════════════════════════════════════════════════════════════

def main():
    t_wall_start = time.time()

    # ── parse ────────────────────────────────────────────────────────────
    ip_defaults = _load_input_params()
    parser      = _build_parser()
    cli_args    = parser.parse_args()
    cfg         = _resolve_config(ip_defaults, cli_args)

    # ── source of each parameter (for provenance info) ───────────────────
    def _src(key, cli_val):
        return "CLI" if cli_val is not None else \
               ("input_params.py" if key in ip_defaults else "default")
    print_banner()
    print_run_config(cfg)
    print(f"\n  Parameter sources:")
    print(f"    out_file    : {_src('out_file',   cli_args.out)}")
    print(f"    psi_file    : {_src('psi_file',   cli_args.psi)}")
    print(f"    eigen_file  : {_src('eigen_file', cli_args.eigen)}")
    print(f"    omega range : {_src('omega_start',cli_args.omega_start)}")
    print(f"    eta         : {_src('eta',        cli_args.eta)}")

    # ── frequency array ───────────────────────────────────────────────────
    Omega = np.linspace(cfg['omega_start'], cfg['omega_end'], cfg['n_omega'])

    # ── STEP 1: read .out ─────────────────────────────────────────────────
    t0 = time.time()
    params = read_out_file(cfg['out_file'])
    t_read_out = time.time() - t0
    print_out_params(params)

    # ── STEP 2: read .psi ─────────────────────────────────────────────────
    t0 = time.time()
    psi, header, _ = read_psi_file(cfg['psi_file'], params,
                                   endian=cfg['psi_endian'])
    t_read_psi = time.time() - t0
    print_psi_summary(psi, header)

    # ── STEP 3: read .eigen ───────────────────────────────────────────────
    t0 = time.time()
    eign, occ, kpts, kpt_wts, I_indices = read_eigen_file(
        cfg['eigen_file'],
        nband = header['nband'],
        nkpt  = header['nkpt'],
    )
    t_read_eigen = time.time() - t0
    print_eigen_summary(eign, occ, kpts, kpt_wts)

    # ── STEP 4: build gradient matrices ───────────────────────────────────
    t0 = time.time()
    grad_info = build_gradient_info(params)
    t_grad = time.time() - t0

    # ── STEP 5: compute KG tensor ─────────────────────────────────────────
    sigma, timing_kg = compute_kubo_greenwood(
        psi, header,
        eign, occ, kpts, kpt_wts, I_indices,
        grad_info, params, Omega,
        eta=cfg['eta'],
    )
    print_kg_tensor(sigma, Omega, cfg['eta'])

    # ── STEP 6: save results ──────────────────────────────────────────────
    save_results(sigma, Omega, cfg['out_dir'], params, cfg)

    # ── timing summary ────────────────────────────────────────────────────
    t_total = time.time() - t_wall_start
    print_timing_summary(t_read_out, t_read_psi, t_read_eigen,
                         t_grad, timing_kg, t_total)

    print(f"\n  All done.  Results in: {cfg['out_dir']}/\n")


if __name__ == "__main__":
    main()
