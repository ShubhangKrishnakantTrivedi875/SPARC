"""
input_params.py
───────────────
Edit this file to supply all inputs to the Kubo-Greenwood conductivity
calculation.  Then run:

    python main.py                          # uses this file
    python main.py --out Al.out ...         # CLI overrides anything here

──────────────────────────────────────────────────────────────────────────
FILE PATHS
──────────────────────────────────────────────────────────────────────────
"""
common_path = '/storage/home/hcoda1/5/strivedi44/scratch/MD11/Forked_SPARC/SPARC/utils/KGconductivitytensor/examples/tests/'

# Path to the SPARC .out file (contains lattice, grid, FD order, BC, …)
OUT_FILE   = common_path + "Al.out_18"

# Path to the SPARC .psi file (binary wavefunction file)
PSI_FILE   = common_path + "Al.psi_18"

# Path to the SPARC .eigen file (eigenvalues, occupations and k-points info)
EIGEN_FILE = common_path + "Al.eigen_18"

# Directory where all output files will be written
OUT_DIR    = common_path + "KG_output_18"

"""
──────────────────────────────────────────────────────────────────────────
FREQUENCY (OMEGA) RANGE
All values in Hartree (atomic units).
──────────────────────────────────────────────────────────────────────────
"""

OMEGA_START = 0   # [Ha]  start of frequency range (start from a small finite value)
OMEGA_END   = 0.2     # [Ha]  end   of frequency range
N_OMEGA     = 200     # number of evenly spaced frequency points

"""
──────────────────────────────────────────────────────────────────────────
BROADENING
──────────────────────────────────────────────────────────────────────────
"""

ETA = 1e-3    # [Ha]  Lorentzian broadening parameter η
              # Controls the width of each delta-function peak.
              # Smaller η → sharper features but needs denser k-mesh.

"""
──────────────────────────────────────────────────────────────────────────
(optional) ENDIANNESS of the .psi binary file
  '<'  little-endian  (default, most modern machines)
  '>'  big-endian
──────────────────────────────────────────────────────────────────────────
"""

PSI_ENDIAN  = "<"
