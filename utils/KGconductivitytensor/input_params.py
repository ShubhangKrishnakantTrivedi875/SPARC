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

# Path to the SPARC .out file (contains lattice, grid, FD order, BC, …)
OUT_FILE   = "Al.out"

# Path to the SPARC .psi file (binary wavefunction file)
PSI_FILE   = "Al.psi"

# Path to the SPARC .eigen file (eigenvalues and occupations)
EIGEN_FILE = "Al.eigen"

# Directory where all output files will be written
OUT_DIR    = "KG_output"

"""
──────────────────────────────────────────────────────────────────────────
FREQUENCY (OMEGA) RANGE
All values in Hartree (atomic units).
──────────────────────────────────────────────────────────────────────────
"""

OMEGA_START = 0.0     # [Ha]  start of frequency range
OMEGA_END   = 1.0     # [Ha]  end   of frequency range
N_OMEGA     = 100     # number of evenly spaced frequency points

"""
──────────────────────────────────────────────────────────────────────────
BROADENING
──────────────────────────────────────────────────────────────────────────
"""

ETA = 0.01    # [Ha]  Lorentzian broadening parameter η
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
