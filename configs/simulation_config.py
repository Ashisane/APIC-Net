"""
Simulation configuration for IEEE 39-bus VSM dataset generation.
All hyperparameters in one place — no magic numbers scattered across files.

VSM bus parameters are extracted from pandapower.networks.case39() via
AC power flow (see extract_case39_params() at bottom of file).
"""

import numpy as np


# ──────────────────────────────────────────────
# Time & Integration
# ──────────────────────────────────────────────
DT = 0.005                          # Euler timestep [s]
T_TOTAL = 10.0                      # Scenario duration [s]
N_STEPS = int(T_TOTAL / DT)         # 2000 timesteps per scenario

# ──────────────────────────────────────────────
# Nominal Electrical Parameters
# ──────────────────────────────────────────────
F_NOMINAL = 50.0                    # Nominal frequency [Hz]
OMEGA_STAR = 2 * np.pi * F_NOMINAL  # Nominal angular frequency [rad/s]

# ──────────────────────────────────────────────
# VSM Parameters (per-VSM, uniform for simplicity)
# ──────────────────────────────────────────────
J = 2000.0                          # Virtual moment of inertia [SI: kg⋅m²]
D = 50.0                            # Virtual damping coefficient
F_COEFF = 20.0                      # Virtual friction coefficient (COI coupling)
S_BASE = 100e6                      # Base power [W] — 100 MVA (IEEE 39-bus)

# ──────────────────────────────────────────────
# AVR / Voltage Controller (FO-PI)
# ──────────────────────────────────────────────
KP = 0.001                          # Proportional gain
KI = 0.5                            # Integral gain
MF = 1.0                            # Mutual inductance coefficient
A_FRAC = 1.0                        # Fractional order (simplified to 1)
V_REF_NOMINAL = 1.0                 # Nominal voltage reference [p.u.]

# ──────────────────────────────────────────────
# IEEE 39-Bus VSM Configuration
# Extracted from pandapower.networks.case39() AC power flow.
#   Bus 30 = ext_grid (slack bus), Buses 31–35 = generators
# ──────────────────────────────────────────────
VSM_BUS_IDS = [30, 31, 32, 33, 34, 35]   # Buses converted to VSMs
N_VSM = len(VSM_BUS_IDS)                  # 6 VSMs

# Active power setpoints [p.u. on 100 MVA base]
# Source: case39() → pp.runpp() → res_bus.p_mw / res_gen.p_mw
#   Bus 30 (ext_grid/slack): ~250 MW → 2.50 p.u.
#   Bus 31 (gen idx 1):       650 MW → 6.50 p.u.
#   Bus 32 (gen idx 2):       632 MW → 6.32 p.u.
#   Bus 33 (gen idx 3):       508 MW → 5.08 p.u.
#   Bus 34 (gen idx 4):       650 MW → 6.50 p.u.
#   Bus 35 (gen idx 5):       560 MW → 5.60 p.u.
P_SETPOINTS = np.array([2.50, 6.50, 6.32, 5.08, 6.50, 5.60])

# Nominal bus voltages [p.u.] from AC power flow
#   Extracted from: net.res_bus.vm_pu at each VSM bus
V_NOM_PER_VSM = np.array([0.9820, 0.9841, 0.9972, 1.0123, 1.0494, 1.0636])

# ──────────────────────────────────────────────
# Load Variation
# ──────────────────────────────────────────────
LOAD_RANGE = (0.8, 1.2)            # p.u. multiplier range for load profiles

# ──────────────────────────────────────────────
# Dataset Generation
# ──────────────────────────────────────────────
N_NORMAL = 500                      # Number of normal scenarios
N_ATTACK = 3000                     # Total attack scenarios (500 per type × 6 types? no: 4 types)
# Distribution: 750 per attack type across 6 VSMs (125 per VSM per type)
# Adjusted: 4 types × 750 = 3000 total

ATTACK_TYPES = ["freq", "coi", "power", "voltage"]
N_PER_ATTACK_TYPE = N_ATTACK // len(ATTACK_TYPES)  # 750

# ──────────────────────────────────────────────
# Attack Parameter Ranges
# ──────────────────────────────────────────────
ATTACK_AMP_RANGE = (1.0, 5.0)      # Amplitude [rad/s or p.u. equivalent]
ATTACK_TRIG_RANGE = (1.0, 7.0)     # Trigger time [s]
ATTACK_DUR_RANGE = (0.5, 2.0)      # Duration [s]
ATTACK_WAVEFORMS = ["constant", "sine", "square", "ramp"]

# ──────────────────────────────────────────────
# Sliding Window (for sequence models)
# ──────────────────────────────────────────────
WINDOW_SIZE = 20                    # 20 timesteps = 100ms
STRIDE = 1                          # Stride of 1 timestep

# ──────────────────────────────────────────────
# Dataset Split
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Validation Thresholds
# ──────────────────────────────────────────────
FREQ_NORMAL_MIN = 49.0              # Hz — normal operation floor
FREQ_NORMAL_MAX = 51.0              # Hz — normal operation ceiling
FREQ_ATTACK_DEV_MIN = 0.1           # Hz — minimum attack deviation

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
