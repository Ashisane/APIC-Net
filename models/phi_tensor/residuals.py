"""
Φ Tensor — Residual Computation Module

Computes 4 physics residuals per VSM at each timestep.
Uses per-unit swing equation matching vsm_simulator.py exactly.

Residuals:
  r1: Swing equation — does dω/dt match the net torque?
  r2: Angle-frequency — does dδ/dt match ω?
  r3: Power balance — does power imbalance explain frequency change?
  r4: COI feedback — does received ω_C match local ω?
"""
import numpy as np

# Physics constants — MUST match vsm_simulator.py
OMEGA_STAR = 2 * np.pi * 50.0
DT = 0.005
H = 5.0       # Per-unit inertia (vsm_simulator.py line 28)
D_VAL = 25.0  # Damping (vsm_simulator.py: D=50/2)
F_VAL = 10.0  # Friction (vsm_simulator.py: F_COEFF=20/2)

# Column indices in recording array
COL_OMEGA   = 0
COL_DELTA   = 1
COL_P       = 2
COL_PSTAR   = 3
COL_VREF    = 4
COL_VDC     = 5
COL_OMEGA_C = 6
COL_IF      = 7

N_RESIDUALS = 4


def compute_residuals_vectorized(data):
    """
    Compute 4 physics residuals for an entire scenario at once.

    Args:
        data: (N_steps, 8) raw signal array for one VSM in one scenario.

    Returns:
        residuals: (N_steps-1, 4) — residuals r1-r4 at each timestep.
        The t-th row corresponds to the transition from data[t] to data[t+1].
    """
    n = data.shape[0] - 1  # number of transitions

    omega     = data[:, COL_OMEGA]
    delta     = data[:, COL_DELTA]
    p         = data[:, COL_P]
    p_star    = data[:, COL_PSTAR]
    omega_C   = data[:, COL_OMEGA_C]

    # Time derivatives (forward difference)
    d_omega = omega[1:] - omega[:-1]   # omega(t+1) - omega(t)
    d_delta = delta[1:] - delta[:-1]

    # Values at time t (used for the RHS of equations)
    omega_t   = omega[:-1]
    p_t       = p[:-1]
    p_star_t  = p_star[:-1]
    omega_C_t = omega_C[:-1]

    # ── r1: Swing equation residual ──
    # 2H * dω/dt = (p* - p) * ω* + D*(ω* - ω) + F*(ωC - ω)
    # Residual = LHS - RHS
    lhs_r1 = 2 * H * d_omega / DT
    rhs_r1 = (
        (p_star_t - p_t) * OMEGA_STAR
        + D_VAL * (OMEGA_STAR - omega_t)
        + F_VAL * (omega_C_t - omega_t)
    )
    r1 = lhs_r1 - rhs_r1

    # ── r2: Angle-frequency consistency ──
    # dδ/dt = ω  →  residual = dδ/dt - ω(t)
    r2 = d_delta / DT - omega_t

    # ── r3: Power balance residual ──
    # From swing eq: (p* - p)*ω* = 2H*dω/dt - D*(ω* - ω) - F*(ωC - ω)
    # So: (p* - p) = [2H*dω/dt - D*(ω* - ω) - F*(ωC - ω)] / ω*
    # Residual = (p* - p) - expected_power_imbalance
    expected_dp = (
        2 * H * d_omega / DT
        - D_VAL * (OMEGA_STAR - omega_t)
        - F_VAL * (omega_C_t - omega_t)
    ) / OMEGA_STAR
    r3 = (p_star_t - p_t) - expected_dp

    # ── r4: COI feedback consistency ──
    # ωC_received - ω_local (proxy: own omega)
    # Under normal operation: ωC ≈ ω_j (all VSMs near ω*)
    # Under COI attack: ωC_received deviates from ω_j
    r4 = omega_C_t - omega_t

    residuals = np.stack([r1, r2, r3, r4], axis=1)  # (n, 4)
    return residuals


def validate_residuals(data_dir=None):
    """
    Gate 1: Verify residuals are near-zero on normal data.
    Loads vsm_0.npz, picks 1000 normal timesteps, reports mean |r|.
    """
    import os
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data"
        )

    d = np.load(os.path.join(data_dir, "vsm_0.npz"))
    recording = d["data"]       # (N_scenarios, N_steps, 8)
    labels = d["labels"]        # (N_scenarios, N_steps)

    # Find normal scenarios
    normal_mask = labels.sum(axis=1) == 0
    normal_data = recording[normal_mask]

    print(f"Normal scenarios: {normal_data.shape[0]}")
    print(f"Testing residuals on first 10 normal scenarios (20,000 timesteps)...\n")

    all_residuals = []
    for i in range(min(10, normal_data.shape[0])):
        r = compute_residuals_vectorized(normal_data[i])  # (1999, 4)
        all_residuals.append(r)

    all_r = np.concatenate(all_residuals, axis=0)

    names = ["r1 (swing)", "r2 (angle)", "r3 (power)", "r4 (COI)"]
    thresholds = [0.01, 0.01, 0.01, 0.5]
    all_pass = True

    print(f"{'Residual':<16} {'mean |r|':>12} {'max |r|':>12} {'threshold':>10} {'PASS?':>6}")
    print("-" * 60)
    for i, (name, thresh) in enumerate(zip(names, thresholds)):
        mean_abs = np.mean(np.abs(all_r[:, i]))
        max_abs = np.max(np.abs(all_r[:, i]))
        passed = mean_abs < thresh
        if not passed:
            all_pass = False
        status = "✅" if passed else "❌ FAIL"
        print(f"{name:<16} {mean_abs:>12.6f} {max_abs:>12.6f} {thresh:>10.4f} {status:>6}")

    print(f"\n{'='*60}")
    if all_pass:
        print("GATE 1 PASSED ✅ — All residuals near-zero on normal data.")
    else:
        print("GATE 1 FAILED ❌ — Fix residual formulas before proceeding.")

    return all_pass, all_r


if __name__ == "__main__":
    validate_residuals()
