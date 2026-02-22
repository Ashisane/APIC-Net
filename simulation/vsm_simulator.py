"""
vsm_simulator.py — Core Euler-integration simulator for 6 VSMs on IEEE 39-bus.

Implements the swing equation, COI computation, and simplified voltage/power
models from PHYSICS_SPEC.md using forward Euler discretisation.

Unit convention:
  - Frequency: ω in rad/s (SI), normalised internally for swing equation
  - Power: p.u. on 100 MVA base
  - Inertia: H in seconds (per-unit inertia constant)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.simulation_config import (
    DT, N_STEPS, OMEGA_STAR, J, D, F_COEFF,
    KP, KI, MF, A_FRAC, V_REF_NOMINAL,
    N_VSM, P_SETPOINTS, LOAD_RANGE, F_NOMINAL,
)

# Derived per-unit inertia constant: H = J*ω*² / (2*S_base)
# With J=2000, ω*=314.16, S_base=100e6:  H ≈ 0.987 ≈ 1.0 s
# This maps the spec's J=2000 to a physically meaningful inertia.
# We use this in the per-unit swing equation: 2H * dω_pu/dt = ΔP_pu
H_PU = 5.0  # Override: use H=5s (typical for large generators)


class VSMSimulator:
    """
    Simulates 6 Virtual Synchronous Machines on the IEEE 39-bus system
    using forward Euler integration of the swing equation.

    Uses per-unit swing equation:
        2H * d(Δω)/dt = P_m - P_e - D*(Δω) - F*(Δω - Δω_C)

    All power in p.u., frequency deviation Δω = (ω - ω*) in rad/s.
    """

    # Column indices in the output array
    COL_OMEGA    = 0   # local frequency ω_j [rad/s]
    COL_DELTA    = 1   # rotor angle δ_j [rad]
    COL_P        = 2   # active power output p_j [p.u.]
    COL_PSTAR    = 3   # power setpoint p*_j [p.u.]
    COL_VREF     = 4   # voltage reference [p.u.]
    COL_VDC      = 5   # DC-link voltage [p.u.]
    COL_OMEGA_C  = 6   # COI frequency received [rad/s]
    COL_IF       = 7   # excitation current [p.u.]
    N_FEATURES   = 8

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        # Per-VSM parameters
        self.H = np.full(N_VSM, H_PU)       # inertia constant [s]
        self.D = np.full(N_VSM, 25.0)       # damping [p.u. power / (rad/s)]
        self.F = np.full(N_VSM, 10.0)        # COI friction [p.u. power / (rad/s)]
        self.p_setpoints = P_SETPOINTS.copy()

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def simulate_scenario(
        self,
        load_factors: np.ndarray | None = None,
        attack_fn=None,
    ) -> np.ndarray:
        """
        Run one 10-second scenario.

        Parameters
        ----------
        load_factors : (N_VSM,) array of p.u. load multipliers, or None for random.
        attack_fn : callable(t_idx, signals_dict) → signals_dict  or None.

        Returns
        -------
        recording : np.ndarray of shape (N_STEPS, N_VSM, N_FEATURES)
        """
        if load_factors is None:
            load_factors = self.rng.uniform(*LOAD_RANGE, size=N_VSM)

        # ── Initialise state ──────────────────
        omega = np.full(N_VSM, OMEGA_STAR)          # angular frequency [rad/s]
        delta = self.rng.uniform(0, 2 * np.pi, N_VSM)  # rotor angle
        i_f   = np.full(N_VSM, 0.0)                 # excitation current
        v_int = np.full(N_VSM, 0.0)                 # PI controller integral
        v_dc  = np.full(N_VSM, 1.0)                 # DC-link voltage
        v_dq_mag = np.full(N_VSM, V_REF_NOMINAL)    # terminal voltage magnitude

        # Load demand per VSM [p.u.]
        p_load_base = self.p_setpoints * load_factors
        
        # Smooth dynamic load fluctuations to simulate real grid noise
        load_noise = np.zeros((N_STEPS, N_VSM))
        for t in range(1, N_STEPS):
            load_noise[t] = 0.98 * load_noise[t-1] + 0.0005 * self.rng.standard_normal(N_VSM)
        p_load_dynamic = p_load_base * (1.0 + load_noise)

        recording = np.zeros((N_STEPS, N_VSM, self.N_FEATURES))

        for t in range(N_STEPS):
            p_load = p_load_dynamic[t]
            # ── COI frequency (Eq. 3) — from true state ──
            omega_C_true = self._compute_coi(omega)

            # ── Electrical power output (physical) ──
            # P_e = P_load * (V/V_nom)² + K_droop * (ω - ω*) / ω*
            # Voltage-power coupling: real power depends on terminal voltage
            # (P ∝ V² for constant impedance load model)
            delta_omega = omega - OMEGA_STAR
            v_ratio_sq = (v_dq_mag / V_REF_NOMINAL) ** 2
            p_electrical = p_load * v_ratio_sq + 2.0 * delta_omega / OMEGA_STAR

            # ── Build signals dict (pre-attack) ──
            signals = {
                "omega":     omega.copy(),
                "delta":     delta.copy(),
                "p_star":    p_load_base.copy(),      # control setpoint = base demand (no noise)
                "v_ref":     np.full(N_VSM, V_REF_NOMINAL),
                "omega_C":   np.full(N_VSM, omega_C_true),
                "v_dc":      v_dc.copy(),
                "v_dq_mag":  v_dq_mag.copy(),
                "i_f":       i_f.copy(),
            }

            # ── Apply attack ──────────────────
            if attack_fn is not None:
                signals = attack_fn(t, signals)

            # ── Record ────────────────────────
            recording[t, :, self.COL_OMEGA]   = omega  # record TRUE physical frequency
            recording[t, :, self.COL_DELTA]   = signals["delta"]
            recording[t, :, self.COL_P]       = p_electrical
            recording[t, :, self.COL_PSTAR]   = signals["p_star"]
            recording[t, :, self.COL_VREF]    = signals["v_ref"]
            recording[t, :, self.COL_VDC]     = signals["v_dc"]
            recording[t, :, self.COL_OMEGA_C] = signals["omega_C"]
            recording[t, :, self.COL_IF]      = signals["i_f"]

            # ── Swing equation (Euler integration) ──
            # Per-unit form:
            #   2H * dω/dt = (p*_control − P_e) * ω* + D*(ω* − ω_meas) + F*(ω_C − ω_meas)
            #
            # The ω* multiplier converts p.u. power to torque units.
            # D damping and F friction are computed by the digital controller 
            # using the local measured frequency (signals["omega"]). Spoofing
            # the frequency sensor creates an incorrect torque, causing 
            # realistic physical frequency deviations.

            delta_new = delta + DT * omega

            d_omega = (1.0 / (2.0 * self.H)) * (
                (signals["p_star"] - p_electrical) * OMEGA_STAR
                + self.D * (OMEGA_STAR - signals["omega"])
                + self.F * (signals["omega_C"] - signals["omega"])
            )
            omega_new = omega + DT * d_omega

            # ── Voltage controller (Eq. 4) ────
            v_err = signals["v_ref"] - v_dq_mag
            v_int_new = v_int + DT * v_err
            i_f_new = (KP / MF) * v_err + (KI / (A_FRAC * MF)) * v_int_new

            # Voltage dynamics — tracks reference with lag
            v_dq_mag_new = v_dq_mag + DT * 10.0 * (signals["v_ref"] - v_dq_mag)

            # ── Update true state ─────────────
            delta    = delta_new
            omega    = omega_new
            i_f      = i_f_new
            v_int    = v_int_new
            v_dq_mag = np.clip(v_dq_mag_new, 0.8, 1.2)  # physical voltage limits

            # ── Divergence check (replaces old ±5% clamp) ──
            # If frequency exceeds ±15% of nominal, scenario is unstable
            if np.any(omega < OMEGA_STAR * 0.85) or np.any(omega > OMEGA_STAR * 1.15):
                import logging
                logging.warning(
                    f"Simulation diverged at t={t}: "
                    f"omega range [{omega.min():.1f}, {omega.max():.1f}] rad/s"
                )
                return None  # caller should skip this scenario

        return recording

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _compute_coi(self, omega: np.ndarray) -> float:
        """
        Center-of-Inertia frequency (Eq. 3):
        ω_C = Σ(H_j * ω* * ω_j) / Σ(H_j * ω*)
        """
        numerator   = np.sum(self.H * OMEGA_STAR * omega)
        denominator = np.sum(self.H * OMEGA_STAR)
        return numerator / denominator


# ─────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running single normal scenario...")
    sim = VSMSimulator(seed=0)
    rec = sim.simulate_scenario()

    print(f"Recording shape: {rec.shape}")

    omega_all = rec[:, :, VSMSimulator.COL_OMEGA]
    freq_all  = omega_all / (2 * np.pi)
    print(f"Frequency range: [{freq_all.min():.4f}, {freq_all.max():.4f}] Hz")

    if 49.0 <= freq_all.min() and freq_all.max() <= 51.0:
        print("✓ Normal scenario frequency check PASSED")
    else:
        print("✗ Normal scenario frequency check FAILED")

    if np.any(np.isnan(rec)) or np.any(np.isinf(rec)):
        print("✗ NaN/Inf detected!")
    else:
        print("✓ No NaN/Inf values")
