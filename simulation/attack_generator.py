"""
attack_generator.py — Generates randomised attack parameters and injects
attacks into VSM simulation signals.

Supports 4 attack types × 4 waveforms as defined in DATASET_SPEC.md.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.simulation_config import (
    DT, N_STEPS, T_TOTAL,
    ATTACK_TYPES, ATTACK_WAVEFORMS,
    ATTACK_AMP_RANGE, ATTACK_TRIG_RANGE, ATTACK_DUR_RANGE,
    N_VSM,
)


def generate_attack_params(
    attack_type: str,
    target_vsm: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Generate randomised attack parameters for one scenario.

    Parameters
    ----------
    attack_type : one of 'freq', 'coi', 'power', 'voltage'
    target_vsm  : VSM index (0–5), or None for random
    rng         : numpy RNG instance

    Returns
    -------
    dict with keys: attack_type, target_vsm, amplitude, trigger_time,
                    duration, waveform, trigger_step, end_step, freq_hz
    """
    if rng is None:
        rng = np.random.default_rng()

    assert attack_type in ATTACK_TYPES, f"Unknown attack type: {attack_type}"

    amplitude    = rng.uniform(*ATTACK_AMP_RANGE)
    trigger_time = rng.uniform(*ATTACK_TRIG_RANGE)
    duration     = rng.uniform(*ATTACK_DUR_RANGE)
    waveform     = rng.choice(ATTACK_WAVEFORMS)
    freq_hz      = rng.uniform(0.5, 5.0)  # for sine/square waveforms

    if target_vsm is None:
        target_vsm = int(rng.integers(0, N_VSM))

    # Precompute step indices for efficiency
    trigger_step = int(trigger_time / DT)
    end_step     = min(int((trigger_time + duration) / DT), N_STEPS)

    return {
        "attack_type":  attack_type,
        "target_vsm":   target_vsm,
        "amplitude":    amplitude,
        "trigger_time": trigger_time,
        "duration":     duration,
        "waveform":     waveform,
        "freq_hz":      freq_hz,
        "trigger_step": trigger_step,
        "end_step":     end_step,
    }


def compute_attack_signal(t_idx: int, params: dict) -> float:
    """
    Evaluate the attack perturbation signal at timestep t_idx.

    Returns 0.0 outside the active attack window.
    """
    if t_idx < params["trigger_step"] or t_idx >= params["end_step"]:
        return 0.0

    t = t_idx * DT  # absolute time [s]
    A = params["amplitude"]
    t_start = params["trigger_time"]
    duration = params["duration"]
    wf = params["waveform"]
    f = params["freq_hz"]

    if wf == "constant":
        return A
    elif wf == "sine":
        return A * np.sin(2 * np.pi * f * (t - t_start))
    elif wf == "square":
        return A * np.sign(np.sin(2 * np.pi * f * (t - t_start)))
    elif wf == "ramp":
        progress = (t - t_start) / duration if duration > 0 else 1.0
        return A * min(progress, 1.0)
    else:
        raise ValueError(f"Unknown waveform: {wf}")


def make_attack_fn(params: dict):
    """
    Create a closure that injects attacks into simulation signals.

    Returns a function with signature: (t_idx, signals_dict) → signals_dict
    that can be passed directly to VSMSimulator.simulate_scenario().
    """
    attack_type = params["attack_type"]
    target_vsm  = params["target_vsm"]

    # Map attack types to signal keys
    SIGNAL_MAP = {
        "freq":    "omega",      # corrupts ω_j
        "coi":     "omega_C",    # corrupts ω_C received
        "power":   "p_star",     # corrupts p*_j
        "voltage": "v_ref",      # corrupts v_ref
    }
    signal_key = SIGNAL_MAP[attack_type]

    def attack_fn(t_idx: int, signals: dict) -> dict:
        a = compute_attack_signal(t_idx, params)
        if a != 0.0:
            signals[signal_key][target_vsm] += a
        return signals

    return attack_fn


def get_attack_label_array(params: dict) -> np.ndarray:
    """
    Generate per-timestep labels for one attack scenario.

    Returns
    -------
    labels : (N_STEPS, N_VSM) int array — 0=normal, 1=attack
    """
    labels = np.zeros((N_STEPS, N_VSM), dtype=np.int32)
    target = params["target_vsm"]
    labels[params["trigger_step"]:params["end_step"], target] = 1
    return labels


# ─────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    for atype in ATTACK_TYPES:
        p = generate_attack_params(atype, target_vsm=0, rng=rng)
        print(f"\n{atype}: amp={p['amplitude']:.2f}, "
              f"trigger={p['trigger_time']:.2f}s, "
              f"dur={p['duration']:.2f}s, "
              f"waveform={p['waveform']}, "
              f"steps=[{p['trigger_step']}, {p['end_step']})")

        # Test attack signal at a few timesteps
        vals = [compute_attack_signal(t, p) for t in range(0, N_STEPS, 200)]
        print(f"  Signal samples (every 200 steps): {[f'{v:.2f}' for v in vals]}")

        # Test label array
        labels = get_attack_label_array(p)
        print(f"  Labels: {labels.sum()} attack timesteps out of {N_STEPS}")

    print("\n✓ Attack generator tests passed")
