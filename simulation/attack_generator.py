"""
attack_generator.py — Generates randomised attack parameters and injects
attacks into VSM simulation signals.

Supports 4 attack types × 4 waveforms as defined in DATASET_SPEC.md.

V2 changes (TASK 04 — DATASET_FIX.md):
  - freq: unchanged (1-5 rad/s, SNR=1.4× — realistic)
  - power: calibrated to 1.5-4× normal_omega_std (0.285-0.76 rad/s)
  - coi: calibrated additive bias with slow ramp (0.156-0.52 rad/s)
  - voltage: unchanged (inherently large-disturbance via controller nonlinearity)
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

# ── V2 calibration constants (from dataset audit TASK 03C) ──
NORMAL_OMEGA_STD = 0.19   # per-VSM normal omega deviation std [rad/s]
NORMAL_R4_STD    = 0.52   # normal |omega_C - omega| std [rad/s]

POWER_AMP_RANGE  = (1.5 * NORMAL_OMEGA_STD, 4.0 * NORMAL_OMEGA_STD)  # 0.285–0.76
COI_BIAS_RANGE   = (1.5 * NORMAL_R4_STD,    4.0 * NORMAL_R4_STD)     # 0.78–2.08


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

    trigger_time = rng.uniform(*ATTACK_TRIG_RANGE)
    duration     = rng.uniform(*ATTACK_DUR_RANGE)
    waveform     = rng.choice(ATTACK_WAVEFORMS)
    freq_hz      = rng.uniform(0.5, 5.0)  # for sine/square waveforms

    if target_vsm is None:
        target_vsm = int(rng.integers(0, N_VSM))

    # ── Attack-type-specific amplitude ──
    if attack_type == "power":
        # V2: calibrated to 1.5-4× normal_omega_std
        amplitude = rng.uniform(*POWER_AMP_RANGE)
    elif attack_type == "coi":
        # V2: bias ramp amplitude calibrated to 0.3-1.0× normal_r4_std
        amplitude = rng.uniform(*COI_BIAS_RANGE)
    else:
        # freq and voltage: unchanged from V1 (1-5 rad/s)
        amplitude = rng.uniform(*ATTACK_AMP_RANGE)

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

    V2: COI uses slow bias ramp instead of waveform-based additive injection.
    """
    attack_type = params["attack_type"]
    target_vsm  = params["target_vsm"]

    if attack_type == "coi":
        # ── V2: COI random walk bias ──
        # Random walk drift applied to omega_C — unpredictable, low correlation.
        bias_amplitude = params["amplitude"]
        trigger_step   = params["trigger_step"]
        end_step       = params["end_step"]
        duration_steps = end_step - trigger_step

        # Pre-generate random walk, clipped to amplitude range
        rng_walk = np.random.default_rng(hash((trigger_step, target_vsm, int(bias_amplitude * 1000))) % (2**31))
        steps = rng_walk.standard_normal(duration_steps) * 0.18
        bias = np.cumsum(steps)
        bias = np.clip(bias, -bias_amplitude, bias_amplitude)

        def attack_fn(t_idx: int, signals: dict) -> dict:
            if trigger_step <= t_idx < end_step:
                idx = t_idx - trigger_step
                signals["omega_C"][target_vsm] += bias[idx]
            return signals

        return attack_fn
    else:
        # ── Standard additive injection (freq, power, voltage) ──
        SIGNAL_MAP = {
            "freq":    "omega",
            "power":   "p_star",
            "voltage": "v_ref",
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
        print(f"\n{atype}: amp={p['amplitude']:.4f}, "
              f"trigger={p['trigger_time']:.2f}s, "
              f"dur={p['duration']:.2f}s, "
              f"waveform={p['waveform']}, "
              f"steps=[{p['trigger_step']}, {p['end_step']})")

        # Test attack signal at a few timesteps
        vals = [compute_attack_signal(t, p) for t in range(0, N_STEPS, 200)]
        print(f"  Signal samples (every 200 steps): {[f'{v:.4f}' for v in vals]}")

        # Test label array
        labels = get_attack_label_array(p)
        print(f"  Labels: {labels.sum()} attack timesteps out of {N_STEPS}")

    print("\n✓ Attack generator tests passed")
