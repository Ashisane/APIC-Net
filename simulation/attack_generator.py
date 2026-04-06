"""
attack_generator.py — Generates randomised attack parameters and injects
attacks into VSM simulation signals.

Supports 4 attack types × 4 waveforms as defined in DATASET_SPEC.md.

V2 changes (TASK 04 — DATASET_FIX.md + Option A COI fix):
  - freq: unchanged (1-5 rad/s, SNR=1.4× — realistic)
  - power: calibrated to 1.5-4× normal_omega_std (0.285-0.76 rad/s)
  - coi: UPSTREAM spoof — corrupts VSM j's OUTGOING omega to aggregator
         (not what j receives back). r4 at j is no longer a direct readout.
         Other VSMs detect the inconsistency via their r4.
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
# COI spoof range accounts for 1/6 dilution in 6-VSM aggregation.
# omega_C shift at other VSMs = spoof/N_VSM, so spoof = N_VSM × target_effect.
# target_effect = 1.5-4.0 × NORMAL_R4_STD = 0.78-2.08 rad/s
# spoof = 6 × (0.78-2.08) = 4.68-12.48 rad/s
COI_SPOOF_RANGE  = (6 * 1.5 * NORMAL_R4_STD,  6 * 4.0 * NORMAL_R4_STD)  # 4.68–12.48


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
        # V2 Option A: upstream spoof amplitude (same calibration range)
        amplitude = rng.uniform(*COI_SPOOF_RANGE)
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
    Create injection closure(s) for one attack scenario.

    Returns
    -------
    (attack_fn, coi_spoof_fn) tuple:
      - Non-COI: (callable(t_idx, signals_dict)->signals_dict, None)
      - COI:     (None, callable(t_idx, omega_array)->omega_array)

    Option A — COI upstream spoof:
        coi_spoof_fn corrupts VSM j's omega BEFORE COI aggregation.
        All other VSMs receive the corrupted omega_C.
        r4 at VSM j is NOT a direct readout of the injected bias
        (diluted by 5/6 of true omegas in aggregation).
        Label stays on VSM j (the compromised node).
    """
    attack_type = params["attack_type"]
    target_vsm  = params["target_vsm"]

    if attack_type == "coi":
        # ── Option A: COI upstream spoof ──
        spoof_amplitude = params["amplitude"]
        trigger_step    = params["trigger_step"]
        end_step        = params["end_step"]
        duration_steps  = max(end_step - trigger_step, 1)

        # Pre-generate random walk clipped to spoof_amplitude
        rng_walk = np.random.default_rng(
            hash((trigger_step, target_vsm, int(spoof_amplitude * 1000))) % (2**31)
        )
        steps = rng_walk.standard_normal(duration_steps) * 0.18
        bias  = np.cumsum(steps)
        bias  = np.clip(bias, -spoof_amplitude, spoof_amplitude)

        def coi_spoof_fn(t_idx: int, omega_for_coi: np.ndarray) -> np.ndarray:
            """Corrupt VSM j's omega BEFORE COI aggregation."""
            if trigger_step <= t_idx < end_step:
                idx = t_idx - trigger_step
                omega_for_coi = omega_for_coi.copy()
                omega_for_coi[target_vsm] += bias[idx]
            return omega_for_coi

        return None, coi_spoof_fn
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

        return attack_fn, None


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
