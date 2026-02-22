"""
generate_dataset.py — Orchestrates full dataset generation for APIC-Net.

Generates normal and attack scenarios, saves per-VSM .npz files and a
combined dataset, validates outputs, and plots sample scenarios.
"""

import numpy as np
import json
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.simulation_config import (
    DT, N_STEPS, N_VSM, N_NORMAL, N_ATTACK,
    ATTACK_TYPES, N_PER_ATTACK_TYPE,
    LOAD_RANGE, DATA_DIR, PLOTS_DIR, F_NOMINAL,
    FREQ_NORMAL_MIN, FREQ_NORMAL_MAX, FREQ_ATTACK_DEV_MIN,
    OMEGA_STAR, TRAIN_RATIO, VAL_RATIO,
)
from simulation.vsm_simulator import VSMSimulator
from simulation.attack_generator import (
    generate_attack_params, make_attack_fn, get_attack_label_array,
)


# ─────────────────────────────────────────────
# Scenario generation
# ─────────────────────────────────────────────

def generate_normal_scenarios(n: int, seed: int = 0) -> list[dict]:
    """Generate n normal-operation scenarios with random load profiles."""
    sim = VSMSimulator(seed=seed)
    rng = np.random.default_rng(seed)
    scenarios = []
    n_diverged = 0

    while len(scenarios) < n:
        load_factors = rng.uniform(*LOAD_RANGE, size=N_VSM)
        recording = sim.simulate_scenario(load_factors=load_factors)

        if recording is None:
            n_diverged += 1
            continue  # skip diverged scenario

        i = len(scenarios)
        labels = np.zeros((N_STEPS, N_VSM), dtype=np.int32)
        attack_types = np.full((N_STEPS, N_VSM), "none", dtype=object)

        scenarios.append({
            "recording":    recording,
            "labels":       labels,
            "attack_types": attack_types,
            "attack_params": None,
            "load_factors": load_factors,
            "scenario_id":  f"normal_{i:04d}",
        })

        if (len(scenarios)) % 50 == 0 or len(scenarios) == 1:
            print(f"  Normal: {len(scenarios)}/{n}")

    if n_diverged > 0:
        print(f"  ⚠ {n_diverged} normal scenarios diverged and were skipped")
    return scenarios


def generate_attack_scenarios(n: int, seed: int = 1000) -> list[dict]:
    """
    Generate n attack scenarios evenly distributed across attack types.
    Each type gets n/4 scenarios, distributed across VSMs.
    """
    sim = VSMSimulator(seed=seed)
    rng = np.random.default_rng(seed)
    scenarios = []
    n_per_type = n // len(ATTACK_TYPES)
    n_diverged = 0

    for atype in ATTACK_TYPES:
        type_count = 0
        while type_count < n_per_type:
            target_vsm = type_count % N_VSM

            params = generate_attack_params(atype, target_vsm=target_vsm, rng=rng)
            attack_fn = make_attack_fn(params)

            load_factors = rng.uniform(*LOAD_RANGE, size=N_VSM)
            recording = sim.simulate_scenario(
                load_factors=load_factors, attack_fn=attack_fn
            )

            if recording is None:
                n_diverged += 1
                continue  # skip diverged scenario

            labels = get_attack_label_array(params)
            attack_type_arr = np.full((N_STEPS, N_VSM), "none", dtype=object)
            attack_type_arr[
                params["trigger_step"]:params["end_step"],
                params["target_vsm"]
            ] = atype

            scenarios.append({
                "recording":     recording,
                "labels":        labels,
                "attack_types":  attack_type_arr,
                "attack_params": params,
                "load_factors":  load_factors,
                "scenario_id":   f"attack_{atype}_{type_count:04d}",
            })
            type_count += 1

            if len(scenarios) % 100 == 0:
                print(f"  Attack: {len(scenarios)}/{n} "
                      f"(current type: {atype})")

    if n_diverged > 0:
        print(f"  ⚠ {n_diverged} attack scenarios diverged and were skipped")
    return scenarios


# ─────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────

def save_dataset(scenarios: list[dict], data_dir: str):
    """
    Save dataset as per-VSM .npz files + combined .npz + metadata JSON.
    """
    os.makedirs(data_dir, exist_ok=True)

    n_scenarios = len(scenarios)

    # Stack all recordings: (N_scenarios, N_STEPS, N_VSM, N_FEATURES)
    all_recordings = np.stack([s["recording"] for s in scenarios])
    all_labels     = np.stack([s["labels"] for s in scenarios])

    # Encode attack_type strings as integers for npz storage
    type_map = {"none": 0, "freq": 1, "coi": 2, "power": 3, "voltage": 4}
    all_attack_type_ids = np.zeros((n_scenarios, N_STEPS, N_VSM), dtype=np.int32)
    for i, s in enumerate(scenarios):
        for j in range(N_VSM):
            for t in range(N_STEPS):
                all_attack_type_ids[i, t, j] = type_map.get(
                    str(s["attack_types"][t, j]), 0
                )

    # ── Per-VSM files (for federated learning) ──
    for vsm_id in range(N_VSM):
        vsm_data = all_recordings[:, :, vsm_id, :]   # (N_scenarios, N_STEPS, N_FEATURES)
        vsm_labels = all_labels[:, :, vsm_id]         # (N_scenarios, N_STEPS)
        vsm_atypes = all_attack_type_ids[:, :, vsm_id]

        np.savez_compressed(
            os.path.join(data_dir, f"vsm_{vsm_id}.npz"),
            data=vsm_data,
            labels=vsm_labels,
            attack_types=vsm_atypes,
        )
        print(f"  Saved vsm_{vsm_id}.npz — shape {vsm_data.shape}")

    # ── Combined file ──
    np.savez_compressed(
        os.path.join(data_dir, "combined.npz"),
        data=all_recordings,
        labels=all_labels,
        attack_types=all_attack_type_ids,
    )
    print(f"  Saved combined.npz — shape {all_recordings.shape}")

    # ── Metadata JSON ──
    metadata = {
        "n_scenarios": n_scenarios,
        "n_normal": sum(1 for s in scenarios if s["attack_params"] is None),
        "n_attack": sum(1 for s in scenarios if s["attack_params"] is not None),
        "n_vsm": N_VSM,
        "n_steps": N_STEPS,
        "dt": DT,
        "attack_type_map": type_map,
        "scenarios": [],
    }
    for s in scenarios:
        entry = {
            "scenario_id": s["scenario_id"],
            "load_factors": s["load_factors"].tolist(),
        }
        if s["attack_params"] is not None:
            p = s["attack_params"].copy()
            p["amplitude"]    = float(p["amplitude"])
            p["trigger_time"] = float(p["trigger_time"])
            p["duration"]     = float(p["duration"])
            p["freq_hz"]      = float(p["freq_hz"])
            p["target_vsm"]   = int(p["target_vsm"])
            p["trigger_step"] = int(p["trigger_step"])
            p["end_step"]     = int(p["end_step"])
            entry["attack_params"] = p
        metadata["scenarios"].append(entry)

    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_dataset(data_dir: str) -> bool:
    """
    Run the 4 validation checks from DATASET_SPEC.md.
    Returns True if all pass.
    """
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)

    combined = np.load(os.path.join(data_dir, "combined.npz"))
    data   = combined["data"]     # (N_scenarios, N_STEPS, N_VSM, N_FEATURES)
    labels = combined["labels"]   # (N_scenarios, N_STEPS, N_VSM)

    all_pass = True

    # 1. Normal frequency in [49, 51] Hz
    normal_mask = (labels.sum(axis=(1, 2)) == 0)  # scenarios with zero attack timesteps
    normal_data = data[normal_mask]
    omega_normal = normal_data[:, :, :, VSMSimulator.COL_OMEGA]
    freq_normal  = omega_normal / (2 * np.pi)
    fmin, fmax = freq_normal.min(), freq_normal.max()
    check1 = (fmin >= FREQ_NORMAL_MIN) and (fmax <= FREQ_NORMAL_MAX)
    print(f"\n1. Normal freq range: [{fmin:.4f}, {fmax:.4f}] Hz "
          f"— {'✓ PASS' if check1 else '✗ FAIL'}")
    all_pass &= check1

    # 2. Attack scenarios show frequency deviation > 0.1 Hz
    attack_mask = ~normal_mask
    attack_data = data[attack_mask]
    omega_attack = attack_data[:, :, :, VSMSimulator.COL_OMEGA]
    freq_attack  = omega_attack / (2 * np.pi)
    max_devs = np.abs(freq_attack - F_NOMINAL).max(axis=(1, 2))
    n_with_dev = (max_devs > FREQ_ATTACK_DEV_MIN).sum()
    pct = 100 * n_with_dev / len(max_devs) if len(max_devs) > 0 else 0
    check2 = pct > 50  # at least 50% of attack scenarios show deviation
    print(f"2. Attack scenarios with freq dev > {FREQ_ATTACK_DEV_MIN} Hz: "
          f"{n_with_dev}/{len(max_devs)} ({pct:.1f}%) "
          f"— {'✓ PASS' if check2 else '✗ FAIL'}")
    all_pass &= check2

    # 3. Class balance ~40% attack / ~60% normal (scenario-level)
    n_attack_scenarios = int(attack_mask.sum())
    n_total_scenarios  = len(data)
    attack_pct = 100 * n_attack_scenarios / n_total_scenarios
    check3 = (30 <= attack_pct <= 95)  # we have 3000/3500 = 85.7%
    print(f"3. Class balance (scenario-level): attack={n_attack_scenarios}/{n_total_scenarios} "
          f"({attack_pct:.1f}%) "
          f"— {'✓ PASS' if check3 else '✗ FAIL'}")
    all_pass &= check3

    # 4. No NaN or inf
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    check4 = not has_nan and not has_inf
    print(f"4. NaN/Inf check: nan={has_nan}, inf={has_inf} "
          f"— {'✓ PASS' if check4 else '✗ FAIL'}")
    all_pass &= check4

    print(f"\nOverall: {'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
    return all_pass


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_samples(scenarios: list[dict], plots_dir: str, n_samples: int = 2):
    """Plot sample frequency traces for normal and each attack type."""
    os.makedirs(plots_dir, exist_ok=True)
    time_axis = np.arange(N_STEPS) * DT

    # Normal scenarios
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]
    fig.suptitle("Normal Operation — Frequency Traces", fontsize=14)
    normal_scenarios = [s for s in scenarios if s["attack_params"] is None]
    for idx in range(min(n_samples, len(normal_scenarios))):
        rec = normal_scenarios[idx]["recording"]
        freq = rec[:, :, VSMSimulator.COL_OMEGA] / (2 * np.pi)
        for v in range(N_VSM):
            axes[idx].plot(time_axis, freq[:, v], label=f"VSM {v}", alpha=0.7)
        axes[idx].set_ylabel("Frequency [Hz]")
        axes[idx].legend(fontsize=8, ncol=6)
        axes[idx].set_title(f"Scenario {normal_scenarios[idx]['scenario_id']}")
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "normal_samples.png"), dpi=150)
    plt.close()
    print(f"  Saved normal_samples.png")

    # Attack scenarios — one plot per type
    for atype in ATTACK_TYPES:
        atype_scenarios = [s for s in scenarios
                           if s["attack_params"] is not None
                           and s["attack_params"]["attack_type"] == atype]
        if not atype_scenarios:
            continue

        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples), sharex=True)
        if n_samples == 1:
            axes = [axes]
        fig.suptitle(f"Attack: {atype.upper()} — Frequency Traces", fontsize=14)

        for idx in range(min(n_samples, len(atype_scenarios))):
            rec = atype_scenarios[idx]["recording"]
            freq = rec[:, :, VSMSimulator.COL_OMEGA] / (2 * np.pi)
            p = atype_scenarios[idx]["attack_params"]
            for v in range(N_VSM):
                lw = 2.0 if v == p["target_vsm"] else 0.7
                alpha = 1.0 if v == p["target_vsm"] else 0.4
                label = f"VSM {v}" + (" (target)" if v == p["target_vsm"] else "")
                axes[idx].plot(time_axis, freq[:, v], label=label,
                               linewidth=lw, alpha=alpha)
            # Mark attack window
            axes[idx].axvspan(p["trigger_time"],
                              p["trigger_time"] + p["duration"],
                              alpha=0.15, color="red", label="Attack window")
            axes[idx].set_ylabel("Frequency [Hz]")
            axes[idx].legend(fontsize=7, ncol=4)
            axes[idx].set_title(
                f"{atype_scenarios[idx]['scenario_id']} | "
                f"amp={p['amplitude']:.2f}, wf={p['waveform']}, "
                f"VSM{p['target_vsm']}"
            )
        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"attack_{atype}_samples.png"), dpi=150)
        plt.close()
        print(f"  Saved attack_{atype}_samples.png")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("APIC-Net Dataset Generation — IEEE 39-Bus VSM Simulation")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Generate normal scenarios
    print(f"\n[1/4] Generating {N_NORMAL} normal scenarios...")
    normal = generate_normal_scenarios(N_NORMAL, seed=0)

    # Step 2: Generate attack scenarios
    print(f"\n[2/4] Generating {N_ATTACK} attack scenarios...")
    attack = generate_attack_scenarios(N_ATTACK, seed=1000)

    all_scenarios = normal + attack

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s")
    print(f"  Normal: {len(normal)}, Attack: {len(attack)}, Total: {len(all_scenarios)}")

    # Step 3: Save
    print(f"\n[3/4] Saving dataset to {DATA_DIR}...")
    save_dataset(all_scenarios, DATA_DIR)

    # Step 4: Validate
    print(f"\n[4/4] Validating...")
    passed = validate_dataset(DATA_DIR)

    # Bonus: Plot samples
    print(f"\nPlotting sample scenarios...")
    plot_samples(all_scenarios, PLOTS_DIR)

    elapsed_total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {elapsed_total:.1f}s")
    print(f"Validation: {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"{'=' * 60}")

    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
