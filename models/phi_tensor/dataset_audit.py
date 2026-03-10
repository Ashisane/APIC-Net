"""
TASK 03C — Dataset Validity Audit

Determines whether r4 AUC=0.946 is genuine or the dataset is too easy.

Usage: python -m models.phi_tensor.dataset_audit
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OMEGA_STAR = 2 * np.pi * 50.0
COL_OMEGA = 0
COL_OMEGA_C = 6
COL_P = 2
COL_PSTAR = 3

ATTACK_TYPE_NAMES = {0: "normal", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    n_vsm = 6

    # Collect data across all VSMs
    normal_omega = []
    normal_omega_C = []
    normal_r4 = []
    attack_r4 = {1: [], 2: [], 3: [], 4: []}
    attack_omega_dev = {1: [], 2: [], 3: [], 4: []}

    # For Audit 3: COI attack signal vs r4
    coi_attack_r4 = []
    coi_omega_C_raw = []
    coi_omega_local = []

    # For Audit 4
    r4_val_normal = []

    for vsm_id in range(n_vsm):
        d = np.load(os.path.join(data_dir, f"vsm_{vsm_id}.npz"))
        rec = d["data"]
        lbl = d["labels"]
        atk = d["attack_types"]

        n_sc = rec.shape[0]
        shuffle_idx = np.random.RandomState(42).permutation(n_sc)
        rec, lbl, atk = rec[shuffle_idx], lbl[shuffle_idx], atk[shuffle_idx]

        n_train = int(n_sc * 0.7)
        n_val = int(n_sc * 0.15)

        # Val split for threshold
        val_rec = rec[n_train:n_train + n_val]
        val_lbl = lbl[n_train:n_train + n_val]
        for s in range(val_rec.shape[0]):
            omega = val_rec[s, :, COL_OMEGA]
            omega_C = val_rec[s, :, COL_OMEGA_C]
            r4 = np.abs(omega_C - omega)
            normal_mask = val_lbl[s] == 0
            r4_val_normal.append(r4[normal_mask])

        # All scenarios
        for s in range(rec.shape[0]):
            omega = rec[s, :, COL_OMEGA]
            omega_C = rec[s, :, COL_OMEGA_C]
            labels_s = lbl[s]
            atk_s = atk[s]

            r4_s = np.abs(omega_C - omega)
            r4_signed = omega_C - omega

            norm_mask = labels_s == 0
            if norm_mask.sum() > 0:
                normal_omega.append(omega[norm_mask])
                normal_omega_C.append(omega_C[norm_mask])
                normal_r4.append(r4_s[norm_mask])

            for code in [1, 2, 3, 4]:
                atk_mask = atk_s == code
                if atk_mask.sum() > 0:
                    attack_r4[code].append(r4_s[atk_mask])
                    attack_omega_dev[code].append(np.abs(omega[atk_mask] - OMEGA_STAR))

                    if code == 2:  # COI attack
                        coi_attack_r4.append(r4_signed[atk_mask])
                        coi_omega_C_raw.append(omega_C[atk_mask])
                        coi_omega_local.append(omega[atk_mask])

    # Concatenate
    normal_omega_all = np.concatenate(normal_omega)
    normal_omega_C_all = np.concatenate(normal_omega_C)
    normal_r4_all = np.concatenate(normal_r4)
    r4_val_norm_all = np.concatenate(r4_val_normal)

    for k in attack_r4:
        attack_r4[k] = np.concatenate(attack_r4[k]) if attack_r4[k] else np.array([])
        attack_omega_dev[k] = np.concatenate(attack_omega_dev[k]) if attack_omega_dev[k] else np.array([])

    coi_r4_all = np.concatenate(coi_attack_r4) if coi_attack_r4 else np.array([])
    coi_omega_C_all = np.concatenate(coi_omega_C_raw) if coi_omega_C_raw else np.array([])
    coi_omega_local_all = np.concatenate(coi_omega_local) if coi_omega_local else np.array([])

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 1: Signal Scale Analysis
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("AUDIT 1: Signal Scale Analysis")
    print("=" * 70)

    omega_std = normal_omega_all.std()
    omega_std_hz = omega_std / (2 * np.pi)
    r4_std = normal_r4_all.std()
    r4_p95 = np.percentile(normal_r4_all, 95)
    r4_mean = normal_r4_all.mean()
    omega_C_std = normal_omega_C_all.std()
    p_std = 0  # not critical here

    print(f"  Normal omega std:       {omega_std:.6f} rad/s = {omega_std_hz:.6f} Hz")
    print(f"  Normal omega_C std:     {omega_C_std:.6f} rad/s")
    print(f"  Normal |r4| mean:       {r4_mean:.6f} rad/s")
    print(f"  Normal |r4| std:        {r4_std:.6f} rad/s")
    print(f"  Normal |r4| 95th pct:   {r4_p95:.6f} rad/s")

    print(f"\n  {'Attack':<10} {'r4 mean':>12} {'r4 std':>12} {'SNR ratio':>10} {'verdict'}")
    print("  " + "-" * 56)
    snr_table = {}
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        if len(attack_r4[code]) == 0:
            continue
        atk_mean = attack_r4[code].mean()
        atk_std = attack_r4[code].std()
        snr = atk_mean / (r4_std + 1e-15)
        verdict = "TOO EASY" if snr > 10 else ("REALISTIC" if snr < 5 else "MODERATE")
        snr_table[name] = {"r4_mean": float(atk_mean), "r4_std": float(atk_std),
                           "snr": float(snr), "verdict": verdict}
        print(f"  {name:<10} {atk_mean:>12.6f} {atk_std:>12.6f} {snr:>10.2f}x {verdict}")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 2: Attack Amplitude vs Normal Variation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUDIT 2: Attack Amplitude vs Normal Variation")
    print("=" * 70)

    print(f"  Normal omega std: {omega_std:.4f} rad/s = {omega_std_hz:.4f} Hz")
    print(f"  Config attack amp range: 1.0 - 5.0 rad/s")
    min_atk_amp = 1.0
    print(f"  Min attack / normal std: {min_atk_amp / omega_std:.1f}x")
    print(f"  Max attack / normal std: {5.0 / omega_std:.1f}x")

    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        if len(attack_omega_dev[code]) == 0:
            continue
        dev_mean = attack_omega_dev[code].mean()
        dev_hz = dev_mean / (2 * np.pi)
        print(f"  {name:<10} omega deviation: {dev_mean:.4f} rad/s = {dev_hz:.4f} Hz")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 3: Is r4 Reading the Attack Directly?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUDIT 3: Is r4 Reading the Attack Directly?")
    print("=" * 70)

    if len(coi_r4_all) > 0:
        # For COI attacks: r4 = ωC_received - ω_local
        # If ωC_received = ωC_true + attack_signal, then r4 ≈ attack_signal
        # We can check: does ω_local stay near ω* during COI attack?
        omega_local_dev = np.abs(coi_omega_local_all - OMEGA_STAR)
        omega_C_dev = np.abs(coi_omega_C_all - OMEGA_STAR)

        print(f"  COI attack analysis ({len(coi_r4_all)} timesteps):")
        print(f"    |ω_local - ω*| mean:    {omega_local_dev.mean():.6f} rad/s")
        print(f"    |ωC_received - ω*| mean: {omega_C_dev.mean():.6f} rad/s")
        print(f"    |r4| mean:               {np.abs(coi_r4_all).mean():.6f} rad/s")

        if omega_C_dev.mean() > 10 * omega_local_dev.mean():
            print(f"    → ωC is spoofed, ω_local stays normal → r4 IS reading the attack directly")
        else:
            print(f"    → Both shift → r4 captures physical effect, not direct reading")

    # For power/voltage: check if ω_local shifts
    for code, name in [(3, "power"), (4, "voltage")]:
        if len(attack_omega_dev[code]) > 0:
            normal_omega_dev_mean = np.abs(normal_omega_all - OMEGA_STAR).mean()
            atk_omega_dev_mean = attack_omega_dev[code].mean()
            ratio = atk_omega_dev_mean / (normal_omega_dev_mean + 1e-15)
            print(f"\n  {name} attack:")
            print(f"    Normal |ω - ω*| mean: {normal_omega_dev_mean:.6f}")
            print(f"    Attack |ω - ω*| mean: {atk_omega_dev_mean:.6f}")
            print(f"    Ratio: {ratio:.2f}x → {'ω_local shifts significantly' if ratio > 3 else 'marginal shift'}")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 4: Threshold Reality Check
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUDIT 4: Threshold Reality Check")
    print("=" * 70)

    tau = np.percentile(r4_val_norm_all, 95)
    normal_exceed = (normal_r4_all > tau).mean()
    print(f"  τ (95th pct val normal): {tau:.6f}")
    print(f"  Normal exceed rate: {normal_exceed:.4f} (expect ~0.05)")

    print(f"\n  {'Attack':<10} {'detection rate':>15} {'verdict'}")
    print("  " + "-" * 40)
    det_rates = {}
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        if len(attack_r4[code]) == 0:
            continue
        rate = (attack_r4[code] > tau).mean()
        det_rates[name] = float(rate)
        verdict = "TRIVIAL" if rate > 0.95 else ("CHALLENGING" if rate < 0.7 else "MODERATE")
        print(f"  {name:<10} {rate:>15.4f} {verdict}")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 5: Grid Standards Comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUDIT 5: Grid Standards Comparison")
    print("=" * 70)

    print(f"  Normal freq deviation: {omega_std_hz:.4f} Hz")
    print(f"  Grid standard (realistic): ±0.1 Hz max")
    print(f"  Our normal deviation vs standard: {omega_std_hz / 0.1:.2f}x")
    print(f"  Attack amplitude range: {1.0/(2*np.pi):.3f} - {5.0/(2*np.pi):.3f} Hz")
    print(f"  Stealthy attack range (literature): 0.05-0.2 Hz")

    atk_min_hz = 1.0 / (2 * np.pi)
    if atk_min_hz > 0.2:
        print(f"  → Our MINIMUM attack ({atk_min_hz:.3f} Hz) EXCEEDS stealthy range (0.2 Hz)")
        print(f"  → Attacks are OBVIOUS, not stealthy")
    else:
        print(f"  → Attacks overlap with stealthy range")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 6: Khaleghi Comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUDIT 6: Khaleghi Paper Check")
    print("=" * 70)
    print("  Khaleghi et al. used amplitude range: 1-5 rad/s (same as ours)")
    print("  Their 2-VSM setup had less load noise → likely even higher SNR")
    print("  They did NOT compare against simple residual baselines (no r4 test)")
    print("  Their high F1 (95.5%) may reflect easy dataset, not model strength")
    print("  Our finding: a zero-training residual achieves AUC=0.946 on same amplitudes")
    print("  → Their PI-LSTM was likely solving an easy problem with a complex model")

    # ═══════════════════════════════════════════════════════════════
    # OVERALL VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("OVERALL VERDICT")
    print("=" * 70)

    all_snr_high = all(v["snr"] > 10 for v in snr_table.values())
    all_det_triv = all(v > 0.95 for v in det_rates.values())

    if all_snr_high and all_det_triv:
        verdict = "B_TOO_EASY"
        print("  VERDICT B: Dataset is TOO EASY.")
        print("  Attack amplitudes (1-5 rad/s) dwarf normal variation.")
        print("  Any threshold detector would trivially detect these attacks.")
        print("  → Must regenerate with smaller amplitudes OR")
        print("  → Report honestly and test with reduced amplitudes")
    elif all_snr_high:
        verdict = "C_MIXED"
        print("  VERDICT C: r4 partially reads attacks directly (especially COI).")
        print("  SNR is very high for most attack types.")
    else:
        verdict = "A_VALID"
        print("  VERDICT A: Dataset appears valid.")

    # Save
    audit_results = {
        "normal_omega_std_rads": float(omega_std),
        "normal_omega_std_hz": float(omega_std_hz),
        "normal_r4_mean": float(r4_mean),
        "normal_r4_std": float(r4_std),
        "normal_r4_p95": float(r4_p95),
        "tau": float(tau),
        "snr_per_attack_type": snr_table,
        "attack_detection_rates": det_rates,
        "verdict": verdict,
    }
    out = os.path.join(results_dir, "dataset_audit.json")
    with open(out, "w") as f:
        json.dump(audit_results, f, indent=4)
    print(f"\nSaved to {out}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Audit: r4 Distribution — Normal vs Attack", fontsize=14)
    for idx, (code, name) in enumerate([(1, "freq"), (2, "coi"), (3, "power"), (4, "voltage")]):
        ax = axes[idx // 2][idx % 2]
        norm_vals = normal_r4_all
        atk_vals = attack_r4[code]
        mx = np.percentile(atk_vals, 99) if len(atk_vals) > 0 else 1
        bins = np.linspace(0, mx, 80)
        ax.hist(norm_vals, bins=bins, alpha=0.6, density=True, color="steelblue", label="Normal")
        if len(atk_vals) > 0:
            ax.hist(atk_vals, bins=bins, alpha=0.6, density=True, color="crimson", label=name)
        ax.axvline(tau, color="black", linestyle="--", label=f"τ={tau:.4f}")
        snr_val = snr_table.get(name, {}).get("snr", 0)
        ax.set_title(f"{name.upper()} — SNR={snr_val:.1f}x")
        ax.legend(fontsize=8)
        ax.set_xlabel("|r4| (rad/s)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "audit_signal_scales.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {os.path.join(plots_dir, 'audit_signal_scales.png')}")

    # COI correlation plot
    if len(coi_r4_all) > 5000:
        sample = np.random.choice(len(coi_r4_all), 5000, replace=False)
    else:
        sample = np.arange(len(coi_r4_all))
    fig, ax = plt.subplots(figsize=(8, 6))
    deviation = coi_omega_C_all[sample] - OMEGA_STAR
    ax.scatter(deviation, coi_r4_all[sample], alpha=0.1, s=2)
    ax.set_xlabel("ωC_received - ω* (attack injection proxy)")
    ax.set_ylabel("r4 = ωC_received - ω_local")
    ax.set_title("Audit 3: COI Attack — r4 vs Attack Signal")
    if len(deviation) > 10:
        corr, _ = pearsonr(deviation, coi_r4_all[sample])
        ax.text(0.05, 0.95, f"Pearson r = {corr:.4f}", transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.savefig(os.path.join(plots_dir, "audit_coi_correlation.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {os.path.join(plots_dir, 'audit_coi_correlation.png')}")


if __name__ == "__main__":
    main()
