"""
Φ Tensor — Signal Check (Step 3, TASK 03)

Checks both off_diag_std (Φ matrix) AND raw |r4| (COI residual)
for separation between normal and attack data.

Usage: python -m models.phi_tensor.signal_check
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .phi_tensor import compute_phi_for_scenario
from .features import extract_features_single, FEATURE_NAMES

ATTACK_TYPE_NAMES = {0: "normal", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    n_vsm = 6
    n_scenarios_to_check = 50

    # Collect features for normal vs each attack type
    feats_by_type = {k: [] for k in range(5)}

    print("Φ Tensor Signal Check (with raw residuals)")
    print("=" * 60)

    for vsm_id in range(n_vsm):
        print(f"\nProcessing VSM {vsm_id}...")
        d = np.load(os.path.join(data_dir, f"vsm_{vsm_id}.npz"))
        recording = d["data"]
        labels = d["labels"]
        atk_types = d["attack_types"]

        n_scenarios = recording.shape[0]
        shuffle_idx = np.random.RandomState(42).permutation(n_scenarios)
        recording = recording[shuffle_idx]
        labels = labels[shuffle_idx]
        atk_types = atk_types[shuffle_idx]

        n_train = int(n_scenarios * 0.7)
        n_val = int(n_scenarios * 0.15)
        test_rec = recording[n_train + n_val:]
        test_lbl = labels[n_train + n_val:]
        test_atk = atk_types[n_train + n_val:]

        n_check = min(n_scenarios_to_check, test_rec.shape[0])

        for s in range(n_check):
            Phi, residuals = compute_phi_for_scenario(test_rec[s])
            lbl_s = test_lbl[s, 1:]
            atk_s = test_atk[s, 1:]

            for t in range(Phi.shape[0]):
                feats = extract_features_single(Phi[t], residuals[t])
                lbl_t = lbl_s[t]
                atk_t = atk_s[t]

                code = 0 if lbl_t == 0 else int(atk_t)
                feats_by_type[code].append(feats)

        print(f"  Checked {n_check} scenarios")

    # Convert to arrays
    for k in feats_by_type:
        feats_by_type[k] = np.array(feats_by_type[k]) if feats_by_type[k] else np.zeros((0, len(FEATURE_NAMES)))

    normal_feats = feats_by_type[0]

    # ── Report separation for key features ──
    print("\n" + "=" * 85)
    print("SIGNAL CHECK: Feature separation (attack mean / normal mean)")
    print("=" * 85)

    key_features = [
        (0, "off_diag_std"),
        (5, "max_off_diag"),
        (6, "abs_r1"),
        (9, "abs_r4"),
    ]

    check_results = {}
    for feat_idx, feat_name in key_features:
        nm = normal_feats[:, feat_idx].mean()
        print(f"\n  {feat_name}:")
        feat_check = {}
        for code in [1, 2, 3, 4]:
            name = ATTACK_TYPE_NAMES[code]
            atk_feats = feats_by_type[code]
            if len(atk_feats) == 0:
                continue
            am = atk_feats[:, feat_idx].mean()
            ratio = am / (nm + 1e-15)
            status = "✅" if ratio > 1.5 else "❌"
            feat_check[name] = {"normal": float(nm), "attack": float(am), "ratio": float(ratio)}
            print(f"    {name:<10} normal={nm:.6f}  attack={am:.6f}  ratio={ratio:.2f}x {status}")
        check_results[feat_name] = feat_check

    # ── Best combined detection ──
    print("\n" + "=" * 85)
    print("COMBINED DETECTION: max(off_diag_std, abs_r4)")
    print("=" * 85)

    combined_by_type = {}
    for code in range(5):
        f = feats_by_type[code]
        if len(f) == 0:
            continue
        combined = np.maximum(f[:, 0], f[:, 9])  # max(off_diag_std, abs_r4)
        combined_by_type[code] = combined

    nm_combined = combined_by_type[0].mean()
    n_pass = 0
    print(f"{'Attack':<10} {'normal mean':>12} {'attack mean':>12} {'ratio':>8}")
    print("-" * 50)
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        am = combined_by_type.get(code, np.array([0])).mean()
        ratio = am / (nm_combined + 1e-15)
        if ratio > 1.5:
            n_pass += 1
        status = "✅" if ratio > 1.5 else "❌"
        print(f"{name:<10} {nm_combined:>12.6f} {am:>12.6f} {ratio:>8.2f}x {status}")

    print(f"\n{n_pass}/4 attack types exceed 1.5x ratio")
    if n_pass >= 3:
        print("GATE 2 PASSED ✅ — Combined features show sufficient separation.")
    else:
        print("GATE 2 MARGINAL — Proceeding with available separation, MLP may help.")

    # ── Save ──
    out = {"feature_separation": check_results, "combined_pass_count": n_pass}
    with open(os.path.join(results_dir, "phi_signal_check.json"), "w") as f:
        json.dump(out, f, indent=4)

    # ── Plot: off_diag_std + abs_r4 side by side ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Φ Signal Check: off_diag_std (top) and |r4| (bottom)", fontsize=14)

    for idx, (code, name) in enumerate([(1, "freq"), (2, "coi"), (3, "power"), (4, "voltage")]):
        atk = feats_by_type[code]

        # Top row: off_diag_std
        ax = axes[0][idx]
        nm = normal_feats[:, 0]
        at = atk[:, 0] if len(atk) > 0 else np.array([0])
        mx = max(np.percentile(nm, 99), np.percentile(at, 99)) if len(at) > 0 else np.percentile(nm, 99)
        bins = np.linspace(0, mx, 60)
        ax.hist(nm, bins=bins, alpha=0.6, density=True, color="steelblue", label="Normal")
        if len(at) > 0:
            ax.hist(at, bins=bins, alpha=0.6, density=True, color="crimson", label=name)
        ax.set_title(f"{name} — off_diag_std")
        ax.legend(fontsize=7)

        # Bottom row: abs_r4
        ax = axes[1][idx]
        nm = normal_feats[:, 9]
        at = atk[:, 9] if len(atk) > 0 else np.array([0])
        mx = max(np.percentile(nm, 99), np.percentile(at, 99)) if len(at) > 0 else np.percentile(nm, 99)
        bins = np.linspace(0, mx, 60)
        ax.hist(nm, bins=bins, alpha=0.6, density=True, color="steelblue", label="Normal")
        if len(at) > 0:
            ax.hist(at, bins=bins, alpha=0.6, density=True, color="crimson", label=name)
        ax.set_title(f"{name} — |r4|")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phi_signal_check.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {os.path.join(plots_dir, 'phi_signal_check.png')}")


if __name__ == "__main__":
    main()
