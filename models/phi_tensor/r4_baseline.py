"""
TASK 03B — r4-Only Baseline + Combined Scores

No training needed. Pure threshold/AUC computation.
Tests whether r4 alone outperforms everything, making Φ redundant.

Usage: python -m models.phi_tensor.r4_baseline
"""
import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ATTACK_TYPE_NAMES = {0: "normal", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}

# Column indices
COL_OMEGA   = 0
COL_OMEGA_C = 6


def compute_r4_and_rphys(data_dir, split, n_vsm=6):
    """
    Compute r4 = |ωC - ω| and r_phys = |ω(t+1) - ω_euler_predicted|
    for all timesteps in a split.
    """
    from models.phi_tensor.residuals import OMEGA_STAR, DT, H, D_VAL, F_VAL
    COL_P, COL_PSTAR = 2, 3

    all_r4 = []
    all_rphys = []
    all_labels = []
    all_atk = []

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

        if split == "val":
            rec = rec[n_train:n_train + n_val]
            lbl = lbl[n_train:n_train + n_val]
            atk = atk[n_train:n_train + n_val]
        elif split == "test":
            rec = rec[n_train + n_val:]
            lbl = lbl[n_train + n_val:]
            atk = atk[n_train + n_val:]

        for s in range(rec.shape[0]):
            omega = rec[s, :, COL_OMEGA]
            omega_C = rec[s, :, COL_OMEGA_C]
            p = rec[s, :, COL_P]
            p_star = rec[s, :, COL_PSTAR]

            # r4: |ωC - ω| at each timestep
            r4 = np.abs(omega_C - omega)

            # r_phys: |ω(t+1) - ω_euler| (same as pure_physics_baseline)
            d_omega = (1.0 / (2.0 * H)) * (
                (p_star[:-1] - p[:-1]) * OMEGA_STAR
                + D_VAL * (OMEGA_STAR - omega[:-1])
                + F_VAL * (omega_C[:-1] - omega[:-1])
            )
            omega_euler = omega[:-1] + DT * d_omega
            rphys = np.abs(omega[1:] - omega_euler)

            # Align: r4 uses t, rphys uses t→t+1 transition
            # Use t+1 labels for rphys, t labels for r4
            # To keep them aligned, use timesteps 1..N-1
            all_r4.append(r4[1:])
            all_rphys.append(rphys)
            all_labels.append(lbl[s, 1:])
            all_atk.append(atk[s, 1:])

    return (np.concatenate(all_r4), np.concatenate(all_rphys),
            np.concatenate(all_labels), np.concatenate(all_atk))


def eval_detector(scores, labels, atk_types, tau, name):
    """Evaluate a threshold detector and print results."""
    preds = (scores > tau).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0

    print(f"\n  {name}:")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC:       {auc:.4f}")
    print(f"    τ:         {tau:.6f}")

    results = {"overall": {"precision": float(precision), "recall": float(recall),
                           "f1": float(f1), "auc": float(auc), "tau": float(tau)},
               "by_attack": {}}

    print(f"    Per-attack:")
    for code in [1, 2, 3, 4]:
        aname = ATTACK_TYPE_NAMES[code]
        mask = (atk_types == code) | (labels == 0)
        sub_l, sub_p, sub_s = labels[mask], preds[mask], scores[mask]
        if len(np.unique(sub_l)) < 2:
            results["by_attack"][aname] = {"precision": 0, "recall": 0, "f1": 0, "auc": 0}
        else:
            results["by_attack"][aname] = {
                "precision": float(precision_score(sub_l, sub_p, zero_division=0)),
                "recall": float(recall_score(sub_l, sub_p, zero_division=0)),
                "f1": float(f1_score(sub_l, sub_p, zero_division=0)),
                "auc": float(roc_auc_score(sub_l, sub_s)),
            }
        m = results["by_attack"][aname]
        print(f"      {aname:>8}: F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    return results


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("TASK 03B — r4 Decision Gate")
    print("=" * 60)

    # ── Compute scores ──
    print("\nComputing r4 and r_phys for validation set...")
    r4_val, rphys_val, lbl_val, atk_val = compute_r4_and_rphys(data_dir, "val")

    print("Computing r4 and r_phys for test set...")
    r4_test, rphys_test, lbl_test, atk_test = compute_r4_and_rphys(data_dir, "test")

    print(f"\nTest samples: {len(r4_test)} (normal: {(lbl_test==0).sum()}, attack: {(lbl_test==1).sum()})")

    # ── Thresholds from val normal ──
    r4_normal_val = r4_val[lbl_val == 0]
    rphys_normal_val = rphys_val[lbl_val == 0]

    tau_r4 = np.percentile(r4_normal_val, 95)
    tau_rphys = np.percentile(rphys_normal_val, 95)

    # ── PART A: r4-only ──
    print("\n" + "=" * 60)
    print("PART A: r4-Only Detector")
    print("=" * 60)
    r4_results = eval_detector(r4_test, lbl_test, atk_test, tau_r4, "r4-only")

    # Save r4 results
    with open(os.path.join(results_dir, "r4_only_results.json"), "w") as f:
        json.dump(r4_results, f, indent=4)

    # ── PART B: Decision Gate ──
    print("\n" + "=" * 60)
    print("PART B: Decision Gate")
    print("=" * 60)
    r4_auc = r4_results["overall"]["auc"]
    per_attack_aucs = {k: v["auc"] for k, v in r4_results["by_attack"].items()}
    min_attack_auc = min(per_attack_aucs.values()) if per_attack_aucs else 0

    if r4_auc > 0.85 and min_attack_auc > 0.75:
        print(f"  r4 AUC={r4_auc:.4f} > 0.85, min attack AUC={min_attack_auc:.4f} > 0.75")
        print("  ⚠ r4 IS DOING ALL THE WORK — Φ cross-constraint adds no value")
        print("  → STOP Φ MLP training, reframe contribution")
        decision = "r4_dominates"
    elif r4_auc < 0.75 or min_attack_auc < 0.65:
        print(f"  r4 AUC={r4_auc:.4f}, min attack AUC={min_attack_auc:.4f}")
        print("  → Φ+r4 combination is meaningful, continue training")
        decision = "phi_needed"
    else:
        print(f"  r4 AUC={r4_auc:.4f} (0.75-0.85 range), min attack AUC={min_attack_auc:.4f}")
        print("  → Φ+r4 likely adds value, continue but flag in report")
        decision = "marginal"

    # ── PART D: Combined Scores ──
    print("\n" + "=" * 60)
    print("PART D: Combined Score Tests")
    print("=" * 60)

    # Normalize using val stats
    def normalize(arr, val_arr):
        mu = val_arr[lbl_val == 0].mean()
        sigma = val_arr[lbl_val == 0].std() + 1e-10
        return (arr - mu) / sigma

    r4_norm = normalize(r4_test, r4_val)
    rphys_norm = normalize(rphys_test, rphys_val)

    # Score 1: r4 + r_phys
    combined1 = 0.5 * r4_norm + 0.5 * rphys_norm
    tau_c1 = np.percentile(
        0.5 * normalize(r4_val, r4_val)[lbl_val == 0] +
        0.5 * normalize(rphys_val, rphys_val)[lbl_val == 0], 95)
    c1_results = eval_detector(combined1, lbl_test, atk_test, tau_c1, "r4 + r_phys")

    # Score 2: compute off_diag_std for test
    # Quick: just use r4 and r_phys since off_diag_std ≈ 0 for 3/4 attacks
    # Actually, let's compute it properly from the Φ features
    print("\n  Computing off_diag_std for combined test (this may take a while)...")
    from models.phi_tensor.phi_tensor import compute_phi_for_scenario

    off_diag_std_all = []
    for vsm_id in range(6):
        d = np.load(os.path.join(data_dir, f"vsm_{vsm_id}.npz"))
        rec = d["data"]
        n_sc = rec.shape[0]
        shuffle_idx = np.random.RandomState(42).permutation(n_sc)
        rec = rec[shuffle_idx]
        n_train = int(n_sc * 0.7)
        n_val = int(n_sc * 0.15)
        test_rec = rec[n_train + n_val:]

        for s in range(test_rec.shape[0]):
            Phi, _ = compute_phi_for_scenario(test_rec[s])
            mask_od = ~np.eye(4, dtype=bool)
            for t in range(Phi.shape[0]):
                off_diag_std_all.append(np.std(Phi[t][mask_od]))

    ods_test = np.array(off_diag_std_all)

    # Also compute for val
    ods_val_all = []
    for vsm_id in range(6):
        d = np.load(os.path.join(data_dir, f"vsm_{vsm_id}.npz"))
        rec = d["data"]
        n_sc = rec.shape[0]
        shuffle_idx = np.random.RandomState(42).permutation(n_sc)
        rec = rec[shuffle_idx]
        n_train = int(n_sc * 0.7)
        n_val = int(n_sc * 0.15)
        val_rec = rec[n_train:n_train + n_val]

        for s in range(val_rec.shape[0]):
            Phi, _ = compute_phi_for_scenario(val_rec[s])
            mask_od = ~np.eye(4, dtype=bool)
            for t in range(Phi.shape[0]):
                ods_val_all.append(np.std(Phi[t][mask_od]))

    ods_val = np.array(ods_val_all)

    ods_norm = normalize(ods_test, ods_val)
    combined2 = 0.5 * ods_norm + 0.5 * r4_norm
    tau_c2 = np.percentile(
        0.5 * normalize(ods_val, ods_val)[lbl_val == 0] +
        0.5 * normalize(r4_val, r4_val)[lbl_val == 0], 95)
    c2_results = eval_detector(combined2, lbl_test, atk_test, tau_c2, "off_diag_std + r4")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY: AUC Comparison")
    print("=" * 60)
    print(f"  r4-only:          {r4_results['overall']['auc']:.4f}")
    print(f"  r4 + r_phys:      {c1_results['overall']['auc']:.4f}")
    print(f"  off_diag_std+r4:  {c2_results['overall']['auc']:.4f}")
    print(f"  Pure r_phys (02E): 0.6337")
    print(f"  PI-LSTM (02D):     0.5250")

    # Determine winner
    aucs = {"r4_only": r4_results['overall']['auc'],
            "r4_plus_rphys": c1_results['overall']['auc'],
            "ods_plus_r4": c2_results['overall']['auc']}
    winner = max(aucs, key=aucs.get)
    print(f"\n  Winner: {winner} (AUC={aucs[winner]:.4f})")

    # Save combined results
    combined_results = {
        "r4_only": r4_results,
        "r4_plus_rphys": c1_results,
        "ods_plus_r4": c2_results,
        "decision": decision,
        "winner": winner,
    }
    with open(os.path.join(results_dir, "combined_scores_results.json"), "w") as f:
        json.dump(combined_results, f, indent=4)
    print(f"\nSaved to {os.path.join(results_dir, 'combined_scores_results.json')}")


if __name__ == "__main__":
    main()
