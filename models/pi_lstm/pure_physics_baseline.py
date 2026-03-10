"""
TASK 02E — Pure Physics Residual Baseline

No LSTM. No training. Just: does observed omega match what the swing
equation predicts? Uses same data splits and normalization as PI-LSTM.

Usage: python -m models.pi_lstm.pure_physics_baseline
"""
import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Physics constants — must match vsm_simulator.py exactly
OMEGA_STAR = 2 * np.pi * 50.0
DT = 0.005
H, D_VAL, F_VAL = 5.0, 25.0, 10.0

ATTACK_TYPE_NAMES = {0: "normal", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}

# Same column indices as simulator
COL_OMEGA   = 0
COL_DELTA   = 1
COL_P       = 2
COL_PSTAR   = 3
COL_VREF    = 4
COL_VDC     = 5
COL_OMEGA_C = 6
COL_IF      = 7


def compute_r_phys_raw(data, labels):
    """
    Compute per-timestep r_phys directly from raw (unnormalized) data.

    Args:
        data:   (N_steps, 8) raw signal array for one scenario
        labels: (N_steps,)   binary labels

    Returns:
        r_phys: (N_steps-1,) physics residual at each step
        labs:   (N_steps-1,) corresponding labels
    """
    n_steps = data.shape[0]

    omega     = data[:, COL_OMEGA]
    p         = data[:, COL_P]
    p_star    = data[:, COL_PSTAR]
    omega_C   = data[:, COL_OMEGA_C]

    # Euler prediction: omega_phys[t] = what omega should be at t+1
    d_omega = (1.0 / (2.0 * H)) * (
        (p_star[:-1] - p[:-1]) * OMEGA_STAR
        + D_VAL * (OMEGA_STAR - omega[:-1])
        + F_VAL * (omega_C[:-1] - omega[:-1])
    )
    omega_phys_next = omega[:-1] + DT * d_omega

    # Compare prediction to actual next omega
    r_phys = np.abs(omega[1:] - omega_phys_next)

    return r_phys, labels[1:]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    n_vsm = 6
    np.random.seed(42)

    all_r_phys = []
    all_labels = []
    all_atk_types = []
    all_r_phys_val_normal = []  # for threshold

    print("Computing pure physics residuals for all VSMs...")

    for vsm_id in range(n_vsm):
        data_file = os.path.join(data_dir, f"vsm_{vsm_id}.npz")
        d = np.load(data_file)
        recording = d["data"]         # (N_scenarios, N_steps, 8)
        labels_arr = d["labels"]       # (N_scenarios, N_steps)
        atk_types_arr = d["attack_types"]  # (N_scenarios, N_steps)

        n_scenarios = recording.shape[0]

        # Same shuffle + split as dataset.py
        shuffle_idx = np.random.RandomState(42).permutation(n_scenarios)
        recording    = recording[shuffle_idx]
        labels_arr   = labels_arr[shuffle_idx]
        atk_types_arr = atk_types_arr[shuffle_idx]

        n_train = int(n_scenarios * 0.7)
        n_val   = int(n_scenarios * 0.15)

        val_rec   = recording[n_train:n_train + n_val]
        val_lbl   = labels_arr[n_train:n_train + n_val]

        test_rec  = recording[n_train + n_val:]
        test_lbl  = labels_arr[n_train + n_val:]
        test_atk  = atk_types_arr[n_train + n_val:]

        # Validation normal r_phys → threshold
        for scnt in range(val_rec.shape[0]):
            rp, lbl = compute_r_phys_raw(val_rec[scnt], val_lbl[scnt])
            normal_mask = lbl == 0
            all_r_phys_val_normal.append(rp[normal_mask])

        # Test residuals
        for scnt in range(test_rec.shape[0]):
            rp, lbl = compute_r_phys_raw(test_rec[scnt], test_lbl[scnt])
            atk = test_atk[scnt, 1:]  # offset by 1 (prediction is at t+1)
            all_r_phys.append(rp)
            all_labels.append(lbl)
            all_atk_types.append(atk)

        print(f"  VSM {vsm_id}: {test_rec.shape[0]} test scenarios")

    # Concatenate
    r_phys     = np.concatenate(all_r_phys)
    labels     = np.concatenate(all_labels)
    atk_types  = np.concatenate(all_atk_types)
    val_normal = np.concatenate(all_r_phys_val_normal)

    # Threshold: 95th percentile of val normal r_phys
    tau = np.percentile(val_normal, 95)
    preds = (r_phys > tau).astype(int)

    print(f"\nTotal test samples: {len(r_phys)}")
    print(f"  Normal: {(labels==0).sum()}")
    print(f"  Attack: {(labels==1).sum()}")
    print(f"  Threshold τ = {tau:.6f}")

    # ── r_phys distribution table ────────────────────────────────
    print("\n" + "="*75)
    print("r_phys DISTRIBUTION TABLE")
    print("="*75)
    header = f"{'Sample Type':<14} {'r_phys mean':>12} {'r_phys std':>11} {'r_phys p95':>11} {'ratio':>8} {'n':>8}"
    print(header)
    print("-" * len(header))

    normal_mean = r_phys[labels == 0].mean()
    dist_table = {}

    for code, name in [(0, "normal"), (1, "freq"), (2, "coi"), (3, "power"), (4, "voltage")]:
        mask = atk_types == code if code != 0 else labels == 0
        if mask.sum() == 0:
            continue
        mean_v  = float(r_phys[mask].mean())
        std_v   = float(r_phys[mask].std())
        p95_v   = float(np.percentile(r_phys[mask], 95))
        ratio   = mean_v / normal_mean if code != 0 else 1.0
        dist_table[name] = {"mean": mean_v, "std": std_v, "p95": p95_v,
                            "ratio": ratio, "n": int(mask.sum())}
        print(f"{name:<14} {mean_v:>12.6f} {std_v:>11.6f} {p95_v:>11.6f} {ratio:>8.2f}x {mask.sum():>8}")

    # ── Overall metrics ───────────────────────────────────────────
    print("\n" + "="*75)
    print("DETECTION METRICS")
    print("="*75)

    precision  = precision_score(labels, preds, zero_division=0)
    recall     = recall_score(labels, preds, zero_division=0)
    f1         = f1_score(labels, preds, zero_division=0)
    auc        = roc_auc_score(labels, r_phys) if len(np.unique(labels)) > 1 else 0.0

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  τ (95th pct val normal): {tau:.6f}")

    # ── Per-attack metrics ────────────────────────────────────────
    print("\n  Per-attack:")
    per_attack = {}
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        mask = (atk_types == code) | (labels == 0)
        if mask.sum() == 0:
            continue
        sub_l = labels[mask]
        sub_p = preds[mask]
        sub_r = r_phys[mask]
        if len(np.unique(sub_l)) < 2:
            per_attack[name] = {"precision": 0, "recall": 0, "f1": 0, "auc": 0}
        else:
            per_attack[name] = {
                "precision": float(precision_score(sub_l, sub_p, zero_division=0)),
                "recall":    float(recall_score(sub_l, sub_p, zero_division=0)),
                "f1":        float(f1_score(sub_l, sub_p, zero_division=0)),
                "auc":       float(roc_auc_score(sub_l, sub_r)),
            }
        m = per_attack[name]
        print(f"    {name:>8}: F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    results = {
        "overall": {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "auc":       float(auc),
            "tau":       float(tau),
        },
        "by_attack": per_attack,
        "r_phys_distribution": dist_table,
    }
    out = os.path.join(results_dir, "pure_physics_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
