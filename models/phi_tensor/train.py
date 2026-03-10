"""
Φ Tensor — Full Training and Evaluation Pipeline

Computes Φ features for all VSMs, trains MLP detector with FedAvg,
evaluates, and saves results.

Usage: python -m models.phi_tensor.train
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from .phi_tensor import compute_phi_for_scenario
from .features import extract_features_batch, aggregate_window, N_FEATURES

ATTACK_TYPE_NAMES = {0: "normal", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}
WINDOW_SIZE = 20
MLP_INPUT_DIM = N_FEATURES * 2  # mean + std = 20


# ── MLP Detector ──────────────────────────────────────────────

class PhiDetectorMLP(nn.Module):
    def __init__(self, input_dim=MLP_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Focal Loss ────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ── Feature Computation ──────────────────────────────────────

def compute_features_for_vsm(data_dir, vsm_id, split="test"):
    """
    Compute windowed features + labels for a VSM split.

    Returns:
        features: (N_windows, 20) — aggregated features.
        labels:   (N_windows,) — binary labels (majority in window).
        atk_types:(N_windows,) — attack type of the window.
    """
    d = np.load(os.path.join(data_dir, f"vsm_{vsm_id}.npz"))
    recording = d["data"]
    labels_arr = d["labels"]
    atk_types_arr = d["attack_types"]

    n_scenarios = recording.shape[0]
    shuffle_idx = np.random.RandomState(42).permutation(n_scenarios)
    recording = recording[shuffle_idx]
    labels_arr = labels_arr[shuffle_idx]
    atk_types_arr = atk_types_arr[shuffle_idx]

    n_train = int(n_scenarios * 0.7)
    n_val = int(n_scenarios * 0.15)

    if split == "train":
        rec = recording[:n_train]
        lbl = labels_arr[:n_train]
        atk = atk_types_arr[:n_train]
    elif split == "val":
        rec = recording[n_train:n_train + n_val]
        lbl = labels_arr[n_train:n_train + n_val]
        atk = atk_types_arr[n_train:n_train + n_val]
    else:
        rec = recording[n_train + n_val:]
        lbl = labels_arr[n_train + n_val:]
        atk = atk_types_arr[n_train + n_val:]

    all_feats = []
    all_labels = []
    all_atk = []

    for s in range(rec.shape[0]):
        Phi, residuals = compute_phi_for_scenario(rec[s])
        feats = extract_features_batch(Phi, residuals)  # (T, 10)
        agg, indices = aggregate_window(feats, WINDOW_SIZE)  # (N_w, 20)

        if len(agg) == 0:
            continue

        # Window label = 1 if ANY timestep in window is attack
        for i, idx in enumerate(indices):
            win_lbl = lbl[s, idx + 1: idx + 1 + WINDOW_SIZE]
            win_atk = atk[s, idx + 1: idx + 1 + WINDOW_SIZE]
            label = 1 if win_lbl.any() else 0
            # Attack type = most common non-zero type in window
            atk_in_win = win_atk[win_atk > 0]
            atk_type = int(np.bincount(atk_in_win.astype(int)).argmax()) if len(atk_in_win) > 0 else 0
            all_labels.append(label)
            all_atk.append(atk_type)

        all_feats.append(agg)

    features = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels)
    atk_types = np.array(all_atk)

    return features, labels, atk_types


# ── FL Training ────────────────────────────────────────────────

def train_mlp_fedavg(data_dir, n_vsm=6, n_rounds=20, local_epochs=3, lr=0.001, batch_size=256):
    """
    Train MLP with FedAvg across 6 VSMs.
    """
    device = torch.device("cpu")

    # Pre-compute features for all VSMs
    print("Pre-computing features for all VSMs...")
    train_data = {}
    val_data = {}
    test_data = {}

    for vsm_id in range(n_vsm):
        print(f"  VSM {vsm_id}: ", end="", flush=True)
        t0 = time.time()
        train_data[vsm_id] = compute_features_for_vsm(data_dir, vsm_id, "train")
        val_data[vsm_id] = compute_features_for_vsm(data_dir, vsm_id, "val")
        test_data[vsm_id] = compute_features_for_vsm(data_dir, vsm_id, "test")
        t1 = time.time()
        print(f"train={len(train_data[vsm_id][0])}, val={len(val_data[vsm_id][0])}, "
              f"test={len(test_data[vsm_id][0])} ({t1-t0:.1f}s)")

    # Compute normalization (mean/std from train normal samples)
    all_train_feats = []
    for vsm_id in range(n_vsm):
        feats, labels, _ = train_data[vsm_id]
        normal_mask = labels == 0
        all_train_feats.append(feats[normal_mask])
    all_normal = np.concatenate(all_train_feats, axis=0)
    feat_mean = all_normal.mean(axis=0)
    feat_std = all_normal.std(axis=0) + 1e-8

    # Normalize all splits
    for vsm_id in range(n_vsm):
        train_data[vsm_id] = ((train_data[vsm_id][0] - feat_mean) / feat_std,
                               train_data[vsm_id][1], train_data[vsm_id][2])
        val_data[vsm_id] = ((val_data[vsm_id][0] - feat_mean) / feat_std,
                             val_data[vsm_id][1], val_data[vsm_id][2])
        test_data[vsm_id] = ((test_data[vsm_id][0] - feat_mean) / feat_std,
                              test_data[vsm_id][1], test_data[vsm_id][2])

    # Global model
    global_model = PhiDetectorMLP(MLP_INPUT_DIM).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    print(f"\nStarting FedAvg training: {n_rounds} rounds, {local_epochs} local epochs")
    print(f"MLP input dim: {MLP_INPUT_DIM}, architecture: {MLP_INPUT_DIM}→32→16→1")

    for rnd in range(n_rounds):
        local_weights = []
        local_losses = []

        for vsm_id in range(n_vsm):
            feats, labels, _ = train_data[vsm_id]

            # Create local model from global weights
            local_model = PhiDetectorMLP(MLP_INPUT_DIM).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            local_model.train()

            X = torch.tensor(feats, dtype=torch.float32).to(device)
            y = torch.tensor(labels, dtype=torch.float32).to(device)

            epoch_loss = 0
            for epoch in range(local_epochs):
                # Shuffle
                perm = torch.randperm(len(X))
                X, y = X[perm], y[perm]

                for i in range(0, len(X), batch_size):
                    xb = X[i:i + batch_size]
                    yb = y[i:i + batch_size]
                    pred = local_model(xb)
                    loss = criterion(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            local_weights.append(local_model.state_dict())
            local_losses.append(epoch_loss)

        # FedAvg
        avg_state = {}
        for key in local_weights[0]:
            avg_state[key] = torch.stack([w[key].float() for w in local_weights]).mean(dim=0)
        global_model.load_state_dict(avg_state)

        avg_loss = np.mean(local_losses)
        print(f"[Round {rnd+1}/{n_rounds}] Avg Loss: {avg_loss:.4f}")

    return global_model, test_data, feat_mean, feat_std


# ── Evaluation ────────────────────────────────────────────────

def evaluate(model, test_data, n_vsm=6):
    """Evaluate on test set, return per-VSM and overall metrics."""
    device = torch.device("cpu")
    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []
    all_atk = []

    with torch.no_grad():
        for vsm_id in range(n_vsm):
            feats, labels, atk_types = test_data[vsm_id]
            X = torch.tensor(feats, dtype=torch.float32).to(device)
            scores = model(X).cpu().numpy()
            preds = (scores > 0.5).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)
            all_scores.append(scores)
            all_atk.append(atk_types)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    atk_types = np.concatenate(all_atk)

    # Overall
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0

    results = {
        "overall": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        },
        "by_attack": {}
    }

    # Per-attack
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        mask = (atk_types == code) | (labels == 0)
        if mask.sum() == 0:
            continue
        sub_l = labels[mask]
        sub_p = preds[mask]
        sub_s = scores[mask]
        if len(np.unique(sub_l)) < 2:
            results["by_attack"][name] = {"precision": 0, "recall": 0, "f1": 0, "auc": 0}
        else:
            results["by_attack"][name] = {
                "precision": float(precision_score(sub_l, sub_p, zero_division=0)),
                "recall": float(recall_score(sub_l, sub_p, zero_division=0)),
                "f1": float(f1_score(sub_l, sub_p, zero_division=0)),
                "auc": float(roc_auc_score(sub_l, sub_s)),
            }

    return results


# ── Also: threshold-only detector ─────────────────────────────

def evaluate_threshold(data_dir, n_vsm=6):
    """
    Simple threshold detector using combined max(off_diag_std, abs_r4)
    with τ = 95th percentile on val normal.
    """
    print("\n--- Threshold-only detector ---")

    # Compute val normal scores for threshold
    val_scores_normal = []
    for vsm_id in range(n_vsm):
        feats, labels, _ = compute_features_for_vsm(data_dir, vsm_id, "val")
        # Combined score: use raw feature indices 0 (off_diag_std) and 9 (abs_r4)
        # But features are aggregated windows with mean+std, so:
        # mean_off_diag_std = col 0, mean_abs_r4 = col 9
        combined = np.maximum(feats[:, 0], feats[:, 9])
        normal_mask = labels == 0
        val_scores_normal.append(combined[normal_mask])

    val_normal = np.concatenate(val_scores_normal)
    tau = np.percentile(val_normal, 95)
    print(f"  Threshold τ = {tau:.6f}")

    # Evaluate on test
    all_labels = []
    all_scores = []
    all_atk = []
    for vsm_id in range(n_vsm):
        feats, labels, atk_types = compute_features_for_vsm(data_dir, vsm_id, "test")
        combined = np.maximum(feats[:, 0], feats[:, 9])
        all_labels.append(labels)
        all_scores.append(combined)
        all_atk.append(atk_types)

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    atk_types = np.concatenate(all_atk)
    preds = (scores > tau).astype(int)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

    results = {
        "overall": {"precision": float(precision), "recall": float(recall),
                    "f1": float(f1), "auc": float(auc), "tau": float(tau)},
        "by_attack": {}
    }
    for code in [1, 2, 3, 4]:
        name = ATTACK_TYPE_NAMES[code]
        mask = (atk_types == code) | (labels == 0)
        sub_l, sub_p, sub_s = labels[mask], preds[mask], scores[mask]
        if len(np.unique(sub_l)) < 2:
            results["by_attack"][name] = {"precision": 0, "recall": 0, "f1": 0, "auc": 0}
        else:
            results["by_attack"][name] = {
                "precision": float(precision_score(sub_l, sub_p, zero_division=0)),
                "recall": float(recall_score(sub_l, sub_p, zero_division=0)),
                "f1": float(f1_score(sub_l, sub_p, zero_division=0)),
                "auc": float(roc_auc_score(sub_l, sub_s)),
            }
        print(f"  {name:>8}: F1={results['by_attack'][name]['f1']:.4f}  "
              f"AUC={results['by_attack'][name]['auc']:.4f}")

    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Φ TENSOR — Full Training Pipeline")
    print("=" * 60)

    t_start = time.time()

    # 1. Threshold detector
    threshold_results = evaluate_threshold(data_dir)

    # 2. MLP detector with FedAvg
    print("\n--- MLP detector with FedAvg ---")
    model, test_data, feat_mean, feat_std = train_mlp_fedavg(
        data_dir, n_vsm=6, n_rounds=20, local_epochs=3, lr=0.001
    )
    mlp_results = evaluate(model, test_data)

    t_end = time.time()

    # Print MLP results
    print(f"\n  MLP Overall:")
    for k, v in mlp_results["overall"].items():
        print(f"    {k}: {v:.4f}")
    print(f"  MLP Per-attack:")
    for name, m in mlp_results["by_attack"].items():
        print(f"    {name:>8}: F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # Save combined results
    final = {
        "threshold_detector": threshold_results,
        "mlp_detector": mlp_results,
        "total_time_seconds": t_end - t_start,
    }
    out_file = os.path.join(results_dir, "phi_tensor_results.json")
    with open(out_file, "w") as f:
        json.dump(final, f, indent=4)
    print(f"\nSaved to {out_file}")
    print(f"Total time: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
