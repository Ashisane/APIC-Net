"""
Quick validation script for PI-LSTM.
Loads existing checkpoint and evaluates with fixed evaluation pipeline.
No training needed — just checks if detection works with current model weights.

Usage: python -m models.pi_lstm.quick_eval
"""
import os
import json
import torch
import numpy as np

from .dataset import get_dataloaders
from .federated import FLClient
from .model import PILSTM, PILoss
from .evaluate import evaluate_client, evaluate_inference_anomaly_scores


CONFIG = {
    "input_sequence_length": 20,
    "hidden_units": 64,
    "recurrent_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "local_epochs": 2,
    "num_clients": 6,
    "lambda_physics": 0.6,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    checkpoint_path = os.path.join(results_dir, "pi_lstm_global.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print("Run full training first: python -m models.pi_lstm.train")
        return

    # Load global weights
    print(f"Loading checkpoint: {checkpoint_path}")
    global_weights = torch.load(checkpoint_path, map_location=device)

    # Create clients and load data
    print("Loading data...")
    all_metrics = []

    for vsm_id in range(CONFIG["num_clients"]):
        train_ld, val_ld, test_ld = get_dataloaders(
            data_dir, vsm_id, CONFIG["batch_size"], CONFIG["input_sequence_length"]
        )

        # Create a client just for evaluation
        client = FLClient(vsm_id, train_ld, val_ld, CONFIG, device)
        client.set_weights(global_weights)

        print(f"\n--- VSM {vsm_id} ---")

        # First: dump raw score distributions
        test_scores, test_labels, test_atk_types, inf_time = evaluate_inference_anomaly_scores(
            client.model, client.loss_fn, test_ld, device
        )

        normal_scores = test_scores[test_labels == 0]
        attack_scores = test_scores[test_labels == 1]

        print(f"  Normal scores: n={len(normal_scores)}, mean={normal_scores.mean():.4f}, "
              f"std={normal_scores.std():.4f}, p95={np.percentile(normal_scores, 95):.4f}")
        if len(attack_scores) > 0:
            print(f"  Attack scores: n={len(attack_scores)}, mean={attack_scores.mean():.4f}, "
                  f"std={attack_scores.std():.4f}, p95={np.percentile(attack_scores, 95):.4f}")
            separation = attack_scores.mean() - normal_scores.mean()
            print(f"  Separation (attack - normal): {separation:.4f} "
                  f"{'✅ GOOD' if separation > 0 else '❌ INVERTED'}")

        # Per-attack type score distributions
        ATTACK_TYPE_NAMES = {0: "none", 1: "freq", 2: "coi", 3: "power", 4: "voltage"}
        for atype_code in sorted(np.unique(test_atk_types)):
            if atype_code == 0:
                continue
            mask = test_atk_types == atype_code
            atype_scores = test_scores[mask]
            atype_name = ATTACK_TYPE_NAMES.get(int(atype_code), str(atype_code))
            if len(atype_scores) > 0:
                print(f"  {atype_name:>8}: n={len(atype_scores):>5}, mean={atype_scores.mean():.4f}, "
                      f"sep={atype_scores.mean() - normal_scores.mean():+.4f}")

        # Full evaluation
        metrics, _, _ = evaluate_client(client, val_ld, test_ld, device)
        all_metrics.append(metrics)
        print(f"  F1: {metrics['overall']['f1']:.4f}, AUC: {metrics['overall']['auc']:.4f}, "
              f"Precision: {metrics['overall']['precision']:.4f}, Recall: {metrics['overall']['recall']:.4f}")
        print(f"  Threshold (tau): {metrics['overall']['tau']:.4f}")

        for atype, m in metrics["by_attack"].items():
            print(f"    {atype:>8}: F1={m['f1']:.4f}, AUC={m['auc']:.4f}")

    # Overall summary
    print("\n\n=== OVERALL SUMMARY ===")
    avg_precision = np.mean([m['overall']['precision'] for m in all_metrics])
    avg_recall = np.mean([m['overall']['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['overall']['f1'] for m in all_metrics])
    avg_auc = np.mean([m['overall']['auc'] for m in all_metrics])
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1:        {avg_f1:.4f}")
    print(f"AUC:       {avg_auc:.4f}")

    # Per-attack overall
    all_attack_keys = set()
    for m in all_metrics:
        all_attack_keys.update(m["by_attack"].keys())
    print("\nPer-attack:")
    for atype in sorted(all_attack_keys):
        vals = [m["by_attack"][atype] for m in all_metrics if atype in m["by_attack"]]
        if vals:
            avg_f1_a = np.mean([v["f1"] for v in vals])
            avg_auc_a = np.mean([v["auc"] for v in vals])
            print(f"  {atype:>8}: F1={avg_f1_a:.4f}, AUC={avg_auc_a:.4f}")

    # Save
    results = {
        "overall": {
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
            "auc": float(avg_auc),
        },
        "by_attack": {},
        "note": "Quick eval on existing checkpoint — no retraining"
    }
    for atype in sorted(all_attack_keys):
        vals = [m["by_attack"][atype] for m in all_metrics if atype in m["by_attack"]]
        if vals:
            results["by_attack"][atype] = {
                "f1": float(np.mean([v["f1"] for v in vals])),
                "auc": float(np.mean([v["auc"] for v in vals])),
            }

    out_file = os.path.join(results_dir, "pi_lstm_quick_eval.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
