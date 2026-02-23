import os
import json
import torch
import numpy as np

from .dataset import get_dataloaders
from .federated import FLServer, FLClient
from .evaluate import evaluate_client

# Define config locally per spec
CONFIG = {
    "input_sequence_length": 20,
    "hidden_units": 64,
    "recurrent_layers": 2,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "batch_size": 32,
    "fl_rounds": 25,
    "local_epochs": 2,
    "num_clients": 6,
    "lambda_physics": 0.6,
}

EVAL_EVERY = 5  # Evaluate and save checkpoint every N rounds


def evaluate_and_save(server, val_loaders, test_loaders, device, round_losses, results_dir, tag=""):
    """Evaluate all clients and save results + checkpoint."""
    global_weights = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
    overall_metrics = []

    for i, client in enumerate(server.clients):
        client.set_weights(global_weights)
        metrics, _, _ = evaluate_client(client, val_loaders[i], test_loaders[i], device)
        overall_metrics.append(metrics)
        print(f"  VSM {i} F1: {metrics['overall']['f1']:.4f}, AUC: {metrics['overall']['auc']:.4f}")

    # Average metrics across clients
    avg_metrics = {
        "overall": {
            "precision": float(np.mean([m['overall']['precision'] for m in overall_metrics])),
            "recall": float(np.mean([m['overall']['recall'] for m in overall_metrics])),
            "f1": float(np.mean([m['overall']['f1'] for m in overall_metrics])),
            "auc": float(np.mean([m['overall']['auc'] for m in overall_metrics])),
            "inference_time_ms": float(np.mean([m['overall']['inference_time_ms'] for m in overall_metrics])),
        },
        "by_attack": {},
        "training_loss_per_round": round_losses,
    }

    # Average per-attack metrics
    all_attack_keys = set()
    for m in overall_metrics:
        all_attack_keys.update(m["by_attack"].keys())
    for atype in all_attack_keys:
        vals = [m["by_attack"][atype] for m in overall_metrics if atype in m["by_attack"]]
        if vals:
            avg_metrics["by_attack"][atype] = {
                "precision": float(np.mean([v["precision"] for v in vals])),
                "recall": float(np.mean([v["recall"] for v in vals])),
                "f1": float(np.mean([v["f1"] for v in vals])),
                "auc": float(np.mean([v["auc"] for v in vals])),
            }

    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "pi_lstm_results.json")
    with open(results_file, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    torch.save(server.global_model.state_dict(), os.path.join(results_dir, "pi_lstm_global.pt"))

    f1 = avg_metrics['overall']['f1']
    auc = avg_metrics['overall']['auc']
    print(f"  >> {tag}Overall F1: {f1:.4f}, AUC: {auc:.4f} — saved to {results_file}")
    return avg_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")

    # Init server
    server = FLServer(CONFIG, device)

    # Init clients
    print("Initializing clients and loading data...")
    val_loaders = []
    test_loaders = []

    for vsm_id in range(CONFIG["num_clients"]):
        train_ld, val_ld, test_ld = get_dataloaders(
            data_dir, vsm_id, CONFIG["batch_size"], CONFIG["input_sequence_length"]
        )
        val_loaders.append(val_ld)
        test_loaders.append(test_ld)

        client = FLClient(vsm_id, train_ld, val_ld, CONFIG, device)
        server.add_client(client)

    print(f"Starting Federated Training for {CONFIG['fl_rounds']} rounds (eval every {EVAL_EVERY})...")
    round_losses = []

    for rnd in range(CONFIG["fl_rounds"]):
        avg_loss, start_t = server.train_round()
        round_losses.append(avg_loss)
        print(f"[Round {rnd+1}/{CONFIG['fl_rounds']}] Avg Loss: {avg_loss:.6f}")

        # Periodic evaluation
        if (rnd + 1) % EVAL_EVERY == 0:
            print(f"--- Checkpoint at round {rnd+1} ---")
            evaluate_and_save(server, val_loaders, test_loaders, device,
                              round_losses, results_dir, tag=f"Round {rnd+1}: ")

    # Final evaluation (if not already done at last round)
    if CONFIG["fl_rounds"] % EVAL_EVERY != 0:
        print("--- Final evaluation ---")
        evaluate_and_save(server, val_loaders, test_loaders, device,
                          round_losses, results_dir, tag="Final: ")

    print("Done!")

if __name__ == "__main__":
    main()

