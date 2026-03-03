"""Quick sanity check: 3 rounds, then print anomaly score distributions."""
import os, sys, torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.pi_lstm.dataset import VSMDataset
from models.pi_lstm.federated import FLServer, FLClient
from models.pi_lstm.evaluate import evaluate_inference_anomaly_scores
from torch.utils.data import DataLoader

CONFIG = {
    "input_sequence_length": 20,
    "hidden_units": 64,
    "recurrent_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "fl_rounds": 3,
    "local_epochs": 2,
    "num_clients": 2,  # Only 2 VSMs for speed
    "lambda_physics": 0.6,
}

STRIDE = 20  # Faster for sanity check

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")

    server = FLServer(CONFIG, device)
    val_loaders = []

    print("Loading data (stride=20, 2 VSMs only)...")
    for vsm_id in range(CONFIG["num_clients"]):
        data_file = os.path.join(data_dir, f"vsm_{vsm_id}.npz")
        meta_file = os.path.join(data_dir, "metadata.json")
        
        train_ds = VSMDataset(data_file, meta_file, split="train", window_size=20, stride=STRIDE)
        val_ds = VSMDataset(data_file, meta_file, split="val", window_size=20, stride=STRIDE)
        
        train_ld = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        val_loaders.append(val_ld)
        client = FLClient(vsm_id, train_ld, val_ld, CONFIG, device)
        server.add_client(client)
        print(f"  VSM {vsm_id}: {len(train_ds)} train, {len(val_ds)} val samples")

    for rnd in range(CONFIG["fl_rounds"]):
        avg_loss, _ = server.train_round()
        print(f"[Round {rnd+1}/{CONFIG['fl_rounds']}] Loss: {avg_loss:.6f}")

    # Evaluate on val set
    print("\n--- Score Distribution (VSM 0, Val Set) ---")
    global_weights = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
    server.clients[0].set_weights(global_weights)

    scores, labels, atk_types, _ = evaluate_inference_anomaly_scores(
        server.clients[0].model, server.clients[0].loss_fn, val_loaders[0], device
    )

    normal_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]

    print(f"Normal: n={len(normal_scores)}, mean={normal_scores.mean():.4f}, "
          f"std={normal_scores.std():.4f}, p95={np.percentile(normal_scores, 95):.4f}")
    if len(attack_scores) > 0:
        print(f"Attack: n={len(attack_scores)}, mean={attack_scores.mean():.4f}, "
              f"std={attack_scores.std():.4f}, p95={np.percentile(attack_scores, 95):.4f}")
        sep = attack_scores.mean() - normal_scores.mean()
        print(f"\nSeparation: {sep:+.4f} {'✓ GOOD' if sep > 0 else '✗ BAD'}")
    else:
        print("  NO ATTACK SAMPLES")

    # Per-attack type
    for at in np.unique(atk_types):
        if at == 0: continue
        mask = atk_types == at
        print(f"  Type {at}: n={mask.sum()}, mean_score={scores[mask].mean():.4f}")

if __name__ == "__main__":
    main()
