#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# APIC-Net Full Pipeline — Run on Remote GPU Server
# Logs EVERYTHING for debugging if things go south.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh 2>&1 | tee pipeline_master.log
#
# After completion, tar the logs folder and send back:
#   tar czf apic_logs.tar.gz logs/ results/
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

LOGDIR="logs"
RESULTSDIR="results"
mkdir -p "$LOGDIR" "$RESULTSDIR/plots"

echo "════════════════════════════════════════"
echo "  APIC-Net Pipeline — $(date)"
echo "════════════════════════════════════════"

# ── Step 0: Environment Info ──────────────────────────────────
echo ""
echo ">>> Step 0: Environment snapshot"
{
    echo "=== Date ==="
    date
    echo ""
    echo "=== Python ==="
    python3 --version
    echo ""
    echo "=== GPU ==="
    nvidia-smi 2>/dev/null || echo "No nvidia-smi found"
    echo ""
    echo "=== CUDA ==="
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not installed yet"
    echo ""
    echo "=== Disk ==="
    df -h .
    echo ""
    echo "=== RAM ==="
    free -h 2>/dev/null || echo "free not available"
} > "$LOGDIR/00_environment.log" 2>&1
cat "$LOGDIR/00_environment.log"
echo "  → Saved to $LOGDIR/00_environment.log"

# ── Step 1: Setup venv & install deps ─────────────────────────
echo ""
echo ">>> Step 1: Setting up Python environment"
{
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo "Created new venv"
    else
        echo "Using existing venv"
    fi
    source .venv/bin/activate

    echo ""
    echo "=== Installing PyTorch ==="
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo ""
    echo "=== Installing other deps ==="
    pip install numpy scikit-learn matplotlib pandas

    echo ""
    echo "=== Installed packages ==="
    pip list
} > "$LOGDIR/01_setup.log" 2>&1
source .venv/bin/activate
echo "  → Saved to $LOGDIR/01_setup.log"

# Verify torch+CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}')"

# ── Step 2: Generate dataset ──────────────────────────────────
echo ""
echo ">>> Step 2: Generating dataset"
{
    python3 -m simulation.generate_dataset
} > "$LOGDIR/02_dataset.log" 2>&1
echo "  → Saved to $LOGDIR/02_dataset.log"

# Verify dataset
echo "  Dataset files:"
ls -lh data/*.npz 2>/dev/null || echo "  ⚠ No .npz files found in data/"

# ── Step 3: Standard training (50 rounds) ─────────────────────
echo ""
echo ">>> Step 3: Standard training (50 rounds)"
echo "  Started at $(date)"
{
    python3 -m models.pi_lstm.train
} > "$LOGDIR/03_train_standard.log" 2>&1
echo "  Finished at $(date)"
echo "  → Saved to $LOGDIR/03_train_standard.log"
echo "  Last 5 lines:"
tail -5 "$LOGDIR/03_train_standard.log"

# ── Step 4: Diagnostic ───────────────────────────────────────
echo ""
echo ">>> Step 4: Running diagnostic"
{
    python3 -m models.pi_lstm.diagnostic
} > "$LOGDIR/04_diagnostic.log" 2>&1
echo "  → Saved to $LOGDIR/04_diagnostic.log"
cat "$LOGDIR/04_diagnostic.log"

# ── Step 5: Normal-only training (50 rounds) ──────────────────
echo ""
echo ">>> Step 5: Normal-only training (50 rounds)"
echo "  Started at $(date)"
{
    python3 -m models.pi_lstm.train_normal_only
} > "$LOGDIR/05_train_normal_only.log" 2>&1
echo "  Finished at $(date)"
echo "  → Saved to $LOGDIR/05_train_normal_only.log"
echo "  Last 5 lines:"
tail -5 "$LOGDIR/05_train_normal_only.log"

# ── Step 6: Dump all results ─────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  ALL RESULTS"
echo "════════════════════════════════════════"
{
    echo "=== Standard Training Results ==="
    cat "$RESULTSDIR/pi_lstm_results.json" 2>/dev/null || echo "NOT FOUND"
    echo ""
    echo "=== Normal-Only Training Results ==="
    cat "$RESULTSDIR/pi_lstm_normal_only_results.json" 2>/dev/null || echo "NOT FOUND"
    echo ""
    echo "=== Diagnostic Tables ==="
    cat "$RESULTSDIR/diagnostic_tables.json" 2>/dev/null || echo "NOT FOUND"
} > "$LOGDIR/06_all_results.log" 2>&1
cat "$LOGDIR/06_all_results.log"

# ── Step 7: Bundle everything ─────────────────────────────────
echo ""
echo ">>> Bundling logs + results for transport"
tar czf apic_logs.tar.gz logs/ results/
echo "  → Created apic_logs.tar.gz ($(du -h apic_logs.tar.gz | cut -f1))"
echo ""
echo "════════════════════════════════════════"
echo "  DONE — $(date)"
echo "  Send apic_logs.tar.gz back to your machine."
echo "════════════════════════════════════════"
