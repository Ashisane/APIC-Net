#!/bin/bash
# Run the PI-LSTM pipeline. Install deps yourself first.
# Usage: ./run_pipeline.sh

mkdir -p logs results/plots

echo "=== Step 1: Generate dataset ==="
python3 -m simulation.generate_dataset 2>&1 | tee logs/01_dataset.log

echo "=== Step 2: Standard training (50 rounds) ==="
python3 -m models.pi_lstm.train 2>&1 | tee logs/02_train_standard.log

echo "=== Step 3: Diagnostic ==="
python3 -m models.pi_lstm.diagnostic 2>&1 | tee logs/03_diagnostic.log

echo "=== Step 4: Normal-only training (50 rounds) ==="
python3 -m models.pi_lstm.train_normal_only 2>&1 | tee logs/04_train_normal_only.log

echo "=== Step 5: Dump results ==="
echo "--- pi_lstm_results.json ---" | tee logs/05_results.log
cat results/pi_lstm_results.json 2>&1 | tee -a logs/05_results.log
echo "--- pi_lstm_normal_only_results.json ---" | tee -a logs/05_results.log
cat results/pi_lstm_normal_only_results.json 2>&1 | tee -a logs/05_results.log
echo "--- diagnostic_tables.json ---" | tee -a logs/05_results.log
cat results/diagnostic_tables.json 2>&1 | tee -a logs/05_results.log

echo "=== DONE ==="
tar czf apic_logs.tar.gz logs/ results/
echo "Send back: apic_logs.tar.gz"
