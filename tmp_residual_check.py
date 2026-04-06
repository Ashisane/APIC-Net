"""
Option A Smoke Test — COI upstream spoof.

For each COI scenario:
1. True correlation: pearsonr(injected_spoof_bias, r4 at VSM j)
2. r4 at VSM j mean (target) during attack
3. r4 at other VSMs mean during attack
4. Normal r4 baseline

Gate targets:
- Correlation at VSM j < 0.70
- r4 at other VSMs: SNR 1.5-5.0x (they are the detector)
- r4 at VSM j: can be anything (VSM j is the compromised node, not the detector)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr

from simulation.vsm_simulator import VSMSimulator
from simulation.attack_generator import generate_attack_params, make_attack_fn
from configs.simulation_config import N_VSM, OMEGA_STAR, N_STEPS

NORMAL_R4_STD = 0.52
N = 30

sim = VSMSimulator(seed=99)
rng = np.random.default_rng(99)

# ── Normal baseline ──
print("Computing normal r4 baseline (10 scenarios)...")
normal_r4 = []
for _ in range(10):
    rec = sim.simulate_scenario(load_factors=rng.uniform(0.8, 1.2, N_VSM))
    if rec is not None:
        r4 = np.abs(rec[:, :, 6] - rec[:, :, 0])
        normal_r4.extend(r4.flatten())
normal_r4_mean = np.mean(normal_r4)
normal_r4_std  = np.std(normal_r4)
print(f"  Normal r4 mean={normal_r4_mean:.4f}, std={normal_r4_std:.4f}")

# ── COI Option A smoke ──
corr_at_j     = []
r4_at_j       = []
r4_at_others  = []

print(f"\nRunning {N} COI scenarios (Option A)...")
for i in range(N):
    params = generate_attack_params("coi", target_vsm=i % N_VSM, rng=rng)
    attack_fn, coi_spoof_fn = make_attack_fn(params)
    assert coi_spoof_fn is not None, "Expected coi_spoof_fn!"
    assert attack_fn is None, "Expected attack_fn=None for COI!"

    ts = params["trigger_step"]
    es = params["end_step"]
    tv = params["target_vsm"]
    bias_amp = params["amplitude"]
    dur = max(es - ts, 1)

    # Reproduce the exact injected bias
    rng_walk = np.random.default_rng(
        hash((ts, tv, int(bias_amp * 1000))) % (2**31)
    )
    steps = rng_walk.standard_normal(dur) * 0.18
    injected_bias = np.cumsum(steps)
    injected_bias = np.clip(injected_bias, -bias_amp, bias_amp)

    rec = sim.simulate_scenario(
        load_factors=rng.uniform(0.8, 1.2, N_VSM),
        attack_fn=None,
        coi_spoof_fn=coi_spoof_fn,
    )
    if rec is None or (es - ts) < 5:
        continue

    # r4 at VSM j (target)
    omega_j  = rec[ts:es, tv, 0]
    omegaC_j = rec[ts:es, tv, 6]
    r4_j     = np.abs(omegaC_j - omega_j)
    r4_at_j.append(r4_j.mean())

    # r4 at other VSMs
    other_vsms = [v for v in range(N_VSM) if v != tv]
    r4_others = []
    for v in other_vsms:
        r4_v = np.abs(rec[ts:es, v, 6] - rec[ts:es, v, 0])
        r4_others.append(r4_v.mean())
    r4_at_others.append(np.mean(r4_others))

    # True correlation at VSM j
    r4_j_signed = omegaC_j - omega_j
    if len(injected_bias) == len(r4_j_signed) and len(injected_bias) > 2:
        r, _ = pearsonr(injected_bias, r4_j_signed)
        corr_at_j.append(r)

print("\n" + "=" * 60)
print("OPTION A SMOKE TEST RESULTS")
print("=" * 60)

avg_corr_j      = np.mean(corr_at_j) if corr_at_j else float('nan')
avg_r4_j        = np.mean(r4_at_j) if r4_at_j else 0
avg_r4_others   = np.mean(r4_at_others) if r4_at_others else 0
snr_j           = avg_r4_j / NORMAL_R4_STD
snr_others      = avg_r4_others / NORMAL_R4_STD

print(f"  Injection-r4 correlation at VSM j: {avg_corr_j:.4f}")
print(f"  r4 at VSM j (target):  mean={avg_r4_j:.4f}, SNR={snr_j:.2f}x")
print(f"  r4 at other VSMs:      mean={avg_r4_others:.4f}, SNR={snr_others:.2f}x")

print("\n  Gate checks:")
g1 = avg_corr_j < 0.70
g2 = 1.5 <= snr_others <= 5.0
print(f"  Correlation at j < 0.70: {avg_corr_j:.4f} → {'✅ PASS' if g1 else '❌ FAIL'}")
print(f"  SNR at other VSMs 1.5-5.0x: {snr_others:.2f}x → {'✅ PASS' if g2 else '❌ FAIL'}")

if g1 and g2:
    print("\n  ALL GATES PASSED ✅ — Option A is working correctly")
else:
    print("\n  GATES FAILED ❌")
