# Khaleghi PI-LSTM Paper Analysis — Critical Comparison

## Paper Reference
**"Federated Learning Detection of Cyberattacks on Virtual Synchronous Machines Under Grid-Forming Control Using Physics-Informed LSTM"**  
Ali Khaleghi, Soroush Oshnoei, Saeed Mirzajani — *Fractal and Fractional*, 2025

---

## Their Results vs Ours

| Metric | Khaleghi (FL-PI-LSTM) | Our Reproduction |
|---|---|---|
| Precision | **95.2%** | 66.4% |
| Recall | **96.7%** | 7.5% |
| F1 | **95.5%** | 13.5% |
| Inference | 0.39 ms | 0.034 ms |

These are **very different** numbers. Before challenging their paper, we must understand why.

---

## Critical Discrepancies Found

### 1. 🔴 Swing Equation is Different

**Paper (Khaleghi):**
```
ω_phys(t+1) = ω(t) + Δt · (1/J) · [ (1/ω*)(P* - P) + D·(ω* - ω) ]
```
- Uses **J = 2×10³** (moment of inertia, SI units)
- Power term is divided by **ω\*** (not multiplied)
- **No F (COI friction) term** — only J and D

**Our simulator (vsm_simulator.py):**
```
ω_phys(t+1) = ω(t) + Δt · (1/(2H)) · [ (P* - P)·ω* + D·(ω* - ω) + F·(ω_C - ω) ]
```
- Uses **H = 5.0** (per-unit inertia constant)
- Power term is multiplied by **ω\*** (opposite scaling)
- Has **F (COI friction)** term coupling to ω_C

> [!CAUTION]
> This is the most significant discrepancy. Our physics loss is computing a different physical equation than the paper's. The paper's simpler equation (no COI term, J instead of 2H, 1/ω* instead of ·ω*) would produce very different residuals. However, **our simulator generated the data with OUR equation**, so our physics loss MUST match our simulator — not the paper's. This is **correct for our setup** but means we're not reproducing their exact physics.

### 2. 🔴 Input Features Are Different

**Paper:** 8 features  
`[P_j, Q_j, V_j, ω_j, P_ref, V_ref, I_f, V_dc]`

**Ours:** 6 features  
`[ω, p, v_dc, ω_C, p*, v_ref]`

Differences:
- Paper includes **Q_j (reactive power)** — we don't have this
- Paper includes **V_j (terminal voltage magnitude)** — we have v_dc instead
- We include **ω_C (COI frequency)** — they don't (their equation has no COI term)
- Different ordering

> [!WARNING]
> The input feature mismatch means the LSTM sees different information. Their model sees reactive power and voltage magnitude directly. Ours sees COI frequency and DC-link voltage. This is an intentional design difference (IEEE 39-bus has COI coupling) but means we can't claim exact reproduction.

### 3. 🟡 Loss Function Weighting is Different

**Paper:** `L_total = L_data + λ · L_phys` (additive, λ=0.6)  
**Ours:** `L_total = λ · L_data + (1-λ) · L_phys` (weighted blend, λ=0.6)

- Paper: L_data has weight **1.0**, L_phys has weight **0.6**
- Ours: L_data has weight **0.6**, L_phys has weight **0.4**

> [!IMPORTANT]
> In the paper, data loss dominates (weight 1.0 vs 0.6). In our code, they're both dampened (0.6 vs 0.4). This could explain poor convergence — we're giving L_data less emphasis than they do.

### 4. 🟡 Threshold Strategy

**Paper:** Uses **dynamic threshold** (adaptive based on validation variance) → **2.4% FPR**  
**Ours (fixed):** mean+3σ → gave 7.5% recall (bad). Now using 95th percentile.

The paper explicitly compared fixed vs dynamic thresholds and showed dynamic was superior. They don't specify the exact dynamic formula but it adapts per-window.

### 5. 🟢 Scale Difference (Legitimate)

**Paper:** IEEE **9-bus**, **2 VSMs**, 2000 total sequences  
**Ours:** IEEE **39-bus**, **6 VSMs**, 3500 scenarios × 2000 timesteps

This is an **intentional** improvement. A 2-VSM FL setup is trivial. Our 6-VSM setup is more realistic but also harder — more diverse operating conditions, more noise diluting attack signatures.

### 6. 🟡 Data Normalization Not Specified in Paper

The paper mentions sliding windows but doesn't explicitly state normalization method. We use Z-score (mean/std from train split). If they use min-max or no normalization, the loss landscape would be entirely different.

### 7. 🟢 FOC-VSM (Fractional Order Controller)

Paper uses a **Fractional Order PI controller** for excitation current with a "fractal factor" `a`. We simplified this to `a=1` (standard PI). This is noted in our `simulation_config.py`. Minor difference — unlikely to explain the F1 gap.

---

## Verdict: Why Our Results Are So Different

The gap is **not a single bug** but a combination of factors:

1. **Different physics** — our simulator is more complex (COI coupling, different swing equation scaling). This is correct for our dataset.
2. **More VSMs, harder problem** — 6 VSMs with diverse operating conditions dilute attack signatures
3. **Loss weighting disadvantages data loss** — paper gives L_data full weight (1.0), we give it only 0.6
4. **Conservative threshold** — now fixed with 95th percentile, but may still need tuning
5. **Input features** — we're missing reactive power Q, which may carry attack signatures

---

## What We Can Defensibly Claim

✅ **Safe to say:**
- "We adapted the PI-LSTM architecture from Khaleghi et al. to a more challenging IEEE 39-bus, 6-VSM setting"
- "The physics loss was modified to match our simulator's swing equation (which includes COI coupling)"
- "On a larger, more realistic network, the PI-LSTM baseline shows significantly degraded detection performance"

❌ **Cannot say:**
- "We exactly reproduced their method" — we didn't, and the differences are material
- "Their results are invalid" — their method works as described on their simpler setup
- "PI-LSTM fails on VSM cyberattack detection" — it works well on their setup, just not ours

---

## Recommendations

### Option A: Fix Loss Weighting (Quick)
Change loss from `λ·L_data + (1-λ)·L_phys` to `L_data + λ·L_phys` matching the paper exactly. This gives data loss full weight.

### Option B: Accept the Gap (Honest)
Frame the result as: "When scaled to a realistic 6-VSM IEEE 39-bus system, the PI-LSTM baseline's performance degrades significantly, motivating our Φ Tensor approach." This is a valid research contribution.

### Option C: Both
Fix the loss weighting AND frame the gap. If performance improves significantly with the correct loss weighting, great. If it's still poor, the architectural limitation argument is stronger.
