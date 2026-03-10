import numpy as np
from models.phi_tensor.residuals import compute_residuals_vectorized

d = np.load('data/vsm_0.npz')
rec, lbl, atk = d['data'], d['labels'], d['attack_types']

NAMES = {0: 'normal', 1: 'freq', 2: 'coi', 3: 'power', 4: 'voltage'}
r_by = {k: [] for k in range(5)}

for s in range(rec.shape[0]):
    r = compute_residuals_vectorized(rec[s])
    l, a = lbl[s, 1:], atk[s, 1:]
    for c in range(5):
        m = (l == 0) if c == 0 else (a == c)
        if m.sum() > 0:
            r_by[c].append(r[m])

print(f"{'Type':<10} {'|r1| mean':>12} {'|r2| mean':>12} {'|r3| mean':>12} {'|r4| mean':>12}")
print("-" * 62)
for c, n in NAMES.items():
    if not r_by[c]:
        print(f"{n:<10} --- no samples ---")
        continue
    v = np.concatenate(r_by[c])
    m = np.mean(np.abs(v), axis=0)
    print(f"{n:<10} {m[0]:>12.6f} {m[1]:>12.6f} {m[2]:>12.6f} {m[3]:>12.6f}")
