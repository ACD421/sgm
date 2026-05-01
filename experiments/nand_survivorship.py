#!/usr/bin/env python3
"""
NAND GateMesh Survivorship Amplification
==========================================
Tests whether the survivorship effect is substrate-independent.
Measures per-gate improvement at different lock percentages
on a Boolean NAND circuit mesh (5 bytes/gate).

Result: alpha=0.026, R^2=0.794, ratio@95%=16.7x
Confirms: the same exponential emerges in discrete Boolean space.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import linregress
from sgm.gates import GateMesh

W = 512
DEPTH = 16

print('SURVIVORSHIP AMPLIFICATION IN NAND SPACE')
print('=' * 55)

n_patterns = 64
np.random.seed(42)
test_inputs = [np.random.randint(0, 2, W, dtype=np.uint8) for _ in range(n_patterns)]
test_outputs = [np.random.randint(0, 2, W, dtype=np.uint8) for _ in range(n_patterns)]


def eval_mapping(mesh, inputs, outputs):
    total_correct = total_bits = 0
    for inp, exp in zip(inputs, outputs):
        out = mesh.forward_vectorized(inp, residual=False)
        total_correct += np.sum(out == exp)
        total_bits += len(exp)
    return total_correct / total_bits


def train_mesh(mesh, inputs, outputs, n_gen=500):
    best = eval_mapping(mesh, inputs, outputs)
    initial = best
    for g in range(n_gen):
        wa, wb, tt = mesh.wire_a.copy(), mesh.wire_b.copy(), mesh.truth_tables.copy()
        mesh.mutate_unlocked(rate=0.02)
        s = eval_mapping(mesh, inputs, outputs)
        if s >= best:
            best = s
        else:
            mesh.wire_a[:], mesh.wire_b[:], mesh.truth_tables[:] = wa, wb, tt
    return initial, best


# Pretrain + organic lock
mesh = GateMesh(depth=DEPTH, width=W)
total_gates = DEPTH * W
print(f'Mesh: {DEPTH}x{W} = {total_gates} gates')

pretrain_mappings = []
for t in range(5):
    rng = np.random.RandomState(t + 10)
    inp = [rng.randint(0, 2, W, dtype=np.uint8) for _ in range(32)]
    out = [rng.randint(0, 2, W, dtype=np.uint8) for _ in range(32)]
    pretrain_mappings.append((inp, out))

print('Pretraining...')
for t, (inp, out) in enumerate(pretrain_mappings):
    ini, fin = train_mesh(mesh, inp, out, n_gen=300)
    free_mask = mesh.locked == 0
    n_free = free_mask.sum()
    if n_free > 100:
        n_lock = max(1, int(n_free * 0.15))
        change = np.random.rand(DEPTH, W).astype(np.float32)
        free_changes = change[free_mask]
        threshold = np.partition(free_changes, n_lock)[n_lock]
        converged = free_mask & (change <= threshold)
        for idx in mesh.plastic_indices:
            converged[:, idx] = False
        mesh.locked[converged] = 1
    pct = mesh.n_locked() / total_gates * 100
    print(f'  Task {t}: {ini:.3f} -> {fin:.3f}, locked={pct:.0f}%')

# Gate order for lock sweep
gate_order = [(d, w) for d in range(DEPTH - 1, -1, -1) for w in range(W)]

print(f'\nSURVIVORSHIP MEASUREMENT:')
results = []
for lock_pct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    test_mesh = GateMesh(depth=DEPTH, width=W)
    test_mesh.wire_a[:] = mesh.wire_a
    test_mesh.wire_b[:] = mesh.wire_b
    test_mesh.truth_tables[:] = mesh.truth_tables
    test_mesh.locked[:] = 0

    n_lock = int(total_gates * lock_pct)
    for i in range(n_lock):
        d, w = gate_order[i % len(gate_order)]
        if w not in test_mesh.plastic_indices:
            test_mesh.locked[d, w] = 1

    n_free = total_gates - test_mesh.n_locked()
    ini, fin = train_mesh(test_mesh, test_inputs, test_outputs, n_gen=500)
    imp = (fin - ini) / max(ini, 0.001) * 100
    pg = imp / max(n_free, 1)
    results.append({'lp': lock_pct, 'nf': n_free, 'imp': imp, 'pg': pg})
    print(f'  {lock_pct:>5.0%} locked ({n_free:>5} free): per_gate={pg:.4f}%')

# Fit
xd = np.array([r['lp'] * 100 for r in results])
yd = np.array([r['pg'] for r in results])
if yd[0] > 0:
    yn = yd / yd[0]
    valid = yn > 0
    if valid.sum() >= 4:
        slope, _, rv, pv, _ = linregress(xd[valid], np.log(yn[valid]))
        print(f'\nalpha = {slope:.6f}')
        print(f'R^2   = {rv**2:.4f}')
        print(f'ratio@95% = {yd[-1]/yd[0]:.1f}x')
        print(f'SURVIVORSHIP IN BOOLEAN SPACE: {"YES" if slope > 0.01 else "NO"}')
