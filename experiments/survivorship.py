#!/usr/bin/env python3
"""
HYBRID BIGBOY - Andrew's CoalitionLockingSystem + GPU vectorization
====================================================================
Andrew's primitive: fixed mutation count (50), causal ablation, coalition
detection, convergence-based organic locking.

My addition: GPU batch evaluation, scaled steps for high dims, variance-
tracked convergence.

Enough steps to lock up hard. Push to 5M+.
Then run against academic STS-B and continual learning benchmarks.
"""
import numpy as np
import time
import cupy as cp
from scipy.stats import linregress

mempool = cp.get_default_memory_pool()
alpha_predicted = -np.log(np.cos(1/np.pi))

free_v, total_v = cp.cuda.runtime.memGetInfo()
print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"VRAM: {free_v/1e9:.2f}GB / {total_v/1e9:.2f}GB")
print(f"Target alpha: {alpha_predicted:.6f}")
print(flush=True)


class SGMBigBoy:
    """
    Andrew's CoalitionLockingSystem ported to GPU.
    Fixed mutation count (the intentional selective pressure).
    Convergence + causal + coalition locking (organic, distributed).
    """

    def __init__(self, dim):
        self.dim = dim
        self.x = cp.random.randn(dim, dtype=cp.float32) * 0.3
        self.lock = cp.zeros(dim, dtype=cp.bool_)
        self.causal_scores = cp.zeros(dim, dtype=cp.float32)
        self.causal_count = cp.ones(dim, dtype=cp.float32)
        self.coalition_credits = cp.zeros(dim, dtype=cp.float32)

        # Variance tracking for convergence detection
        self.running_sum = cp.zeros(dim, dtype=cp.float32)
        self.running_sq_sum = cp.zeros(dim, dtype=cp.float32)
        self.n_snapshots = 0

    def n_free(self):
        return int((~self.lock).sum())

    def n_locked(self):
        return int(self.lock.sum())

    def train(self, target, n_steps=200, lr=0.03, pop_size=32):
        """
        Evolutionary training. Fixed mutation count = 50.
        Batch population on GPU. Track convergence.
        """
        free_idx = cp.where(~self.lock)[0]
        n_free = len(free_idx)
        if n_free == 0:
            return float(cp.mean((self.x - target)**2))

        # FIXED mutation count - this is the primitive
        n_mutate = min(50, n_free)

        best = self.x.copy()
        best_loss = float(cp.mean((best - target)**2))

        for step in range(n_steps):
            # Batch: generate pop_size candidates
            pop = cp.tile(best, (pop_size, 1))

            # Fixed 50 mutations per member, different random indices
            for i in range(pop_size):
                idx = free_idx[cp.random.choice(n_free, n_mutate, replace=False)]
                pop[i, idx] += cp.random.randn(n_mutate, dtype=cp.float32) * lr

            # Batch evaluate
            losses = cp.mean((pop - target[None, :])**2, axis=1)
            mi = int(cp.argmin(losses))
            if float(losses[mi]) < best_loss:
                best = pop[mi].copy()
                best_loss = float(losses[mi])

            # Track for convergence (every 20 steps)
            if step % 20 == 0:
                self.running_sum += best
                self.running_sq_sum += best ** 2
                self.n_snapshots += 1

            del pop, losses

        self.x = best
        return best_loss

    def measure_causality(self, target, n_samples=100):
        """Causal importance via ablation. GPU-batched."""
        free_idx = cp.where(~self.lock)[0]
        if len(free_idx) == 0:
            return

        base_loss = float(cp.mean((self.x - target)**2))

        # Sample free dims
        n_sample = min(n_samples, len(free_idx))
        sample = free_idx[cp.random.choice(len(free_idx), n_sample, replace=False)]

        for d in sample:
            saved = float(self.x[d])
            self.x[d] = 0
            delta = float(cp.mean((self.x - target)**2)) - base_loss
            self.x[d] = saved
            self.causal_scores[d] += delta
            self.causal_count[d] += 1

        # Coalition detection: test groups of weak dims
        avg_causal = self.causal_scores[free_idx] / self.causal_count[free_idx]
        weak = free_idx[(avg_causal > 0) & (avg_causal < 0.001)]

        if len(weak) >= 3:
            for _ in range(min(30, len(weak))):
                k = min(5, len(weak))
                group = weak[cp.random.choice(len(weak), k, replace=False)]
                saved = self.x[group].copy()
                self.x[group] = 0
                delta = float(cp.mean((self.x - target)**2)) - base_loss
                self.x[group] = saved
                if delta > 0.005:
                    self.coalition_credits[group] += 1

    def update_locks(self, target, task_id=0):
        """Lock converged + causally important dims. Organic, distributed."""
        self.measure_causality(target)
        free_idx = cp.where(~self.lock)[0]

        # Convergence criterion: dims with low variance across snapshots
        converged = cp.zeros(self.dim, dtype=cp.bool_)
        if self.n_snapshots >= 3:
            mean = self.running_sum / self.n_snapshots
            var = self.running_sq_sum / self.n_snapshots - mean**2
            var = cp.maximum(var, 0)

            # Low variance = converged
            free_var = var[~self.lock]
            if len(free_var) > 0:
                threshold = float(cp.percentile(free_var, 30))  # Bottom 30% variance
                converged = (var < threshold) & (~self.lock)

        # Causal criterion
        causally_important = cp.zeros(self.dim, dtype=cp.bool_)
        avg_causal = self.causal_scores / self.causal_count
        causally_important[free_idx] = (
            (avg_causal[free_idx] > 0.0001) |
            (self.coalition_credits[free_idx] >= 2)
        )

        # Lock dims that are BOTH converged AND causally important
        # OR just converged (with looser threshold for large dims)
        new_locks = converged & causally_important
        # Also lock strongly converged dims even without causal confirmation
        if self.n_snapshots >= 5:
            very_converged = cp.zeros(self.dim, dtype=cp.bool_)
            if len(free_var) > 0:
                tight_threshold = float(cp.percentile(free_var, 15))
                very_converged = (var < tight_threshold) & (~self.lock)
            new_locks = new_locks | very_converged

        n_new = int(new_locks.sum())
        self.lock = self.lock | new_locks

        # Reset running stats for next phase
        self.running_sum *= 0
        self.running_sq_sum *= 0
        self.n_snapshots = 0

        return n_new


def test_bigboy(dim, n_tasks=5, steps_per_task=None, pop_size=32):
    """Run the hybrid SGM at given dim, measure survivorship."""
    mempool.free_all_blocks()

    if steps_per_task is None:
        # Scale steps: more dims = more steps to converge
        steps_per_task = max(200, min(1000, dim // 1000))

    # VRAM check
    free = cp.cuda.runtime.memGetInfo()[0]
    need = pop_size * dim * 4
    if need > free * 0.5:
        pop_size = max(8, int(free * 0.3 / (dim * 4)))

    print(f"  Config: {n_tasks} tasks x {steps_per_task} steps, pop={pop_size}", flush=True)

    # Create tasks
    rng = np.random.RandomState(42)
    targets = [cp.asarray(rng.randn(dim).astype(np.float32)) for _ in range(n_tasks)]
    probe = cp.asarray(rng.randn(dim).astype(np.float32))

    # Train + organic lock
    sgm = SGMBigBoy(dim)

    for t in range(n_tasks):
        loss = sgm.train(targets[t], n_steps=steps_per_task, pop_size=pop_size)
        n_locked = sgm.update_locks(targets[t], task_id=t)
        pct = sgm.n_locked() / dim
        print(f"    Task {t}: loss={loss:.4f}, +{n_locked} locked, total={pct:.1%}", flush=True)

    total_locked_pct = sgm.n_locked() / dim
    print(f"  Final lock: {total_locked_pct:.1%} ({sgm.n_locked():,} / {dim:,})", flush=True)

    if total_locked_pct < 0.05:
        print(f"  WARNING: Only {total_locked_pct:.1%} locked. Need more steps.", flush=True)

    # Build importance ordering from causal scores
    imp_order = cp.argsort(sgm.causal_scores / sgm.causal_count)[::-1]

    # Measure survivorship at each saturation
    results = []
    for lp in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        nl = int(dim * lp)
        nf = dim - nl
        if nf < 10:
            continue

        tl = cp.zeros(dim, dtype=cp.bool_)
        tl[imp_order[:nl]] = True

        # Fresh SGM for probe task, with locked dims
        probe_sgm = SGMBigBoy(dim)
        probe_sgm.lock = tl.copy()

        initial_loss = float(cp.mean((probe_sgm.x - probe)**2))
        final_loss = probe_sgm.train(probe, n_steps=steps_per_task, pop_size=pop_size)

        imp_val = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
        pd = imp_val / nf if nf > 0 else 0
        results.append({'lp': lp, 'nf': nf, 'imp': imp_val, 'pd': pd})

        del probe_sgm
        mempool.free_all_blocks()

    del sgm
    mempool.free_all_blocks()

    # Fit alpha
    xd = np.array([r['lp'] * 100 for r in results])
    yd = np.array([r['pd'] for r in results])
    if len(yd) >= 4 and yd[0] > 0:
        yn = yd / yd[0]
        valid = yn > 0
        if valid.sum() >= 4:
            slope, _, rv, pv, _ = linregress(xd[valid], np.log(yn[valid]))
            r99 = yd[-1] / yd[0] if yd[0] > 0 else 0
            return slope, rv**2, pv, r99, results
    return None, None, None, 0, results


# ============================================================
# RUN
# ============================================================

dims = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
all_results = []

for dim in dims:
    mempool.free_all_blocks()
    free = cp.cuda.runtime.memGetInfo()[0]
    print(f"\ndim={dim:>10,}: VRAM={free/1e9:.2f}GB", flush=True)

    t0 = time.time()
    try:
        alpha, r2, pval, ratio, res = test_bigboy(dim)
        elapsed = time.time() - t0

        if alpha is not None and alpha > 0:
            diff = (alpha - alpha_predicted) / alpha_predicted * 100
            d = "over" if alpha > alpha_predicted else "under"
            all_results.append((dim, alpha, r2, ratio, elapsed))
            print(f"  RESULT: alpha={alpha:.6f} ({abs(diff):.1f}% {d}), "
                  f"R^2={r2:.3f}, ratio@99%={ratio:.1f}x, {elapsed:.0f}s", flush=True)
        else:
            print(f"  RESULT: alpha={'None' if alpha is None else f'{alpha:.6f}'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for r in res:
                print(f"    {r['lp']:>5.1%}: imp={r['imp']:.4f}%, pd={r['pd']:.10f}%", flush=True)

    except cp.cuda.memory.OutOfMemoryError:
        print(f"  OOM", flush=True)
        break
    except Exception as e:
        print(f"  {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        break

# Final table
print(f"\n{'='*60}")
print(f"  HYBRID BIGBOY RESULTS")
print(f"{'='*60}")

prior = [(10000,0.013,0.957,3.8), (20000,0.017,0.943,5.7), (50000,0.024,0.915,14.8)]
combined = list(prior) + [(r[0],r[1],r[2],r[3]) for r in all_results]
combined.sort()

print(f"  {'Dim':>12} {'Alpha':>10} {'R^2':>8} {'Ratio':>14}")
print(f"  {'-'*48}")
for d,a,r2,ratio in combined:
    print(f"  {d:>12,} {a:>10.6f} {r2:>8.3f} {ratio:>14.1f}x")

print(f"\n  Target: {alpha_predicted:.6f}")
print(f"  Method: Hybrid (fixed 50 mutations, causal+coalition, convergence lock)")
print(f"  Total: {sum(r[4] for r in all_results):.0f}s", flush=True)
