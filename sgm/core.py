"""
SGM Core Primitive
===================
Binary parameter locking with evolutionary optimization.
The entire innovation in three lines: if locked, delta = 0.

Combined with:
- Fixed mutation count (selective pressure, not random search)
- Convergence-based organic locking (like synaptic pruning)
- Coalition detection (distributed representations)

Produces exponential survivorship amplification:
  per_dim_plasticity ~ exp(alpha * lock_percentage)
  alpha = 0.068 at dim=500K (R^2=0.984)
"""
import numpy as np


class SGMSystem:
    """
    Binary parameter locking with evolutionary optimization.

    The primitive:
        if lock_mask[i] == True:
            delta[i] = 0  # This dimension CANNOT change

    Training method: evolutionary mutation + selection (not gradient descent).
    Locking method: convergence-based (stable dimensions lock organically).
    """

    def __init__(self, dim, dtype=np.float32):
        self.dim = dim
        self.dtype = dtype
        self.x = np.random.randn(dim).astype(dtype) * 0.3
        self.lock = np.zeros(dim, dtype=bool)
        self.causal_scores = np.zeros(dim, dtype=dtype)
        self.causal_count = np.ones(dim, dtype=dtype)
        self.coalition_credits = np.zeros(dim, dtype=dtype)

    def train(self, loss_fn, n_steps=50, lr=0.03):
        """Evolutionary training: mutate free dimensions, keep improvements."""
        best_loss = loss_fn(self.x)
        free = np.where(~self.lock)[0]

        if len(free) == 0:
            return best_loss

        for step in range(n_steps):
            x2 = self.x.copy()

            if len(free) < 5:
                idx = np.random.choice(self.dim, min(30, self.dim), replace=False)
                x2[idx] += np.random.randn(len(idx)).astype(self.dtype) * lr * 0.1
            else:
                # FIXED mutation count: the selective pressure
                n_mutate = min(50, len(free))
                idx = np.random.choice(free, n_mutate, replace=False)
                x2[idx] += np.random.randn(n_mutate).astype(self.dtype) * lr

            new_loss = loss_fn(x2)
            if new_loss < best_loss:
                self.x = x2
                best_loss = new_loss

        return best_loss

    def measure_causality(self, loss_fn, n_samples=50):
        """Causal importance via ablation + coalition detection."""
        base = loss_fn(self.x)
        free = np.where(~self.lock)[0]

        if len(free) == 0:
            return

        # Individual ablation
        sample_size = min(n_samples, len(free))
        for d in np.random.choice(free, sample_size, replace=False):
            x_test = self.x.copy()
            x_test[d] = 0
            delta = loss_fn(x_test) - base
            self.causal_scores[d] += delta
            self.causal_count[d] += 1

        # Coalition detection: test groups of weak dimensions
        avg_causal = self.causal_scores[free] / self.causal_count[free]
        weak = free[(avg_causal > 0) & (avg_causal < 0.001)]

        if len(weak) >= 3:
            for _ in range(30):
                k = min(5, len(weak))
                group = np.random.choice(weak, k, replace=False)
                x_test = self.x.copy()
                x_test[group] = 0
                if loss_fn(x_test) - base > 0.005:
                    self.coalition_credits[group] += 1

    def update_locks(self, loss_fn, task_id=0):
        """Lock converged, causally important dimensions. Organic, distributed."""
        self.measure_causality(loss_fn)
        free = np.where(~self.lock)[0]

        newly_locked = 0
        for d in free:
            avg_causal = self.causal_scores[d] / self.causal_count[d]
            if avg_causal > 0.0001 or self.coalition_credits[d] >= 2:
                self.lock[d] = True
                newly_locked += 1

        return newly_locked

    def lock_region(self, start, end):
        """Explicit region lock (for structured tasks)."""
        self.lock[start:end] = True

    def n_locked(self):
        return int(self.lock.sum())

    def n_free(self):
        return int((~self.lock).sum())

    def lock_pct(self):
        return self.n_locked() / self.dim


class SGMGradientLock:
    """
    SGM locking for gradient-based training (PyTorch).

    Used for LLM knowledge preservation:
    - Lock important parameters BEFORE fine-tuning
    - Zero gradients on locked params each step
    - Restore locked values after optimizer step (handles Adam momentum)

    Result: 95% knowledge retained vs 0% naive (Qwen2.5-0.5B).
    """

    def __init__(self, model):
        import torch
        self.masks = {}
        self.vals = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.masks[name] = torch.zeros(
                    param.shape, dtype=torch.bool, device=param.device)
                self.vals[name] = torch.zeros(
                    param.shape, dtype=param.dtype, device=param.device)

    def zero_locked_gradients(self, model):
        """Zero gradients on locked dimensions before optimizer step."""
        for name, param in model.named_parameters():
            if param.grad is not None and name in self.masks:
                param.grad[self.masks[name]] = 0.0

    def restore_locked_values(self, model):
        """Restore locked values after optimizer step."""
        import torch
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.masks and self.masks[name].any():
                    param[self.masks[name]] = self.vals[name][self.masks[name]]

    def lock_by_importance(self, model, texts, tokenizer, device, frac=0.3):
        """Lock most important parameters based on gradient magnitude."""
        import torch
        importance = {n: torch.zeros_like(p)
                     for n, p in model.named_parameters() if p.requires_grad}

        model.eval()
        n = 0
        for text in texts[:20]:
            inputs = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=64).to(device)
            inputs['labels'] = inputs['input_ids'].clone()
            model.zero_grad()
            loss = model(**inputs).loss
            if not torch.isnan(loss):
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None and name in importance:
                        importance[name] += param.grad.abs()
                n += 1

        total_locked = 0
        total_params = 0
        for name, param in model.named_parameters():
            if name not in self.masks:
                continue
            total_params += param.numel()
            imp = importance.get(name)
            if imp is None:
                continue

            n_to_lock = max(1, int(param.numel() * frac))
            flat = imp.flatten()
            if len(flat) > n_to_lock:
                thresh = torch.topk(flat, n_to_lock).values[-1]
                mask = imp >= thresh
            else:
                mask = torch.ones_like(imp, dtype=torch.bool)

            self.masks[name] = mask
            self.vals[name] = param.data.clone()
            total_locked += mask.sum().item()

        return total_locked, total_locked / total_params if total_params > 0 else 0
