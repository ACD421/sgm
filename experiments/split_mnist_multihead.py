#!/usr/bin/env python3
"""
Split-MNIST with Multi-Head Output
====================================
Each task gets its own 2-class output head.
SGM locks hidden layers after each task.
This eliminates the shared-head interference problem.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIG = Path(os.path.join(os.path.dirname(__file__), '..', 'figures'))
FIG.mkdir(exist_ok=True)


class MultiHeadMLP(nn.Module):
    """MLP with shared hidden layers + separate 2-class head per task."""
    def __init__(self, n_tasks=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(n_tasks)])

    def forward(self, x, task_id):
        h = self.shared(x.view(x.size(0), -1))
        return self.heads[task_id](h)


class SGMLock:
    def __init__(self, model):
        self.masks = {n: torch.zeros_like(p, dtype=torch.bool)
                     for n, p in model.named_parameters()}
        self.vals = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    def zero_grads(self, model):
        for n, p in model.named_parameters():
            if p.grad is not None and n in self.masks:
                p.grad[self.masks[n]] = 0

    def restore(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.masks and self.masks[n].any():
                    p[self.masks[n]] = self.vals[n][self.masks[n]]

    def lock_shared(self, model, loader, criterion, task_id, frac=0.3):
        """Lock important SHARED params only (not the task head)."""
        imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval(); nn_count = 0
        for x, y in loader:
            x, y = x.to(DEVICE), (y % 2).to(DEVICE)  # Binary labels within task
            model.zero_grad()
            loss = criterion(model(x, task_id), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    imp[n] += p.grad.abs()
            nn_count += 1
        for n in imp:
            imp[n] /= max(nn_count, 1)

        total_locked = 0
        for n, p in model.named_parameters():
            # Only lock SHARED layers, not task-specific heads
            if 'heads' in n:
                continue
            free = ~self.masks[n]
            nf = free.sum().item()
            if nf == 0:
                continue
            nl = max(1, int(nf * frac))
            fi = imp[n][free]
            if len(fi) > nl:
                thresh = torch.topk(fi.flatten(), nl).values[-1]
                new = free & (imp[n] >= thresh)
            else:
                new = free
            self.masks[n] = self.masks[n] | new
            self.vals[n][new] = p.data[new].clone()
            total_locked += new.sum().item()
        return total_locked


def get_tasks():
    tr = transforms.ToTensor()
    train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tr)
    test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tr)
    tasks = []
    for t in range(5):
        c1, c2 = t * 2, t * 2 + 1
        tr_idx = [i for i, (_, y) in enumerate(train) if y in (c1, c2)]
        te_idx = [i for i, (_, y) in enumerate(test) if y in (c1, c2)]
        tasks.append({
            'train': DataLoader(Subset(train, tr_idx), batch_size=64, shuffle=True),
            'test': DataLoader(Subset(test, te_idx), batch_size=256)
        })
    return tasks


def evaluate(model, loader, task_id):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), (y % 2).to(DEVICE)
            pred = model(x, task_id).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0


print("=" * 60)
print("  MULTI-HEAD SPLIT-MNIST: SGM vs EWC vs NAIVE")
print("  Separate output head per task. 3 seeds.")
print("=" * 60, flush=True)

all_results = {m: {'accs': [], 'bwts': [], 'per_task': []} for m in ['naive', 'ewc', 'sgm']}

for seed in range(3):
    tasks = get_tasks()
    criterion = nn.CrossEntropyLoss()

    for method in ['naive', 'ewc', 'sgm']:
        torch.manual_seed(seed * 42 + 7)
        model = MultiHeadMLP().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        sgm = SGMLock(model) if method == 'sgm' else None

        # Simple EWC
        ewc_fisher = {}
        ewc_old = {}
        ewc_lam = 1000

        per_task_accs = []

        for t, task in enumerate(tasks):
            model.train()
            for epoch in range(10):
                for x, y in task['train']:
                    x, y = x.to(DEVICE), (y % 2).to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(model(x, t), y)

                    # EWC penalty
                    if method == 'ewc' and ewc_fisher:
                        for n, p in model.named_parameters():
                            if n in ewc_fisher:
                                loss += ewc_lam * (ewc_fisher[n] * (p - ewc_old[n])**2).sum()

                    loss.backward()
                    if sgm:
                        sgm.zero_grads(model)
                    opt.step()
                    if sgm:
                        sgm.restore(model)

            # Lock shared layers (SGM)
            if sgm:
                sgm.lock_shared(model, task['train'], criterion, t)

            # Update Fisher (EWC)
            if method == 'ewc':
                for n, p in model.named_parameters():
                    ewc_fisher[n] = ewc_fisher.get(n, torch.zeros_like(p))
                model.eval()
                nn_c = 0
                for x, y in task['train']:
                    x, y = x.to(DEVICE), (y % 2).to(DEVICE)
                    model.zero_grad()
                    criterion(model(x, t), y).backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            ewc_fisher[n] += p.grad**2
                    nn_c += 1
                for n in ewc_fisher:
                    ewc_fisher[n] /= max(nn_c, 1)
                ewc_old = {n: p.data.clone() for n, p in model.named_parameters()}

            # Evaluate ALL tasks
            accs = [evaluate(model, tasks[i]['test'], i) for i in range(t + 1)]
            per_task_accs.append(accs)

        # Final evaluation
        final = [evaluate(model, tasks[i]['test'], i) for i in range(5)]
        avg = np.mean(final)
        bwt = sum(final[t] - per_task_accs[t][t] for t in range(4)) / 4

        all_results[method]['accs'].append(avg)
        all_results[method]['bwts'].append(bwt)
        all_results[method]['per_task'].append(final)

        print(f"  Seed {seed} {method:>5}: avg={avg:.1%} BWT={bwt:+.1%} "
              f"tasks={[f'{a:.0%}' for a in final]}", flush=True)

# Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
methods = ['naive', 'ewc', 'sgm']
labels = ['Naive', 'EWC', 'SGM']
colors = ['#9E9E9E', '#FF9800', '#4CAF50']

for i, (m, l, c) in enumerate(zip(methods, labels, colors)):
    ma = np.mean(all_results[m]['accs']) * 100
    sa = np.std(all_results[m]['accs']) * 100
    ax1.bar(i, ma, yerr=sa, color=c, capsize=5, edgecolor='black', linewidth=0.5)
    ax1.text(i, ma + sa + 1, f'{ma:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylabel('Avg Accuracy (%)', fontsize=12)
ax1.set_title('Multi-Head Split-MNIST: Accuracy', fontsize=14)
ax1.set_ylim(0, 105)

for i, (m, l, c) in enumerate(zip(methods, labels, colors)):
    mb = np.mean(all_results[m]['bwts']) * 100
    sb = np.std(all_results[m]['bwts']) * 100
    ax2.bar(i, mb, yerr=sb, color=c, capsize=5, edgecolor='black', linewidth=0.5)
    ax2.text(i, mb - 3 if mb < -5 else mb + 1, f'{mb:+.1f}%',
            ha='center', fontweight='bold', fontsize=12,
            color='white' if mb < -10 else 'black')
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels, fontsize=12)
ax2.set_ylabel('Backward Transfer (%)', fontsize=12)
ax2.set_title('Multi-Head Split-MNIST: BWT', fontsize=14)

plt.tight_layout()
plt.savefig(FIG / 'fig_split_mnist.png')
plt.close()

print(f"\nSaved fig_split_mnist.png")
print(f"\nFINAL:")
for m, l in zip(methods, labels):
    ma = np.mean(all_results[m]['accs']) * 100
    mb = np.mean(all_results[m]['bwts']) * 100
    print(f"  {l:>5}: {ma:.1f}% avg, {mb:+.1f}% BWT", flush=True)
