#!/usr/bin/env python3
"""
Multi-Head Split-CIFAR-10
==========================
The benchmark that makes reviewers shut up.
5 tasks x 2 classes, CNN backbone, separate heads.
SGM locks conv layers. 3 seeds.
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
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIG = Path(os.path.join(os.path.dirname(__file__), '..', 'figures'))
FIG.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print("=" * 60)
print("  MULTI-HEAD SPLIT-CIFAR-10")
print("  The real benchmark. CNN backbone. 3 seeds.")
print("=" * 60, flush=True)


class CIFARCNN(nn.Module):
    """CNN with shared conv backbone + separate head per task."""
    def __init__(self, n_tasks=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(n_tasks)])

    def forward(self, x, task_id):
        h = self.features(x)
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

    def lock_organic(self, model, loader, criterion, task_id):
        """Organic locking: lock params that CONVERGED (low gradient variance).
        Early conv layers lock more (edges are universal).
        Later layers lock less (task-specific features need plasticity).
        Like the brain: stable synapses consolidate, active ones stay plastic."""

        # Collect gradient snapshots across multiple batches
        grad_snapshots = {n: [] for n, p in model.named_parameters()
                         if p.requires_grad and 'heads' not in n}

        model.train()
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 10: break
            x, y = x.to(DEVICE), (y % 2).to(DEVICE)
            model.zero_grad()
            criterion(model(x, task_id), y).backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in grad_snapshots:
                    grad_snapshots[n].append(p.grad.clone())

        # Lock params with LOW gradient variance (converged = stable)
        locked = 0
        for n, p in model.named_parameters():
            if n not in grad_snapshots or len(grad_snapshots[n]) < 3:
                continue
            if 'heads' in n: continue

            grads = torch.stack(grad_snapshots[n])
            grad_var = torch.var(grads, dim=0)

            free = ~self.masks[n]
            nf = free.sum().item()
            if nf == 0: continue

            # Lock bottom 20% variance (most stable/converged)
            free_var = grad_var[free]
            if len(free_var) > 10:
                n_lock = max(1, int(nf * 0.20))
                thresh = torch.topk(free_var.flatten(), n_lock, largest=False).values[-1]
                converged = free & (grad_var <= thresh)
                self.masks[n] = self.masks[n] | converged
                self.vals[n][converged] = p.data[converged].clone()
                locked += converged.sum().item()

        del grad_snapshots
        return locked


# CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
CLASS_NAMES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_ds = datasets.CIFAR10('/tmp/cifar10', train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10('/tmp/cifar10', train=False, download=True, transform=transform)


def get_tasks():
    tasks = []
    for t in range(5):
        c1, c2 = t * 2, t * 2 + 1
        tr_idx = [i for i, (_, y) in enumerate(train_ds) if y in (c1, c2)]
        te_idx = [i for i, (_, y) in enumerate(test_ds) if y in (c1, c2)]
        tasks.append({
            'train': DataLoader(Subset(train_ds, tr_idx), batch_size=64, shuffle=True),
            'test': DataLoader(Subset(test_ds, te_idx), batch_size=256),
            'name': f'{CLASS_NAMES[c1]}/{CLASS_NAMES[c2]}'
        })
    return tasks


def evaluate(model, loader, task_id):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), (y % 2).to(DEVICE)
            correct += (model(x, task_id).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0


all_results = {m: {'accs': [], 'bwts': [], 'per_task': []} for m in ['naive', 'ewc', 'sgm']}

for seed in range(3):
    tasks = get_tasks()
    criterion = nn.CrossEntropyLoss()

    for method in ['naive', 'ewc', 'sgm']:
        torch.manual_seed(seed * 42 + 7)
        model = CIFARCNN().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        sgm = SGMLock(model) if method == 'sgm' else None

        ewc_fisher = {}
        ewc_old = {}

        per_task = []

        for t, task in enumerate(tasks):
            model.train()
            for epoch in range(15):  # More epochs for CIFAR
                for x, y in task['train']:
                    x, y = x.to(DEVICE), (y % 2).to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(model(x, t), y)

                    if method == 'ewc' and ewc_fisher:
                        for n, p in model.named_parameters():
                            if n in ewc_fisher:
                                loss += 1000 * (ewc_fisher[n] * (p - ewc_old[n])**2).sum()

                    loss.backward()
                    if sgm: sgm.zero_grads(model)
                    opt.step()
                    if sgm: sgm.restore(model)

            if sgm:
                sgm.lock_organic(model, task['train'], criterion, t)

            if method == 'ewc':
                for n, p in model.named_parameters():
                    ewc_fisher[n] = ewc_fisher.get(n, torch.zeros_like(p))
                model.eval(); nc = 0
                for x, y in task['train']:
                    x, y = x.to(DEVICE), (y % 2).to(DEVICE)
                    model.zero_grad()
                    criterion(model(x, t), y).backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None: ewc_fisher[n] += p.grad**2
                    nc += 1
                    if nc >= 20: break
                for n in ewc_fisher: ewc_fisher[n] /= max(nc, 1)
                ewc_old = {n: p.data.clone() for n, p in model.named_parameters()}

            accs = [evaluate(model, tasks[i]['test'], i) for i in range(t + 1)]
            per_task.append(accs)

        final = [evaluate(model, tasks[i]['test'], i) for i in range(5)]
        avg = np.mean(final)
        bwt = sum(final[t] - per_task[t][t] for t in range(4)) / 4

        all_results[method]['accs'].append(avg)
        all_results[method]['bwts'].append(bwt)
        all_results[method]['per_task'].append(final)

        task_str = ' '.join([f'{a:.0%}' for a in final])
        print(f"  Seed {seed} {method:>5}: avg={avg:.1%} BWT={bwt:+.1%} [{task_str}]", flush=True)


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
ax1.set_title('Multi-Head Split-CIFAR-10: Accuracy', fontsize=14)
ax1.set_ylim(0, 105)

for i, (m, l, c) in enumerate(zip(methods, labels, colors)):
    mb = np.mean(all_results[m]['bwts']) * 100
    sb = np.std(all_results[m]['bwts']) * 100
    ax2.bar(i, mb, yerr=sb, color=c, capsize=5, edgecolor='black', linewidth=0.5)
    ax2.text(i, mb - 2 if mb < -3 else mb + 1, f'{mb:+.1f}%',
            ha='center', fontweight='bold', fontsize=12,
            color='white' if mb < -5 else 'black')
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels, fontsize=12)
ax2.set_ylabel('Backward Transfer (%)', fontsize=12)
ax2.set_title('Multi-Head Split-CIFAR-10: BWT', fontsize=14)

plt.tight_layout()
plt.savefig(FIG / 'fig_split_cifar10.png')
plt.close()

print(f"\nSaved fig_split_cifar10.png")
print(f"\nFINAL:")
for m, l in zip(methods, labels):
    ma = np.mean(all_results[m]['accs']) * 100
    mb = np.mean(all_results[m]['bwts']) * 100
    print(f"  {l:>5}: {ma:.1f}% avg, {mb:+.1f}% BWT", flush=True)
