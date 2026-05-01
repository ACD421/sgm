#!/usr/bin/env python3
"""
SGM-ONLY LLM TEST
===================
Naive data already captured. Just run SGM with fp16 + AMP.
Separate script = no OOM from model reload.
"""
import torch
import torch.nn as nn
import torch.amp
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device('cuda')
FIG = Path("figures")

torch.cuda.empty_cache()
gc.collect()

print("Loading Qwen2.5-0.5B (fp16 + AMP)...", flush=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.0f}M params, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

# Knowledge probes
PROBES = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The largest ocean is the", "Pacific"),
    ("Water boils at", "100"),
    ("The chemical formula for water is", "H2O"),
    ("The square root of 144 is", "12"),
    ("Pi is approximately 3.", "14"),
    ("The past tense of 'go' is", "went"),
    ("The plural of 'child' is", "children"),
    ("The longest river in the world is the", "Nile"),
    ("The speed of light is approximately", "300"),
    ("Gravity on Earth is approximately 9.8", "m/s"),
]

FINETUNE_DATA = [
    "The function def hello(): print('hello world') is a basic Python function.",
    "In Python, a list is defined using square brackets: my_list = [1, 2, 3].",
    "To iterate over a list in Python: for item in my_list: print(item).",
    "Python dictionaries use curly braces: my_dict = {'key': 'value'}.",
    "The len() function returns the length of a sequence in Python.",
    "Python uses indentation to define code blocks instead of curly braces.",
    "A Python class is defined using: class MyClass: def __init__(self): pass.",
    "Python f-strings are formatted like: f'Hello {name}' for string interpolation.",
    "The range() function generates a sequence: range(start, stop, step).",
    "Python supports list comprehension: [x**2 for x in range(10)].",
    "Import modules in Python using: import numpy as np.",
    "Python exception handling: try: risky() except Exception as e: print(e).",
    "Lambda functions in Python: square = lambda x: x**2.",
    "Python decorators use @syntax: @staticmethod before a method definition.",
    "The __name__ == '__main__' idiom prevents code from running on import.",
] * 10


def eval_knowledge(model):
    model.eval()
    scores = []
    for prompt, expected in PROBES:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=0)
        expected_ids = tokenizer.encode(expected, add_special_tokens=False)
        if expected_ids:
            scores.append(float(probs[expected_ids[0]]))
        else:
            scores.append(0.0)
    return np.mean(scores)


def eval_ppl(model, texts):
    model.eval()
    total_loss = total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        inputs['labels'] = inputs['input_ids'].clone()
        with torch.no_grad():
            loss = model(**inputs).loss
        if not torch.isnan(loss):
            total_loss += loss.item() * inputs['input_ids'].shape[1]
            total_tokens += inputs['input_ids'].shape[1]
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# Baseline
baseline = eval_knowledge(model)
baseline_ppl = eval_ppl(model, ["The capital of France is Paris.", "Water boils at 100 degrees."])
print(f"Baseline knowledge: {baseline:.4f}, ppl: {baseline_ppl:.2f}", flush=True)

# Lock knowledge params BEFORE fine-tuning
print("Locking knowledge params...", flush=True)
lock_masks = {}
lock_vals = {}

# Compute importance on knowledge probes
importance = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

model.eval()
for prompt, expected in PROBES:
    text = f"{prompt} {expected}."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    inputs['labels'] = inputs['input_ids'].clone()
    model.zero_grad()
    loss = model(**inputs).loss
    if not torch.isnan(loss):
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n in importance:
                importance[n] += p.grad.abs()

# Lock top 30%
total_locked = 0
total_params = 0
for n, p in model.named_parameters():
    if n not in importance:
        continue
    total_params += p.numel()
    imp = importance[n]
    n_to_lock = max(1, int(p.numel() * 0.3))
    flat_imp = imp.flatten()
    if len(flat_imp) > n_to_lock:
        thresh = torch.topk(flat_imp, n_to_lock).values[-1]
        mask = imp >= thresh
    else:
        mask = torch.ones_like(imp, dtype=torch.bool)
    lock_masks[n] = mask
    lock_vals[n] = p.data.clone()
    total_locked += mask.sum().item()

lock_pct = total_locked / total_params
print(f"Locked {lock_pct:.1%} of {total_params/1e6:.0f}M params", flush=True)

# Fine-tune with SGM + AMP
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=5e-5)

knowledge_trajectory = []
ppl_trajectory = []

print(f"Fine-tuning on {len(FINETUNE_DATA)} Python examples with SGM locking...", flush=True)

for epoch in range(5):
    model.train()
    epoch_loss = 0
    n_steps = 0

    for text in FINETUNE_DATA:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        inputs['labels'] = inputs['input_ids'].clone()

        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs).loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()

        # Clip + SGM lock grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for n, p in model.named_parameters():
            if p.grad is not None and n in lock_masks:
                p.grad[lock_masks[n]] = 0.0

        optimizer.step()

        # SGM: restore locked values
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in lock_masks and lock_masks[n].any():
                    p[lock_masks[n]] = lock_vals[n][lock_masks[n]]

        epoch_loss += loss.item()
        n_steps += 1

    # Eval
    score = eval_knowledge(model)
    ppl = eval_ppl(model, ["The capital of France is Paris.", "Water boils at 100 degrees."])
    retention = score / baseline if baseline > 0 else 0
    knowledge_trajectory.append(score)
    ppl_trajectory.append(ppl)

    print(f"  SGM Epoch {epoch}: loss={epoch_loss/max(n_steps,1):.4f} "
          f"knowledge={score:.4f} retention={retention:.2f}x ppl={ppl:.1f}", flush=True)

# Naive data (from previous runs)
naive_knowledge = [0.0161, 0.0355, 0.0008, 0.0000, 0.0000]
naive_ppl = [334.2, 581.0, 2393.1, 17387.5, 112036.0]

# Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(5)
ax1.plot(epochs, [n/baseline for n in naive_knowledge], 's-', color='#9E9E9E', label='Naive')
ax1.plot(epochs, [s/baseline for s in knowledge_trajectory], 'o-', color='#4CAF50', label='SGM (30% locked)')
ax1.axhline(y=1.0, color='black', linestyle=':', alpha=0.3)
ax1.set_xlabel('Fine-tuning Epoch')
ax1.set_ylabel('Knowledge Retention')
ax1.set_title('Qwen2.5-0.5B: Knowledge During Python Fine-tuning')
ax1.legend()

ax2.semilogy(epochs, naive_ppl, 's-', color='#9E9E9E', label='Naive')
ax2.semilogy(epochs, ppl_trajectory, 'o-', color='#4CAF50', label='SGM')
ax2.axhline(y=baseline_ppl, color='gray', linestyle=':', alpha=0.5, label=f'Baseline ({baseline_ppl:.1f})')
ax2.set_xlabel('Fine-tuning Epoch')
ax2.set_ylabel('General Knowledge Perplexity (log)')
ax2.set_title('Perplexity on General Knowledge')
ax2.legend()

plt.tight_layout()
plt.savefig(FIG / 'fig_llm_sgm_vs_naive.png')
plt.close()

print(f"\nSaved fig_llm_sgm_vs_naive.png", flush=True)
print(f"\nFINAL:", flush=True)
print(f"  Naive: {baseline:.4f} -> {naive_knowledge[-1]:.4f} ({naive_knowledge[-1]/baseline:.0%} retained)", flush=True)
print(f"  SGM:   {baseline:.4f} -> {knowledge_trajectory[-1]:.4f} ({knowledge_trajectory[-1]/baseline:.0%} retained)", flush=True)
