"""Trening NADZOROWANY na etykiecie bh_* (1) vs reszta (0) - w przeciwienstwie
do check_bh_probe.py (ktory patrzyl na embedding z modelu uczonego cluster/
field i szukal przypadkowego grupowania), tutaj model DOSTAJE prawdziwa
etykiete 'czy to bh' wprost do CrossEntropyLoss.

Silna nierownowaga klas (11/149 = 7.4% bh) - accuracy sama w sobie
zwodnicza (trywialny 'zawsze 0' klasyfikator dostaje ~93%). Uzywam:
  - stratyfikowany split train/val (osobno dla bh i nie-bh, bez wycieku plikow)
  - wazony CrossEntropyLoss (odwrotnosc czestosci klas)
  - precision/recall/F1 dla klasy bh, nie tylko surowa acc
  - Most vs Softmax vs ZERO (ten sam protokol gate co bench_desi.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\drwis\model Ai")))

import numpy as np
import torch
import torch.nn as nn

from config import CANON
from image_batch import cutout_to_tokens, _augment_flip, _pack_batches
from desi_cutouts import CACHE, load_cached
from bridge import BridgeTransformer, SoftmaxTransformer

SEED = 0
VAL_FRAC = 0.3
EPOCHS = 60
AUGMENT = 4

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

all_files = sorted(p for p in CACHE.glob("demo_*.npz"))
files_bh, files_non = [], []
for p in all_files:
    d = load_cached(p)
    img = np.asarray(d["image"], dtype=np.float32)
    if img.shape[-1] != 128 or img.shape[-2] != 128:
        continue
    (files_bh if "bh_" in p.stem else files_non).append(p)

print(f"pliki: bh={len(files_bh)} nie-bh={len(files_non)}")

def _split(files, frac):
    idx = list(files)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * frac)))
    return idx[n_val:], idx[:n_val]

train_bh, val_bh = _split(files_bh, VAL_FRAC)
train_non, val_non = _split(files_non, VAL_FRAC)
print(f"train: bh={len(train_bh)} nie-bh={len(train_non)}  |  val: bh={len(val_bh)} nie-bh={len(val_non)}")

def _build_items(files, label, augment, rng):
    items = []
    for p in files:
        d = load_cached(p)
        base_img = np.asarray(d["image"], dtype=np.float32)
        base_T = np.asarray(d["T"], dtype=np.float32)
        variants = [(base_img, base_T)]
        for _ in range(augment):
            variants.append(_augment_flip(base_img, base_T, rng))
        for img, Tm in variants:
            c, t = cutout_to_tokens(img, Tm, patch=16, d_x=CANON.d_x, n_dim=CANON.n_dim)
            items.append((c, t, label))
    return items

train_items = (
    _build_items(train_bh, 1, AUGMENT, rng) + _build_items(train_non, 0, AUGMENT, rng)
)
val_items = (
    _build_items(val_bh, 1, 0, rng) + _build_items(val_non, 0, 0, rng)
)
rng.shuffle(train_items)

n_train_bh_items = sum(1 for *_, y in train_items if y == 1)
n_train_non_items = len(train_items) - n_train_bh_items
print(f"items train: bh={n_train_bh_items} nie-bh={n_train_non_items}  val: bh={len(val_bh)} nie-bh={len(val_non)}")

train_b = _pack_batches(train_items, batch_size=8, d_x=CANON.d_x, n_dim=CANON.n_dim)
val_b = _pack_batches(val_items, batch_size=8, d_x=CANON.d_x, n_dim=CANON.n_dim)

# wazony CE: klasa mniejszosciowa (bh) wazona wyzej
w0 = 1.0
w1 = n_train_non_items / max(1, n_train_bh_items)
class_weights = torch.tensor([w0, w1], dtype=torch.float32)
print(f"class_weights: nie-bh={w0:.2f} bh={w1:.2f}")

def train_eval_weighted(model, train, val, *, zero_tracer=False, epochs=EPOCHS, lr=CANON.lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    for _ in range(epochs):
        model.train()
        for batch in train:
            x, y = batch.content, batch.labels
            tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
            opt.zero_grad()
            logits, _, _ = model(x, tr, mask=batch.mask)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for batch in val:
            x, y = batch.content, batch.labels
            tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
            logits, _, _ = model(x, tr, mask=batch.mask)
            all_pred.append(logits.argmax(1))
            all_y.append(y)
    pred = torch.cat(all_pred)
    y = torch.cat(all_y)
    acc = (pred == y).float().mean().item()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

n_classes = 2
print("\n=== Softmax (bez tracera) ===")
torch.manual_seed(SEED)
sm = SoftmaxTransformer(input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
                         n_classes=n_classes, dim_tracer=CANON.n_dim, num_layers=CANON.num_layers)
r_sm = train_eval_weighted(sm, train_b, val_b)
print(r_sm)

print("\n=== Most (Bridge, tracer) ===")
torch.manual_seed(SEED)
most = BridgeTransformer(input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
                          n_classes=n_classes, dim_tracer=CANON.n_dim, num_layers=CANON.num_layers,
                          coupling=CANON.coupling)
r_most = train_eval_weighted(most, train_b, val_b)
print(r_most)

print("\n=== ZERO (Most z tau=0) ===")
torch.manual_seed(SEED)
most0 = BridgeTransformer(input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
                           n_classes=n_classes, dim_tracer=CANON.n_dim, num_layers=CANON.num_layers,
                           coupling=CANON.coupling)
r_zero = train_eval_weighted(most0, train_b, val_b, zero_tracer=True)
print(r_zero)

print("\n=== PODSUMOWANIE (val: bh=%d nie-bh=%d) ===" % (len(val_bh), len(val_non)))
for name, r in (("Softmax", r_sm), ("Most", r_most), ("ZERO", r_zero)):
    print(f"  {name:8s} acc={r['acc']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  "
          f"TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}")
