#!/usr/bin/env bash
# ───────────────────────────────
# Sweep m (unlabeled / class) to
# observe accuracy gains without CoT
# ───────────────────────────────

# ===== default task / model params =====
C=3            # number of classes
d=16           # feature dimension
n=4            # labelled per class
EPOCHS=5
BATCH=32
LR=1e-3
DMODEL=256
PROJECT=ic-ssl    # W&B project

# list of unlabeled counts to test
UNLAB_LIST=(0 2 4 6 8 12)

# ===== 1.  Twelve‑layer Transformer  =====
for M in "${UNLAB_LIST[@]}"; do
  python train.py \
    --experiment  no_cot \
    --C $C --d $d --n $n --m $M \
    --n_layers 12  --loops 1 \
    --d_model $DMODEL \
    --epochs $EPOCHS --batch $BATCH --lr $LR \
    --wandb_project $PROJECT \
    --wandb_run "12L_m${M}"
done

# ===== 2.  3‑layer block looped ×4 (depth ≈ 12)  =====
for M in "${UNLAB_LIST[@]}"; do
  python train.py \
    --experiment  no_cot \
    --C $C --d $d --n $n --m $M \
    --n_layers 3   --loops 4 \
    --d_model $DMODEL \
    --epochs $EPOCHS --batch $BATCH --lr $LR \
    --wandb_project $PROJECT \
    --wandb_run "3Lx4_m${M}"
done
