# icssl_expts/train.py
import os, wandb, math, torch
from torch.utils.data import DataLoader
from data import ICSSLDataset, sample_linear_task
from model import ICSSLTransformer

# ----------------------- hyper‑params & W&B setup -------------------------- #
C, d, n, m = 3, 16, 4, 6          # task dimensions
T = 3                              # CoT iterations
aux = 0
seq_dim = d + C + aux
BATCH = 32
EPOCHS = 3
LR = 1e-3
wandb.init(project="ic-ssl", name="cot_baseline", config=locals())

# ---------------------------- sequence builder ---------------------------- #
def build_sequence(x_lab, y_lab, x_unlab):
    """
    Return (seq_len, feat_dim) tensor representing labelled + unlabelled,
    plus *empty* reasoning slots (T*C cols) initialised to zeros.
    """
    one_hot = torch.zeros_like(y_lab[:, None].repeat(1, C), dtype=torch.float)
    one_hot[torch.arange(y_lab.size(0)), y_lab] = 1.0
    lab = torch.cat([x_lab, one_hot], dim=1)            # (C*n, d+C)

    unlab = torch.cat([x_unlab,
                       torch.zeros(x_unlab.size(0), C)], dim=1)  # (C*m,d+C)

    reasoning = torch.zeros(T * C, d + C)               # blank CoT blocks
    seq = torch.cat([lab, unlab, reasoning], dim=0)     # (L, d+C)
    return seq

# ------------------------------- dataloaders ------------------------------ #
ds = ICSSLDataset(C, d, n, m, T, size=2000)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

# ------------------------------- model & opt ------------------------------ #
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ICSSLTransformer(seq_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
mse = torch.nn.MSELoss()

# ----------------------------- training loop ------------------------------ #
global_step = 0
for epoch in range(EPOCHS):
    for batch in dl:
        # unpack & send to GPU
        x_lab = batch["x_lab"].to(device)          # (B,C*n,d)
        y_lab = batch["y_lab"].to(device)
        x_unlab = batch["x_unlab"].to(device)
        em_targets = batch["em_targets"].to(device)  # (B,T,C,d)

        B = x_lab.size(0)
        # build initial sequences
        seq = torch.stack([build_sequence(x_lab[i], y_lab[i], x_unlab[i])
                           for i in range(B)]).to(device)       # (B,L,D)
        # iterative CoT
        loss = 0.0
        for t in range(T):
            out = model(seq)                                    # (B,L,D)
            # predicted μ^(t): last C columns of current sequence
            pred = out[:, -C:, :d]                              # (B,C,d)
            target = em_targets[:, t]                           # (B,C,d)
            loss += mse(pred, target)

            # append the *raw* model output (not grad‑detached!) to form next seq
            seq = torch.cat([seq, out[:, -C:, :]], dim=1)       # grow sequence

        opt.zero_grad()
        loss.backward()
        opt.step()

        # logging
        global_step += 1
        wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

    print(f"Epoch {epoch}  –  loss {loss.item():.4f}")

wandb.finish()
