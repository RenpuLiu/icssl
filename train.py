#!/usr/bin/env python
# icssl_expts/train.py
import os, math, argparse, torch, wandb
from torch.utils.data import DataLoader
from data   import ICSSLDataset, sample_linear_task       # unchanged
from model  import ICSSLTransformer

# ───────────────────────── argparse ────────────────────────────
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data/task
    p.add_argument("--C",          type=int, default=3,   help="#classes")
    p.add_argument("--d",          type=int, default=16,  help="feature dim per point")
    p.add_argument("--n",          type=int, default=4,   help="#labelled / class")
    p.add_argument("--m",          type=int, default=6,   help="#unlabelled / class")
    # model
    p.add_argument("--experiment", choices=["cot", "no_cot"], default="cot")
    p.add_argument("--n_layers",   type=int, default=3,   help="layers in Transformer")
    p.add_argument("--loops",      type=int, default=1,   help="#times to re‑run the same Transformer block (no_cot)")
    p.add_argument("--d_model",    type=int, default=256)
    p.add_argument("--n_heads",    type=int, default=8)
    p.add_argument("--ff_mult",    type=int, default=4)
    # training
    p.add_argument("--T",          type=int, default=3,   help="#CoT steps (cot)")
    p.add_argument("--batch",      type=int, default=32)
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--lr",         type=float, default=1e-3)
    # logging
    p.add_argument("--wandb_project", default="ic-ssl")
    p.add_argument("--wandb_run",     default=None)
    return p.parse_args()

args = get_args()

# ─────────────────────── Weights & Biases ───────────────────────
if not os.getenv("WANDB_API_KEY"):
    os.environ["WANDB_MODE"] = "disabled"        # silent offline fallback
run = wandb.init(project=args.wandb_project,
                 name    =args.wandb_run,
                 config  =vars(args))

# ───────────────────────── dataset ──────────────────────────────
T_effective = args.T if args.experiment == "cot" else 0     # no blank slots
ds = ICSSLDataset(args.C, args.d, args.n, args.m, T_effective, size=2000)
dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

# ───────────────────────── model & aux heads ───────────────────
device  = "cuda" if torch.cuda.is_available() else "cpu"
model   = ICSSLTransformer(input_dim=args.d + args.C,   # no aux in this code
                           d_model   =args.d_model,
                           n_heads   =args.n_heads,
                           n_layers  =args.n_layers,
                           ff_mult   =args.ff_mult).to(device)

# read‑out for "no_cot" case (maps d‑dim feature → logits over C)
readout = torch.nn.Linear(args.d, args.C).to(device) if args.experiment == "no_cot" else None

opt = torch.optim.AdamW(list(model.parameters()) +
                        (list(readout.parameters()) if readout else []),
                        lr=args.lr)

mse  = torch.nn.MSELoss()
xent = torch.nn.CrossEntropyLoss()

# helper: build initial (labelled + unlabelled [+ reasoning]) sequence
def build_sequence(x_lab,y_lab,x_unlab):
    device = x_lab.device
    onehot = torch.zeros_like(y_lab[:,None].repeat(1,args.C), dtype=torch.float)
    onehot[torch.arange(y_lab.size(0)), y_lab] = 1
    lab    = torch.cat([x_lab, onehot], dim=1)                 # (C·n,d+C)
    unlab  = torch.cat([x_unlab,
                        torch.zeros(x_unlab.size(0), args.C, device=device)], dim=1)
    reasoning = (torch.zeros(args.T*args.C, args.d+args.C, device=device)
                 if args.experiment=="cot" else
                 torch.empty(0, args.d+args.C, device=device))
    return torch.cat([lab, unlab, reasoning], dim=0)           # (L, d+C)

# sequence offsets
lab_len  = args.C * args.n
unlab_len= args.C * args.m

# ───────────────────────── training loop ───────────────────────
global_step = 0
for epoch in range(args.epochs):
    for batch in dl:
        x_lab  = batch["x_lab"].to(device)
        y_lab  = batch["y_lab"].to(device)
        x_unlab= batch["x_unlab"].to(device)
        y_unlab= batch["y_unlab"].to(device)   # (B, U)
        em_tgt = batch.get("em_targets")
        if em_tgt is not None: em_tgt = em_tgt.to(device)      # (B,T,C,d)

        B = x_lab.size(0)
        seq = torch.stack([build_sequence(x_lab[i],y_lab[i],x_unlab[i]) for i in range(B)])

        loss = 0.0
        if args.experiment == "cot":
            for t in range(args.T):
                out   = model(seq)
                pred  = out[:, -args.C:, :args.d]          # (B,C,d)
                loss += mse(pred, em_tgt[:,t])
                seq   = torch.cat([seq, out[:, -args.C:, :]], dim=1)
        else:  # no_cot
            out = seq
            for _ in range(args.loops):
                out = model(out)                          # shared weights
            unlab_feat = out[:, lab_len:lab_len+unlab_len, :args.d]  # (B,U,d)
            logits = readout(unlab_feat)                  # (B,U,C)
            loss   = xent(logits.reshape(-1, args.C), y_unlab.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        global_step += 1
        wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

    print(f"Epoch {epoch}  loss {loss.item():.4f}")

wandb.finish()
