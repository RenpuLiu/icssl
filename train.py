#!/usr/bin/env python
import os, math, argparse, torch, wandb
from torch.utils.data import DataLoader
from data   import ICSSLDataset
from model  import ICSSLTransformer

# ─────────────── argparse ───────────────
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--C", type=int, default=3)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--m", type=int, default=6)
    # model
    p.add_argument("--expand",  type=int, default=2, help="read‑in expansion factor k")
    p.add_argument("--experiment", choices=["cot", "no_cot"], default="cot")
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--loops",    type=int, default=1)
    p.add_argument("--d_model",  type=int, default=256)
    p.add_argument("--n_heads",  type=int, default=8)
    p.add_argument("--ff_mult",  type=int, default=4)
    # training
    p.add_argument("--T",      type=int, default=3)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr",     type=float, default=1e-3)
    # logging
    p.add_argument("--wandb_project", default="ic-ssl")
    p.add_argument("--wandb_run",     default=None)
    return p.parse_args()

args = get_args()
base_dim   = args.d + args.C
exp_dim    = args.expand * base_dim          # new input width

# ─────────────── Weights & Biases ───────────────
if not os.getenv("WANDB_API_KEY"):
    os.environ["WANDB_MODE"] = "disabled"
run = wandb.init(project=args.wandb_project,
                 name=args.wandb_run,
                 config=vars(args))

# ─────────────── dataset ───────────────
T_eff = args.T if args.experiment == "cot" else 0
ds = ICSSLDataset(args.C, args.d, args.n, args.m, T_eff, size=2000)
dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

# ─────────────── model & read‑in ───────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

read_in = torch.nn.Sequential(
    torch.nn.Linear(base_dim, exp_dim),
    torch.nn.ReLU(),
).to(device)

model = ICSSLTransformer(
    input_dim = exp_dim,
    d_model   = args.d_model,
    n_heads   = args.n_heads,
    n_layers  = args.n_layers,
    ff_mult   = args.ff_mult,
).to(device)

readout = (torch.nn.Linear(args.d, args.C).to(device)
           if args.experiment == "no_cot" else None)

opt = torch.optim.AdamW(
    list(read_in.parameters()) +
    list(model.parameters()) +
    (list(readout.parameters()) if readout else []),
    lr=args.lr
)

mse  = torch.nn.MSELoss()
xent = torch.nn.CrossEntropyLoss()

# ─────────────── helpers ───────────────
def build_sequence_raw(x_lab,y_lab,x_unlab):
    device = x_lab.device
    onehot = torch.zeros_like(y_lab[:,None].repeat(1,args.C), dtype=torch.float)
    onehot[torch.arange(y_lab.size(0)), y_lab] = 1
    lab   = torch.cat([x_lab, onehot], dim=1)                 # (C·n, d+C)
    unlab = torch.cat([x_unlab,
                       torch.zeros(x_unlab.size(0), args.C, device=device)], dim=1)
    reasoning = (torch.zeros(args.T*args.C, base_dim, device=device)
                 if args.experiment=="cot" else
                 torch.empty(0, base_dim, device=device))
    return torch.cat([lab, unlab, reasoning], dim=0)          # (L, d+C)

lab_len   = args.C * args.n
unlab_len = args.C * args.m

# ─────────────── training loop ───────────────
global_step = 0
for epoch in range(args.epochs):
    for batch in dl:
        x_lab   = batch["x_lab"].to(device)
        y_lab   = batch["y_lab"].to(device)
        x_unlab = batch["x_unlab"].to(device)
        y_unlab = batch["y_unlab"].to(device)
        em_tgt  = batch.get("em_targets")
        if em_tgt is not None:
            em_tgt = em_tgt.to(device)

        # initial read‑in encoding
        seq_raw = torch.stack([build_sequence_raw(x_lab[i],y_lab[i],x_unlab[i])
                               for i in range(x_lab.size(0))])
        seq = read_in(seq_raw)                                # (B,L,exp_dim)

        loss = 0.0
        if args.experiment == "cot":
            for t in range(args.T):
                out = model(seq)                              # (B,L,exp_dim)
                pred = out[:, -args.C:, args.d:args.C+args.d]              # first d dims
                loss += mse(pred, em_tgt[:, t])
                seq  = torch.cat([seq, out[:, -args.C:, :]], dim=1)
        else:  # no_cot
            out = seq
            for _ in range(args.loops):
                out = model(out)
            out_pred = out[:, :, args.d:args.C+args.d]
            unlab_feat = out[:, lab_len:lab_len+unlab_len, args.d:args.C+args.d]
            # logits = readout(unlab_feat)
            logits = out_pred
            labels = torch.cat( [y_lab, y_unlab],dim=1) 
            loss   = xent(logits.reshape(-1, args.C),
                          labels.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        global_step += 1
        wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

    print(f"Epoch {epoch}  loss {loss.item():.4f}")

wandb.finish()
