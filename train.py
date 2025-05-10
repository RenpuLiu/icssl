# icssl_expts/train.py
import os, math, torch
from torch.utils.data import DataLoader
from data import ICSSLDataset, sample_linear_task   # unchanged
from model import ICSSLTransformer                  # unchanged

# ──────────────────── 1. W&B set‑up ────────────────────────────
try:
    import wandb
    # ❶ Try a silent login with an env‑key; if it’s absent → disable wandb
    if os.getenv("WANDB_API_KEY"):
        # relogin=True skips the interactive prompt even if creds already exist
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)  # :contentReference[oaicite:0]{index=0}
        WANDB_ENABLED = True
    else:
        os.environ["WANDB_MODE"] = "disabled"      # local, no network calls
        WANDB_ENABLED = False
        print("WANDB_API_KEY not found → running with WANDB_MODE=disabled "
              "(metrics will be written to ./wandb/offline‑runs).")  # optional
except ImportError:
    # ❷ Fake a minimal wandb interface so the rest of the code stays unchanged
    WANDB_ENABLED = False
    class _Dummy:
        def init(self, *a, **k): return self
        def log(self, *a, **k): pass
        def finish(self): pass
    wandb = _Dummy()
    print("wandb package not installed → tracking disabled.")

# ──────────────────── 2. Hyper‑parameters ──────────────────────
C, d, n, m = 3, 16, 4, 6          # task dimensions
T             = 3                 # CoT iterations
aux           = 0
seq_dim       = d + C + aux
BATCH         = 32
EPOCHS        = 20
LR            = 1e-3

# ──────────────────── 3. Create / init run ─────────────────────
wandb.init(
    project="ic-ssl",
    name="cot_baseline",
    config=dict(C=C, d=d, n=n, m=m, T=T, aux=aux,
                d_model=256, batch=BATCH, epochs=EPOCHS, lr=LR)
)

# ──────────────────── 4. Helpers & data ────────────────────────
def build_sequence(x_lab, y_lab, x_unlab):
    """
    Assemble (L, d+C) sequence:
      • labelled points with one‑hot labels
      • unlabelled points with zero label slots
      • T·C blank rows reserved for CoT reasoning
    """
    device = x_lab.device
    one_hot = torch.zeros_like(y_lab[:, None].repeat(1, C), dtype=torch.float)
    one_hot[torch.arange(y_lab.size(0)), y_lab] = 1.0
    lab = torch.cat([x_lab, one_hot], dim=1)                           # (C·n, d+C)

    unlab = torch.cat([x_unlab,
                       torch.zeros(x_unlab.size(0), C, device=device)], dim=1)

    reasoning = torch.zeros(T * C, d + C, device=device)              # blank CoT

    return torch.cat([lab, unlab, reasoning], dim=0)                  # (L, d+C)


ds = ICSSLDataset(C, d, n, m, T, size=2000)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

# ──────────────────── 5. Model & optimiser ─────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = ICSSLTransformer(seq_dim).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR)
mse    = torch.nn.MSELoss()

# ──────────────────── 6. Training loop ─────────────────────────
global_step = 0
for epoch in range(EPOCHS):
    for batch in dl:
        x_lab, y_lab   = batch["x_lab"].to(device), batch["y_lab"].to(device)
        x_unlab        = batch["x_unlab"].to(device)
        em_targets     = batch["em_targets"].to(device)
        B = x_lab.size(0)

        seq = torch.stack([build_sequence(x_lab[i], y_lab[i], x_unlab[i])
                           for i in range(B)]).to(device)

        loss = 0.0
        for t in range(T):
            out    = model(seq)                     # (B,L,D)
            pred   = out[:, -C:, :d]                # (B,C,d)
            target = em_targets[:, t]               # (B,C,d)
            loss  += mse(pred, target)
            seq    = torch.cat([seq, out[:, -C:, :]], dim=1)  # grow CoT

        opt.zero_grad()
        loss.backward()
        opt.step()

        global_step += 1
        wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

    print(f"Epoch {epoch} – loss {loss.item():.4f}")

wandb.finish()
