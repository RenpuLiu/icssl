# icssl_expts/model.py
import math, torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#                         Multi‑head Self‑Attention                           #
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V, and output projection
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x    : (B, L, d_model)
        mask : (B, L) or (B, 1, 1, L) – optional additive mask (zeros‑vs‑‑inf)
        """
        B, L, _ = x.shape

        # ----- project & reshape to heads ---------------------------------- #
        def reshape(h):
            h = h.view(B, L, self.n_heads, self.d_head)           # (B,L,H,Dh)
            return h.transpose(1, 2)                              # (B,H,L,Dh)

        q = reshape(self.W_q(x))
        k = reshape(self.W_k(x))
        v = reshape(self.W_v(x))

        # ----- scaled dot‑product attention ------------------------------- #
        scores = q @ k.transpose(-2, -1)                          # (B,H,L,L)
        scores = scores / math.sqrt(self.d_head)

        if mask is not None:
            scores += mask                                        # additive mask

        attn = self.softmax(scores)                               # (B,H,L,L)
        context = attn @ v                                        # (B,H,L,Dh)

        # ----- merge heads ------------------------------------------------- #
        context = context.transpose(1, 2).contiguous()            # (B,L,H,Dh)
        context = context.view(B, L, self.d_model)                # (B,L,d_model)
        return self.W_o(context)                                  # (B,L,d_model)


# --------------------------------------------------------------------------- #
#                     Custom Encoder Layer (no dropout)                       #
# --------------------------------------------------------------------------- #
class ICSSLEncoderLayer(nn.Module):
    """
    LN ➜ MH‑Self‑Attn ➜ +res ➜ LN ➜ FFN(softmax) ➜ +res
    """
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model, bias=True),
            nn.Softmax(dim=-1),                      # ← requested activation
            nn.Linear(ff_mult * d_model, d_model, bias=True),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)         # self‑attention
        x = x + self.ffn(self.ln2(x))                # feed‑forward
        return x


# --------------------------------------------------------------------------- #
#                       Three‑layer ICSSL Transformer                         #
# --------------------------------------------------------------------------- #
class ICSSLTransformer(nn.Module):
    """
    Encoder‑only stack; easy to tweak number of layers, heads, etc.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model, bias=False)
        self.layers = nn.ModuleList(
            [ICSSLEncoderLayer(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, input_dim, bias=False)

    def forward(self, x, mask=None):
        """
        x : (B, L, input_dim)
        """
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, mask)
        return self.out_proj(h)                      # (B, L, input_dim)


# --------------------------------------------------------------------------- #
#                             smoke‑test helper                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    B, L, D_in = 2, 10, 32
    model = ICSSLTransformer(input_dim=D_in, d_model=64, n_heads=8, n_layers=3)
    x = torch.randn(B, L, D_in)
    y = model(x)
    print("ok:", y.shape)
