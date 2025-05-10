# icssl_expts/data.py
import torch
from torch.utils.data import Dataset
from typing import List


def sample_linear_task(C: int, d: int, n: int, m: int) -> dict:

    means = torch.randn(C, d)                        # true class means 𝑚_c
    x_lab = torch.cat([means[c] + 0.1*torch.randn(n, d) for c in range(C)])
    y_lab = torch.repeat_interleave(torch.arange(C), n)
    x_unlab = torch.cat([means[c] + 0.1*torch.randn(m, d) for c in range(C)])
    y_unlab = torch.repeat_interleave(torch.arange(C), m)

    return dict(means=means,
                x_lab=x_lab, y_lab=y_lab,
                x_unlab=x_unlab, y_unlab=y_unlab)



# ------------------------------- EM ORACLE --------------------------------- #
@torch.no_grad()
def em_means(
    C: int,
    d: int,
    n: int,
    m: int,
    x_lab: torch.Tensor,
    y_lab: torch.Tensor,
    x_unlab: torch.Tensor,
    T: int,
) -> List[torch.Tensor]:
    """
    Classic EM (isotropic Gaussians, equal priors).  Returns a list
    [μ^(0), μ^(1), … μ^(T)] where μ^(0) uses only labelled samples.
    """
    # init = empirical means of labelled data
    mu = torch.stack([x_lab[y_lab == c].mean(0) for c in range(C)])  # (C,d)
    out = [mu.clone()]

    cov_I = torch.eye(d)
    for _ in range(T):
        # E‑step: p(c | x) ∝ N(x | μ_c, I)
        logp = -(x_unlab.unsqueeze(1) - mu).pow(2).sum(-1) / 2       # (U,C)
        w = logp.softmax(-1)                                         # posteriors
        # M‑step
        num = torch.zeros_like(mu)
        den = torch.zeros(C)
        for c in range(C):
            num[c] = x_lab[y_lab == c].sum(0) + (w[:, c][:, None] * x_unlab).sum(0)
            den[c] = n + w[:, c].sum()
        mu = num / den[:, None]
        out.append(mu.clone())
    return out  # length T+1


# ------------------------------  PyTorch DS  -------------------------------- #
class ICSSLDataset(Dataset):
    def __init__(self, C: int, d: int, n: int, m: int, T: int, size: int):
        self.C, self.d, self.n, self.m, self.T = C, d, n, m, T
        self.size = size

    def __len__(self): return self.size

    def __getitem__(self, idx):
        task = sample_linear_task(self.C, self.d, self.n, self.m)
        em_targets = em_means(self.C, self.d, self.n, self.m,
                              task["x_lab"], task["y_lab"],
                              task["x_unlab"], self.T)              # (T+1,C,d)
        if self.T > 0:
            task["em_targets"] = torch.stack(em_targets[1:])            # discard step‑0
        return task
