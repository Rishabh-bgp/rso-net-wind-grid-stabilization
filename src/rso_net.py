"""RSO-Net: dilated causal TCN + self-attention, probabilistic (quantile) head; bounded, logged, reversible
outer-loop hyperparameter controller; advisory grid-stabilization proposal with a mandatory human-approval gate."""
import json, torch, torch.nn as nn

class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k, dil):
        super().__init__(); self.pad = (k - 1) * dil; self.conv = nn.Conv1d(c_in, c_out, k, dilation=dil)
    def forward(self, x): return self.conv(nn.functional.pad(x, (self.pad, 0)))

class RSONet(nn.Module):
    QUANTILES = (0.1, 0.5, 0.9)
    def __init__(self, n_feat=5, ch=32, k=3, dilations=(1, 2, 4, 8), heads=4):
        super().__init__()
        layers, c = [], n_feat
        for d in dilations: layers += [CausalConv(c, ch, k, d), nn.ReLU()]; c = ch
        self.tcn = nn.Sequential(*layers); self.att = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.head = nn.Linear(ch, len(self.QUANTILES))
    def quantiles(self, x):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2); a, _ = self.att(h, h, h)
        q = self.head(a[:, -1]); return torch.cumsum(torch.cat([q[:, :1], nn.functional.softplus(q[:, 1:])], 1), 1)
    def forward(self, x): return self.quantiles(x)[:, 1]  # median point forecast


class BoundedController:
    """Adjusts LR between adaptation windows from rolling validation loss. Every change is bounded,
    logged and reversible (rolls back if validation worsens)."""
    def __init__(self, lr, lo=1e-4, hi=1e-2, factor=0.5):
        self.lr, self.lo, self.hi, self.f, self.log, self.prev = lr, lo, hi, factor, [], None
    def step(self, val):
        old = self.lr
        if self.prev is not None and val > self.prev: self.lr = max(self.lo, self.lr * self.f); act = "decrease_lr(rollback)"
        else: self.lr = min(self.hi, self.lr * 1.1); act = "increase_lr"
        self.log.append({"val": val, "old_lr": old, "new_lr": self.lr, "action": act}); self.prev = val; return self.lr

def stabilization_proposal(q10, q50, q90, rated_mw=50.0):
    """Advisory only: proposes reserve from forecast uncertainty band. Requires human approval; never actuates."""
    band = float(q90 - q10); reserve = round(min(rated_mw, 0.1 * rated_mw * band), 2)
    return {"proposal": f"hold {reserve} MW spinning reserve", "median_ws": float(q50), "uncertainty_band_ms": band,
            "status": "PENDING_HUMAN_APPROVAL", "auto_execute": False}
