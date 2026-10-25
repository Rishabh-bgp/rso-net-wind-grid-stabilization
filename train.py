"""Train/evaluate RSO-Net (quantile TCN-attention) with bounded self-optimising LR controller.
Example: python train.py --epochs 3"""
import argparse, json, os, torch
from src.train_common import get_data, evaluate
from src.rso_net import RSONet, BoundedController, stabilization_proposal
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthetic/synthetic_scada.csv", help="SCADA-style CSV; pass \"\" to generate synthetic data in memory"); ap.add_argument("--n_synth", type=int, default=3000); ap.add_argument("--lookback", type=int, default=32)
ap.add_argument("--horizon", type=int, default=1); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); torch.manual_seed(a.seed)
tr, va, te, sc = get_data(a.csv, a.n_synth, a.lookback, a.horizon)
m = RSONet(tr[0].shape[2]); ctl = BoundedController(2e-3); opt = torch.optim.Adam(m.parameters(), lr=ctl.lr)
Q = torch.tensor(RSONet.QUANTILES); Xt, Yt = torch.tensor(tr[0]), torch.tensor(tr[1]); Xv, Yv = torch.tensor(va[0]), torch.tensor(va[1])
best, snap = 1e9, None
for ep in range(a.epochs):  # each epoch = one adaptation window
    m.train(); perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 64):
        idx = perm[i:i + 64]; q = m.quantiles(Xt[idx]); e = Yt[idx, None] - q
        loss = torch.mean(torch.maximum(Q * e, (Q - 1) * e)); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): v = torch.mean((m(Xv) - Yv) ** 2).item()
    if v < best: best, snap = v, {k: t.clone() for k, t in m.state_dict().items()}
    else: m.load_state_dict(snap)  # reversible: roll back to best weights
    lr = ctl.step(v)
    for g in opt.param_groups: g["lr"] = lr
    print(f"window {ep+1}/{a.epochs} val_mse={v:.4f} lr->{lr:.5f}")
m.load_state_dict(snap); evaluate(m, te, sc, name="rso_net")
os.makedirs("results", exist_ok=True); json.dump(ctl.log, open("results/controller_log.json", "w"), indent=2)
with torch.no_grad(): q = m.quantiles(torch.tensor(te[0][-1:]))[0].numpy() * sc[1] + sc[0]
print("advisory:", json.dumps(stabilization_proposal(*q)))
