"""Train the small demo model (predicts the 1-step change in wind speed) on the bundled synthetic CSV
and export its weights + demo data to site/model.json for the browser (pure-JS) forward pass.
Usage: python web/export_model.py"""
import json, os, sys, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from src.data import load_scada_csv, make_windows, split_scale
from src.train_common import fit
from src.metrics import regression_metrics
KIND = "rso"
torch.manual_seed(0); np.random.seed(0)
df = load_scada_csv("data/synthetic/synthetic_scada.csv")
LB = 24
X, Y = make_windows(df, LB, 1); last = X[:, -1, 0]
n = len(X); a = int(n * 0.7)
tr, va, te, (ym, ys) = split_scale(X, Y - last)
mu = X[:a].reshape(-1, X.shape[2]).mean(0); sd = X[:a].reshape(-1, X.shape[2]).std(0) + 1e-6
tr = (tr[0][-800:], tr[1][-800:])
from src.rso_net import RSONet
CFG = {"k": 3, "dilations": [1, 2, 4], "heads": 2}
model = RSONet(5, ch=16, dilations=(1, 2, 4), heads=2)
fit(model, tr, va, 15, lr=3e-3, verbose=False)
model.eval()
nt = len(te[0]); Xt = X[-nt:]; yt = Y[-nt:]; lt = last[-nt:]
with torch.no_grad(): pred = lt + model(torch.tensor(te[0])).numpy() * ys + ym
P = {k: v.detach().numpy().astype(float).tolist() for k, v in model.state_dict().items()}
SHOW = min(200, nt)
out = {"kind": KIND, "cfg": CFG, "lookback": LB, "mu": mu.tolist(), "sd": sd.tolist(), "ym": float(ym), "ys": float(ys), "params": P,
       "metrics": {"model": regression_metrics(yt, pred), "persistence": regression_metrics(yt, lt)},
       "demo": {"timestamps": df["timestamp"].astype(str).values[-nt:][-SHOW:].tolist(), "actual": yt[-SHOW:].tolist(),
                "persistence": lt[-SHOW:].tolist(), "py_pred": pred[-SHOW:].astype(float).tolist(),
                "windows": Xt[-SHOW:].astype(float).tolist()}}
os.makedirs("site", exist_ok=True); json.dump(out, open("site/model.json", "w"), separators=(",", ":"))
print(json.dumps(out["metrics"], indent=1))
