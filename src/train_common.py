import json, os, numpy as np, torch
from .data import load_scada_csv, synthetic_scada, make_windows, split_scale
from .metrics import regression_metrics

def get_data(csv, n_synth, lookback, horizon, max_rows=None):
    df = load_scada_csv(csv) if csv else synthetic_scada(n_synth)
    if max_rows: df = df.tail(max_rows).reset_index(drop=True)  # --quick: use only the most recent rows
    return split_scale(*make_windows(df, lookback, horizon))

def fit(model, tr, va, epochs=10, lr=1e-3, bs=64, extra_loss=None, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.tensor(tr[0]), torch.tensor(tr[1]); Xv, Yv = torch.tensor(va[0]), torch.tensor(va[1])
    best, state = 1e9, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; p = model(Xt[idx])
            loss = torch.mean((p - Yt[idx]) ** 2)
            if extra_loss: loss = loss + extra_loss(model, Xt[idx], p)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = torch.mean((model(Xv) - Yv) ** 2).item()
        if v < best: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if verbose: print(f"epoch {ep+1}/{epochs} val_mse={v:.4f}")
    model.load_state_dict(state); return best

def evaluate(model, te, scale, out_dir="results", name="model"):
    model.eval()
    with torch.no_grad(): p = model(torch.tensor(te[0])).numpy() * scale[1] + scale[0]
    m = regression_metrics(te[1], p)
    pers = regression_metrics(te[1][1:], te[1][:-1])  # naive persistence baseline
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"model": m, "persistence_baseline": pers}, open(f"{out_dir}/{name}_metrics.json", "w"), indent=2)
    print(name, json.dumps(m)); print("persistence", json.dumps(pers)); return m
