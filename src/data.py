"""SCADA-style CSV loader and offline synthetic fallback generator."""
import numpy as np, pandas as pd

COLS = {"timestamp": ["date/time", "timestamp", "time"],
        "active_power": ["lv activepower (kw)", "active_power", "activepower"],
        "wind_speed": ["wind speed (m/s)", "wind_speed", "windspeed"],
        "theoretical_power": ["theoretical_power_curve (kwh)", "theoretical_power"],
        "wind_direction": ["wind direction (°)", "wind direction (deg)", "wind_direction"]}

def load_scada_csv(path):
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    out = {}
    for k, alts in COLS.items():
        for a in alts:
            if a in low: out[k] = df[low[a]]; break
        else: raise ValueError(f"column for '{k}' not found; expected one of {alts}")
    out = pd.DataFrame(out)
    out["timestamp"] = pd.to_datetime(out["timestamp"], dayfirst=True, errors="coerce")
    return out.dropna().sort_values("timestamp").reset_index(drop=True)

def power_curve(v, rated=3600.0, cut_in=3.0, rated_v=12.5, cut_out=25.0):
    p = rated * np.clip((v - cut_in) / (rated_v - cut_in), 0, 1) ** 3
    return np.where(v > cut_out, 0.0, p)

def synthetic_scada(n=3000, seed=0, freq="10min"):
    """Diurnal + AR(1) turbulence wind speed, Weibull-like marginals, power curve with noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    steps_day = 144
    base = 7 + 2.0 * np.sin(2 * np.pi * t / steps_day) + 1.5 * np.sin(2 * np.pi * t / (steps_day * 7))
    ar = np.zeros(n)
    for i in range(1, n): ar[i] = 0.95 * ar[i - 1] + rng.normal(0, 0.45)
    ws = np.clip(base + ar, 0, 30)
    wd = (200 + 40 * np.sin(2 * np.pi * t / (steps_day * 3)) + np.cumsum(rng.normal(0, 2, n))) % 360
    tp = power_curve(ws)
    ap = np.clip(tp * rng.normal(0.95, 0.05, n) - rng.normal(0, 20, n), 0, None)
    return pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=n, freq=freq),
                         "active_power": ap, "wind_speed": ws, "theoretical_power": tp, "wind_direction": wd})

def make_windows(df, lookback=24, horizon=1, target="wind_speed"):
    d = df.copy()
    rad = np.deg2rad(d["wind_direction"])
    feats = np.stack([d["wind_speed"], np.sin(rad), np.cos(rad), d["theoretical_power"] / 1000, d["active_power"] / 1000], 1).astype("float32")
    y = d[target].values.astype("float32")
    X, Y = [], []
    for i in range(lookback, len(d) - horizon + 1):
        X.append(feats[i - lookback:i]); Y.append(y[i + horizon - 1])
    return np.array(X), np.array(Y)

def split_scale(X, Y, val=0.15, test=0.15):
    n = len(X); a = int(n * (1 - val - test)); b = int(n * (1 - test))
    mu = X[:a].reshape(-1, X.shape[2]).mean(0); sd = X[:a].reshape(-1, X.shape[2]).std(0) + 1e-6
    ym, ys = Y[:a].mean(), Y[:a].std() + 1e-6
    f = lambda x: (x - mu) / sd; g = lambda y: (y - ym) / ys
    return (f(X[:a]), g(Y[:a])), (f(X[a:b]), g(Y[a:b])), (f(X[b:]), Y[b:]), (ym, ys)
