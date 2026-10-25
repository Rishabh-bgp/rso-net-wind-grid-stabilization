import numpy as np
def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    mape = np.mean(np.abs(e) / np.maximum(np.abs(y), 0.5)) * 100  # floor avoids blow-up near calm winds
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAPE_%": float(mape), "R2": float(1 - np.sum(e**2) / np.sum((y - y.mean())**2))}
