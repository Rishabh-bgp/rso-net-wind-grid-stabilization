import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import synthetic_scada, make_windows, load_scada_csv
from src.metrics import regression_metrics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def test_metrics_perfect():
    m = regression_metrics([1, 2, 3], [1, 2, 3]); assert m["MAE"] == 0 and m["R2"] == 1
def test_synthetic_and_windows():
    X, Y = make_windows(synthetic_scada(300), lookback=12); assert X.shape == (288, 12, 5) and len(Y) == 288
def test_bundled_csv_loads():
    df = load_scada_csv(os.path.join(ROOT, "data/synthetic/synthetic_scada.csv")); assert len(df) > 100 and df["wind_speed"].between(0, 40).all()
