"""Generate an offline synthetic SCADA-style CSV: python scripts/make_synthetic.py --n 3000"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import synthetic_scada
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--out", default="data/synthetic/synthetic_scada.csv")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
synthetic_scada(a.n).to_csv(a.out, index=False); print("wrote", a.out)
