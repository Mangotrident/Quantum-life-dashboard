import argparse, os, json
from joblib import load
import pandas as pd
from src.qlife.data import load_phase2_csv, simulate_phase2_grid
from src.qlife.plotting import regression_report
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, default="data/phase2_benchmark_results.csv")
    ap.add_argument("--model", required=False, default="models/model.pkl")
    ap.add_argument("--out", required=False, default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out,"figs"), exist_ok=True)

    try:
        df = load_phase2_csv(args.data)
    except FileNotFoundError:
        df = simulate_phase2_grid()

    pipe = load(args.model)
    y_pred = pipe.predict(df[["topology","J","gamma","sigma"]])
    m = regression_report(df["QLS"], y_pred)

    # thresholds from params.yaml
    with open("params.yaml") as f:
        p = yaml.safe_load(f)
    r2_min = float(p["benchmark"]["r2_min"])

    with open(os.path.join(args.out,"metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    print("Benchmark metrics:", m)
    if m["r2"] < r2_min:
        raise SystemExit(f"R² below threshold ({m['r2']:.3f} < {r2_min:.3f})")

if __name__ == "__main__":
    main()
