import argparse, os, json
from joblib import dump
from src.qlife.data import load_phase2_csv, simulate_phase2_grid
from src.qlife.model import build_pipeline, split_fit
from src.qlife.plotting import plot_calibration, regression_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, default="data/phase2_benchmark_results.csv")
    ap.add_argument("--out", required=False, default="models/model.pkl")
    ap.add_argument("--metrics", required=False, default="artifacts/metrics.json")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    os.makedirs("artifacts/figs", exist_ok=True)

    try:
        df = load_phase2_csv(args.data)
        print(f"Loaded real CSV: {df.shape}")
    except FileNotFoundError:
        df = simulate_phase2_grid()
        print(f"Simulated dataset: {df.shape}")

    pipe = build_pipeline()
    pipe, (X_te, y_te) = split_fit(pipe, df, test_size=args.test_size, seed=42)
    y_pr = pipe.predict(X_te)

    dump(pipe, args.out)
    print(f"Saved model → {args.out}")

    # metrics + plot
    m = regression_report(y_te, y_pr)
    with open(args.metrics, "w") as f:
        json.dump(m, f, indent=2)
    plot_calibration(y_te, y_pr, "artifacts/figs/calibration.png")
    print("Metrics:", m)

if __name__ == "__main__":
    main()
