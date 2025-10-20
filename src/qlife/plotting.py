import os, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

def plot_calibration(y_true, y_pred, outpath):
    import numpy as np, pandas as pd
    bins = np.linspace(y_pred.min(), y_pred.max(), 12)
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    d["bin"] = np.digitize(d["y_pred"], bins)
    g = d.groupby("bin").agg(y_true_mean=("y_true","mean"), y_pred_mean=("y_pred","mean")).dropna()
    plt.figure(figsize=(5,5))
    plt.scatter(g["y_pred_mean"], g["y_true_mean"], s=40)
    lim = [min(g.min()), max(g.max())]
    plt.plot(lim, lim, "--", color="gray")
    plt.xlabel("Predicted (bin mean)"); plt.ylabel("Observed (bin mean)")
    plt.title("Calibration (binned)"); plt.tight_layout()
    plt.savefig(outpath, dpi=180); plt.close()

def regression_report(y_true, y_pred):
    return dict(r2=float(r2_score(y_true, y_pred)),
                mae=float(mean_absolute_error(y_true, y_pred)))
