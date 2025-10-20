import numpy as np

def qls_from_series(time, coherence=None, purity=None, entropy=None,
                    ideal_entropy=0.4, tail_frac=0.5):
    n = len(time)
    start = int((1 - tail_frac) * n)
    if start >= n:
        start = max(0, n - 1)

    def normC(x):
        if x is None: return 0.5
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        return float(np.nanmean((x - xmin) / (xmax - xmin + 1e-9)))

    def normP(x):
        if x is None: return 0.7
        return float(np.clip(np.nanmean(x), 0, 1))

    def penaltyS(x):
        if x is None: return 1.0 / (1.0 + abs(0.5 - ideal_entropy))
        xm = float(np.nanmean(x))
        return float(1.0 / (1.0 + abs(xm - ideal_entropy)))

    Cn = normC(None if coherence is None else coherence[start:])
    Pn = normP(None if purity is None else purity[start:])
    penS = penaltyS(None if entropy is None else entropy[start:])
    return float(Cn * Pn * penS)
