import os, numpy as np, pandas as pd

def load_phase2_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df["topology"] = df["topology"].astype(str)
    for c in ["J","gamma","sigma","QLS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["topology","J","gamma","sigma","QLS"]).reset_index(drop=True)

def simulate_phase2_grid(seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topologies = ["line","ring","smallworld"]
    J = [0.05, 0.1, 0.2, 0.4]
    G = [0.005, 0.01, 0.02, 0.05]
    S = [0.0, 0.05]
    rows=[]
    for t in topologies:
        for j in J:
            for g in G:
                for s in S:
                    base = 2.3 + 0.9*np.exp(-g*12) + 0.4*np.tanh(j*3) - 0.2*s
                    if t=="smallworld": base -= 0.5
                    rows.append({"topology":t,"J":j,"gamma":g,"sigma":s,"QLS":base})
    return pd.DataFrame(rows)
