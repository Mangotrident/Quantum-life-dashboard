import numpy as np

def qls_from_series(t, coherence, purity, entropy, alpha=1.0, beta=0.5, gamma=0.2):
    """
    Compute Quantum Life Score (QLS) from time-series features.
    Ensures monotonic penalty for entropy and reward for coherence/purity.
    """
    t = np.asarray(t)
    coherence = np.asarray(coherence)
    purity = np.asarray(purity)
    entropy = np.asarray(entropy)

    # Normalize all to [0, 1]
    coherence = (coherence - coherence.min()) / (coherence.ptp() + 1e-9)
    purity = (purity - purity.min()) / (purity.ptp() + 1e-9)
    entropy = (entropy - entropy.min()) / (entropy.ptp() + 1e-9)

    # Compute weighted instantaneous score
    inst_score = alpha * coherence + beta * purity - gamma * entropy

    # Integrate over time to get total QLS
    qls = np.trapz(inst_score, t)

    return float(np.clip(qls, 0, None))
