import numpy as np
from src.qlife.metrics import qls_from_series

def test_qls_monotone_entropy_penalty():
    t = np.linspace(0,1,200)
    coh = np.ones_like(t)*0.8
    pur = np.ones_like(t)*0.9
    ent_good = np.ones_like(t)*0.4
    ent_bad  = np.ones_like(t)*0.9
    q_good = qls_from_series(t, coh, pur, ent_good)
    q_bad  = qls_from_series(t, coh, pur, ent_bad)
    assert q_good > q_bad
