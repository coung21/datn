import numpy as np
from scipy import stats

from scripts.seed_stats import hier_bootstrap, welch_ci


def test_welch_ci_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(69.3, 0.3, 5), rng.normal(68.7, 0.4, 5)
    d, lo, hi, p = welch_ci(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert np.isclose(d, b.mean() - a.mean()) and np.isclose(p, ref.pvalue)
    assert lo < d < hi


def test_hier_bootstrap_covers_true_diff():
    rng = np.random.default_rng(1)
    patient = rng.normal(70, 14, 72)  # độ khó từng bệnh nhân, chung cho 2 mô hình
    A = patient + rng.normal(0, 0.3, (5, 1)) + rng.normal(0, 2, (5, 72))
    B = patient - 0.6 + rng.normal(0, 0.3, (5, 1)) + rng.normal(0, 2, (5, 72))
    d, lo, hi = hier_bootstrap(A, B, n_boot=2000)
    assert lo < -0.6 < hi and hi < 0  # chênh lệch thật nằm trong CI, CI không chứa 0
    # cùng 1 mô hình: CI chứa 0
    _, lo0, hi0 = hier_bootstrap(A, A[::-1], n_boot=2000)
    assert lo0 < 0 < hi0
