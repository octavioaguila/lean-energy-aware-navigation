import numpy as np
from scipy import stats

# Shared success-only metrics for paired tests (canonical_name, label); SR uses McNemar.
PAIRED_METRICS = [
    ("energy", "Energy (J)"),
    ("energy_per_m", "J/m"),
    ("spl", "SPL"),
    ("steps", "Steps"),
    ("alpha_rms", "alpha_rms (rad/s^2)"),
]


def wilcoxon_paired(a_vals, b_vals, min_n=6):
    """Paired Wilcoxon signed-rank on per-episode (a - b). Returns dict or None.

    median_delta is median(a - b). The paired test removes the scenario-to-scenario
    variance that inflates the marginal CIs, so it tests the per-episode difference
    directly rather than relying on CI overlap.
    """
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    n = len(a)
    if n == 0:
        return None
    diff = a - b
    median_delta = float(np.median(diff))
    if n < min_n or np.all(diff == 0):
        return {"n": n, "median_delta": median_delta, "p_value": None}
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:
        return {"n": n, "median_delta": median_delta, "p_value": None}
    return {"n": n, "median_delta": median_delta, "statistic": float(stat), "p_value": float(p)}


def mcnemar_paired(a_succ, b_succ):
    """Exact (binomial) McNemar test on paired success/failure outcomes."""
    n10 = int(sum(1 for x, y in zip(a_succ, b_succ) if x and not y))  # a succeeds, b fails
    n01 = int(sum(1 for x, y in zip(a_succ, b_succ) if y and not x))  # b succeeds, a fails
    disc = n10 + n01
    p = 1.0 if disc == 0 else float(stats.binomtest(min(n10, n01), disc, 0.5).pvalue)
    return {"n": len(a_succ), "n_a_only": n10, "n_b_only": n01, "p_value": p}


def fmt_p(p):
    if p is None:
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def wilson_ci(k, n, z=1.959964):
    """Wilson 95% interval for a binomial proportion k/n. Returns (p, lo, hi)."""
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def paired_ratio(a_vals, b_vals):
    """100*(mean(b)-mean(a))/mean(a) on paired values (the paper's Table IV dE%)."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    return float(100.0 * (b.mean() - a.mean()) / a.mean())


def paired_ratio_bootstrap(a_vals, b_vals, n_boot=2000, seed=0, clusters=None):
    """Percentile 95% CI of paired_ratio by resampling pairs (or whole clusters)."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx_sets = (rng.integers(0, len(a), len(a)) for _ in range(n_boot))
    else:
        cl = np.asarray(clusters)
        groups = [np.flatnonzero(cl == g) for g in np.unique(cl)]
        idx_sets = (np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                    for _ in range(n_boot))
    vals = np.array([100.0 * (b[i].mean() - a[i].mean()) / a[i].mean() for i in idx_sets])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
