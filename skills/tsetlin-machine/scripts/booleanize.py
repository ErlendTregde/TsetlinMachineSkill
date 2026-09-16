"""
Thermometer-booleanize a continuous/tabular matrix for a Tsetlin Machine, and say what it did.

Bundled because every eval run of this skill wrote its own copy of this, badly or otherwise.

Why thermometer rather than sklearn's KBinsDiscretizer one-hot: thermometer bits preserve ordering
("age >= 55" stays true once "age >= 47" is true), so a clause that includes two bits of the same
feature reads back as an interval a clinician understands -- "55 <= age < 70". One-hot buckets lose
that, and the TM has to spend clauses rediscovering the ordering.

`cuts_for_column` deduplicates cuts and gives low-cardinality columns one cut per distinct value.
Be clear about why: a quantile list like np.percentile(n_prior_admissions, [20,40,60,80]) ->
[0, 1, 1, 2] wastes a bit on the duplicate. It does NOT make '>= 3' inexpressible -- bits are
`x > cut`, so the cut at 2 already gives exactly that. Measured, the accuracy difference is inside
split-to-split noise. This is hygiene, not a rescue. See ../references/booleanization.md.

Fit cuts on TRAIN ONLY, then transform everything, or you leak the test set into your thresholds.

    from booleanize import fit_cuts, transform, describe
    cuts = fit_cuts(X_train, names)
    Btr, Bte = transform(X_train, cuts), transform(X_test, cuts)
    print(describe(cuts, names, Btr))

Run `python booleanize.py` to self-check cut generation.
"""

import numpy as np

MAX_DISTINCT_FOR_EXACT = 12  # at or below this, cut on every distinct value instead of quantiles
DEFAULT_BITS = 4


def cuts_for_column(col, n_bits=DEFAULT_BITS, max_distinct=MAX_DISTINCT_FOR_EXACT):
    """Thresholds for one column. Returns a sorted, duplicate-free 1-D array.

    Low-cardinality columns get one cut per distinct value (above the minimum, since a cut at the
    minimum would be true for everything). Continuous columns get evenly spaced quantiles, passed
    through np.unique because percentiles of a skewed column can still collide.
    """
    v = np.unique(col[~np.isnan(col)]) if col.dtype.kind == "f" else np.unique(col)
    if v.size == 0:
        return np.array([])
    if v.size <= max_distinct:
        return v[1:].astype(float)
    qs = np.linspace(0, 100, n_bits + 2)[1:-1]
    return np.unique(np.percentile(col, qs)).astype(float)


def fit_cuts(X, names=None, n_bits=DEFAULT_BITS, max_distinct=MAX_DISTINCT_FOR_EXACT):
    """Fit per-column cuts on the TRAINING matrix only. Returns a list of arrays, one per column."""
    X = np.asarray(X, dtype=float)
    return [cuts_for_column(X[:, j], n_bits, max_distinct) for j in range(X.shape[1])]


def transform(X, cuts):
    """Apply fitted cuts. Returns uint32, which is what TMU requires and pyTsetlinMachine accepts."""
    X = np.asarray(X, dtype=float)
    blocks = [(X[:, j, None] > c[None, :]) for j, c in enumerate(cuts) if c.size]
    if not blocks:
        raise ValueError("every column produced zero cuts -- is X constant?")
    return np.concatenate(blocks, axis=1).astype(np.uint32)


def bit_labels(cuts, names=None):
    """(feature_name, threshold) for each Boolean column, so clauses can be printed in real terms."""
    names = names if names is not None else [f"x{j}" for j in range(len(cuts))]
    return [(names[j], float(t)) for j, c in enumerate(cuts) for t in c]


def describe(cuts, names=None, B=None):
    """Human-readable report. Print this -- the failure modes are obvious on sight and silent otherwise."""
    names = names if names is not None else [f"x{j}" for j in range(len(cuts))]
    n_bits = sum(c.size for c in cuts)
    lines = [f"{len(cuts)} columns -> {n_bits} Boolean features ({n_bits * 2} literals per clause)"]
    if B is not None:
        density = float(B.mean())
        flag = "" if 0.02 < density < 0.98 else "   <-- SUSPICIOUS: almost no signal to include"
        lines.append(f"bit density: {density:.1%} ones{flag}")
        constant = int((B.min(axis=0) == B.max(axis=0)).sum())
        if constant:
            lines.append(f"{constant} constant bit(s) -- these can never appear in a useful clause")
    exact = [names[j] for j, c in enumerate(cuts)
             if c.size and c.size <= MAX_DISTINCT_FOR_EXACT and np.allclose(c, np.round(c))]
    if exact:
        lines.append(f"exact-cut (low-cardinality) columns: {', '.join(exact)}")
    for j, c in enumerate(cuts):
        if c.size == 0:
            lines.append(f"  {names[j]}: NO CUTS -- constant column, drop it")
    return "\n".join(lines)


def _self_check():
    """Checks cut generation: no duplicates, low-cardinality columns keep every threshold."""
    rng = np.random.default_rng(0)
    prior = rng.poisson(1.1, 500).clip(0, 12).astype(float)  # small-integer column, 0-5 in practice

    naive = np.percentile(prior, np.linspace(0, 100, DEFAULT_BITS + 2)[1:-1])
    assert len(np.unique(naive)) < len(naive), "expected naive quantiles to collide, got " + str(naive)
    # Note: the duplicate wastes a bit. It does NOT lose the '>= 3' threshold -- bits are x > cut,
    # so the cut at 2 already expresses it. Asserted below so nobody re-adds that claim.
    naive_bits = prior[:, None] > np.unique(naive)[None, :]
    assert any(np.array_equal(naive_bits[:, j], prior >= 3) for j in range(naive_bits.shape[1])),         "naive cuts should still express '>= 3' via the cut at 2"

    fixed = cuts_for_column(prior)
    assert len(np.unique(fixed)) == len(fixed), "fixed cuts must not contain duplicates"
    assert 3.0 in set(fixed), f"exact cuts should include every distinct value, got {fixed}"

    age = rng.normal(62, 15, 500)
    assert len(cuts_for_column(age)) == DEFAULT_BITS, "continuous columns should still get n_bits cuts"

    X = np.column_stack([prior, age])
    cuts = fit_cuts(X, ["n_prior_admissions", "age"])
    B = transform(X, cuts)
    assert B.dtype == np.uint32 and B.shape[0] == 500
    assert B.shape[1] == sum(c.size for c in cuts)
    assert len(bit_labels(cuts, ["n_prior_admissions", "age"])) == B.shape[1]

    print("self-check passed")
    print(f"  naive cuts: {naive}  <- duplicate bit wasted, but '> 2' still expresses '>= 3'")
    print(f"  exact cuts: {fixed}")
    print(describe(cuts, ["n_prior_admissions", "age"], B))


if __name__ == "__main__":
    _self_check()
