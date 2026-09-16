# TMU — Tsetlin Machine Unified

Source of record: `cair/tmu` README [TMU-README]. Billed as one codebase covering the family, and
described as a central resource for enthusiasts and researchers [TMU-README].

Pick TMU when the user needs a variant or extension that pyTsetlinMachine does not carry.

## Install

```
# stable
pip install git+https://github.com/cair/tmu.git

# development branch
pip install git+https://github.com/cair/tmu.git@dev
```

[TMU-README §Installation]

**Install from git, not PyPI — this matters more than it looks.** `pip install tmu` gives you
0.8.3, published 2024-03-25, which predates numpy 2.0 (June 2024). That release calls
`np.uint32(~0)` while building the clause bank, which numpy 2.x refuses:

```
OverflowError: Python integer -1 out of bounds for uint32
  .../tmu/clause_bank/clause_bank.py:136 in initialize_clauses
```

`main` fixed it — the source now reads `np.array(~0).astype(np.uint32)` with a comment naming
numpy>=2.0 — but **there has been no PyPI release since**, so the fix only reaches you through git.
Verified 2026-09-14: a git install runs fine under numpy 2.4.6. If you are stuck on the PyPI
package for some reason, pin `numpy<2` instead. [verified-2026-09-14]

**Do not pass `seed=0` — it hangs `fit()` forever.** Verified 2026-09-14 on both Windows and
Debian 13, with the git build: `TMClassifier(..., seed=0)` spins indefinitely inside
`cb_type_i_feedback` (`clause_bank.py:248`, reached via `fit -> _fit_sample -> mechanism_feedback`),
on datasets as small as 200x8 with 10 clauses. Seeds 1-7, 42, and omitting `seed` entirely all
train normally in ~2s to 100% on noisy XOR. It looks like a falsy-zero check treating `0` as
"unset". Use `seed=42` or leave it out; if a TMU run hangs with no CPU progress, check the seed
before suspecting your data. [verified-2026-09-14]

This is not documented anywhere upstream and has not been reported as an issue, so expect no
warning from the README, the demos, or a search — you will only find it by hitting it.

This is also the single best argument for the smoke test in SKILL.md Step 2: the failure is a
silent hang, not an exception, and it is indistinguishable from "the model is just slow" unless you
have a known-good baseline run to compare against.

**Prerequisites that trip people up** [TMU-README §Installation]:

- Windows: MSVC build tools, installing the `Workloads → Desktop development with C++` package.
  The README flags this as roughly 6–7 GB.
- Ubuntu: `sudo apt install libffi-dev`

If a user reports a build failure on Windows or a missing-ffi error on Ubuntu, check these first
rather than debugging their Python. [heuristic]

For development work: clone the `dev` branch, then `pip install -e .`, or `pip install -e .[composite]`
for TMU-Composite. The README also suggests creating a branch and working inside the `examples`
folder for new projects [TMU-README §Development].

There is a devcontainers tutorial in `docs/tutorials/devcontainers/devcontainers.md`
[TMU-README §Guides and Tutorials].

## What TMU implements

Core implementations [TMU-README §Features]:

- Tsetlin Machine [arXiv:1804.01508]
- Coalesced Tsetlin Machine [arXiv:2108.07594]
- Convolutional Tsetlin Machine [arXiv:1905.09688]
- Regression Tsetlin Machine [RSTA-2019-0165]
- Weighted Tsetlin Machine [IEEE-9316190]
- Autoencoder [arXiv:2301.00709]
- Multi-task classifier — *upcoming*
- One-vs-one multi-class classifier — *upcoming*
- Relational Tsetlin Machine [Springer-s10844-021-00682-5] — *in progress*

Extended features [TMU-README §Features]:

- Continuous features [arXiv:1905.04199]
- Drop clause [arXiv:2105.14506]
- Literal budget [arXiv:2301.08190]
- Focused negative sampling [IEEE-9923859]
- Type III feedback [arXiv:2309.06315]
- Incremental clause evaluation — *upcoming*
- Sparse computation with absorbing actions [arXiv:2310.11481]
- TMComposites [arXiv:2309.04801] — *in progress*

It is written in Python with wrappers for C and CUDA-based clause evaluation and updating
[TMU-README §Features].

**Read the upcoming/in-progress markers literally.** If a user asks for a multi-task classifier or
TMComposites, check the current repo state before writing code — the README marks these as not
finished.

## API

The README has no API documentation, but the repo ships a dozen runnable demos in
`examples/classification/` — `MNISTDemo.py`, `MNISTConvolutionDemo.py`, `MNISTDemoCoalesced.py`,
`XORDemo.py`, `IMDbTextCategorizationDemo.py`, `InterpretabilityDemo.py` and more
[TMU-Examples]. **Those demos are the documentation.** Copy their shape rather than improvising.

The canonical training loop, from `examples/classification/MNISTDemo.py` [TMU-MNISTDemo]:

```python
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

tm = TMClassifier(
    number_of_clauses=2000,
    T=5000,
    s=10.0,
    max_included_literals=32,
    platform="CPU",          # "CPU" | "CPU_sparse" | "CUDA"
    weighted_clauses=True,
    seed=42,
)

for epoch in range(60):
    tm.fit(x_train.astype(np.uint32), y_train.astype(np.uint32))
    acc = 100 * (tm.predict(x_test) == y_test).mean()
```

Two things that bite: **X and Y must be `uint32`**, and `fit` has no `epochs` argument — you write
the loop. TMU also ships datasets, e.g. `from tmu.data import MNIST; data = MNIST().get()`
[TMU-MNISTDemo].

A second worked configuration, from a CAIR companion repo for the TMComposites paper
[arXiv:2309.04801], showing the GPU/convolution case [TMU-PPExample]:

```python
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time

tm = TMClassifier(
    number_of_clauses=2000,
    T=1500,
    s=2.5,
    max_included_literals=32,
    platform="GPU",
    weighted_clauses=True,
    patch_dim=(3, 3),   # convolution/image case only, see below
)

for epoch in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train)
    stop_training = time()

    Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
    result_test = 100 * (Y_test_scores.argmax(axis=1) == Y_test).mean()
```

[TMU-PPExample]

Confirmed, real differences from pyTsetlinMachine's `MultiClassTsetlinMachine(clauses, T, s)`
[PyTM-README §Multiclass Demo]:

- Keyword arguments (`number_of_clauses`, `T`, `s`, ...), not bare positionals.
- A `platform` kwarg (`"GPU"` in the example) rather than a separate CUDA-specific class.
- `fit(X, Y)` takes **no `epochs` argument** — loop over `fit` calls yourself for per-epoch
  reporting, same idea as pyTsetlinMachine's `incremental=True` pattern but via a plain Python loop
  instead of a flag [TMU-PPExample; contrast PyTM-README §MNIST Demo].
- `predict(X, return_class_sums=True)` returns **two** values: hard predictions and per-class vote
  sums, and the example computes accuracy from `argmax` over the score matrix rather than from the
  hard predictions directly [TMU-PPExample].
- `max_included_literals` and `weighted_clauses` are both explicit constructor kwargs, matching the
  literal budget and weighted-clause features listed in the README [TMU-README §Features].
- `patch_dim` is specific to the convolutional/image case in this example. Do not assume it exists
  or is required for plain tabular classification — that was not verified, since only an image
  script was found. [heuristic]

## Reading the clauses — TMU's convention is a THIRD one

This is the highest-risk part of using TMU, because the call *looks* like pyTsetlinMachine's and
means something different. Compare all three:

```
pyTsetlinMachine:  tm.ta_action(class, clause, literal)
TMU:               tm.get_ta_action(clause, ta, the_class=..., polarity=...)   <- clause FIRST
GraphTsetlinMachine: tm.ta_action(layer, clause, literal)                      <- depth, not class
```

Swap the first two arguments between pyTM and TMU and nothing raises — you get a clause listing
that is quietly wrong. Check which library you are in before writing the extraction loop.

The other difference: pyTsetlinMachine encodes clause polarity in the **even/odd clause index**,
while TMU takes an explicit `polarity` argument (`0` = positive/votes-for, `1` = negative/votes-
against) and you iterate `range(number_of_clauses // 2)` for each polarity.

Verbatim shape from `examples/classification/InterpretabilityDemo.py` [TMU-InterpretabilityDemo]:

```python
precision = tm.clause_precision(the_class, polarity, X_test, Y_test)
recall    = tm.clause_recall(the_class, polarity, X_test, Y_test)

for j in range(number_of_clauses // 2):
    print("Clause #%d W:%d P:%.2f R:%.2f" % (
        j, tm.get_weight(the_class, polarity, j), precision[j], recall[j]), end=' ')
    literals = []
    for k in range(number_of_features * 2):
        if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity):
            if k < number_of_features:
                literals.append("x%d" % k)
            else:
                literals.append("¬x%d" % (k - number_of_features))
    print(" ∧ ".join(literals))
```

Literal indexing matches pyTsetlinMachine: `k < n_features` is `x_k`, `k >= n_features` is
`¬x_{k-n_features}`. That part does transfer.

### Why this is worth more than pyTsetlinMachine's clause dump

TMU gives you per-clause quality metrics, which turn a wall of conjunctions into something you can
rank and filter — the difference between "here are 2000 rules" and "here are the five that matter":

| Call | What it gives you |
|---|---|
| `clause_precision(the_class, polarity, X, Y)` | per-clause precision — how often this rule is right when it fires |
| `clause_recall(the_class, polarity, X, Y)` | per-clause recall — how much of the class it covers |
| `get_weight(the_class, polarity, clause)` | learned clause weight (with `weighted_clauses=True`) |
| `get_ta_state(clause, ta, the_class=, polarity=)` | raw automaton state — how firmly a literal is included |
| `literal_clause_frequency()` | how often each literal appears across clauses |
| `clause_co_occurrence(X, percentage=True)` | which clauses fire together |

[TMU-InterpretabilityDemo]

**Sort clauses by precision and show the user the top few with their support**, rather than dumping
every clause. A clinician or domain expert wants the five rules that carry the decision, not 2000
conjunctions — and precision/recall is exactly what makes that selection defensible instead of
arbitrary. [heuristic]

Import path confirmed elsewhere in the wild too: a hardware-accelerator paper cites TMU's coalesced
classifier module directly at
`tmu/models/classification/coalesced_classifier.py` [IEEE-TM-accelerator-2025], confirming the
`tmu.models.classification.*` module layout is real and stable enough to be cited externally.

## Project health signals worth knowing

At the time this skill was written the repo showed 25 open issues [TMU-README page], and the README
is largely install instructions rather than API documentation. Budget for reading source.
