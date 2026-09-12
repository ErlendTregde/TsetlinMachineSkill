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

The `tmu` README itself has no worked code examples. The API below is taken from a real script in
a CAIR-maintained companion repo for the TMComposites paper [arXiv:2309.04801], not from memory
[TMU-PPExample]:

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

Import path confirmed elsewhere in the wild too: a hardware-accelerator paper cites TMU's coalesced
classifier module directly at
`tmu/models/classification/coalesced_classifier.py` [IEEE-TM-accelerator-2025], confirming the
`tmu.models.classification.*` module layout is real and stable enough to be cited externally.

## Project health signals worth knowing

At the time this skill was written the repo showed 25 open issues [TMU-README page], and the README
is largely install instructions rather than API documentation. Budget for reading source.
