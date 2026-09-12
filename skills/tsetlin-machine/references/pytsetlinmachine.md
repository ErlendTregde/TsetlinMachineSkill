# pyTsetlinMachine

Source of record: `cair/pyTsetlinMachine` README [PyTM-README]. A high-level Python API with fast
C extensions [TM-README §Other Implementations].

Note for expectation-setting: the README says documentation is "coming soon"
[PyTM-README §Documentation]. The README's worked examples are effectively the documentation, so
prefer copying their shape over improvising.

## Install and requirements

```
pip install pyTsetlinMachine
```

Requirements listed: Python 3.7.x, Numpy, and Ubuntu or macOS [PyTM-README §Requirements].
A multi-threaded variant lives in `cair/pyTsetlinMachineParallel`
[PyTM-README §Multi-threading].

## Constructor shape

Positional: **clauses, T, s**, then keyword options.

```python
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)
tm.fit(X_train, Y_train, epochs=200)
tm.predict(X_test)
```

[PyTM-README §Multiclass Demo]

2D convolution takes an extra patch-size tuple:

```python
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D

ctm = MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)
```

with X reshaped to (n, rows, cols) [PyTM-README §2D Convolution Demo].

Regression:

```python
from pyTsetlinMachine.tm import RegressionTsetlinMachine

tm = RegressionTsetlinMachine(1000, 500*10, 2.75, weighted_clauses=True)
```

[PyTM-README §Regression Demo]

## Epoch loops

For per-epoch reporting, call `fit` with `epochs=1` and `incremental=True` inside a loop
[PyTM-README §MNIST Demo]. Without `incremental=True` you are not continuing training.

## Published configurations (use as starting points, say so)

All from [PyTM-README], with the accuracies the README reports:

| Task | Configuration | Reported result |
|---|---|---|
| Noisy XOR | `MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)`, 200 epochs | 100% |
| Interpretability demo (noisy XOR, 20 features) | `MultiClassTsetlinMachine(10, 15, 3.0, boost_true_positive_feedback=0)`, 200 epochs | 100% |
| 2D noisy XOR | `MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2,2), boost_true_positive_feedback=0)`, 5000 epochs | 99.97% |
| Breast cancer (continuous, binarized at 10 bits/feature) | `MultiClassTsetlinMachine(800, 40, 5.0)`, 25 epochs | ~97.5% mean over 100 runs |
| MNIST | `MultiClassTsetlinMachine(2000, 50, 10.0)` | 98.07% at 250 epochs |
| MNIST, weighted | `MultiClassTsetlinMachine(2000, 50*100, 10.0, weighted_clauses=True)` | 98.19% at 60 epochs |
| MNIST, 2D conv, weighted | `MultiClassConvolutionalTsetlinMachine2D(2000, 50*100, 5.0, (10,10), weighted_clauses=True)` | 99.16% at 30 epochs |
| Fashion-MNIST, 2D conv, weighted | same shape, adaptive-threshold preprocessing | ~90% at 30 epochs |
| IMDb (5000 selected n-gram features) | `MultiClassTsetlinMachine(10000, 8000, 5.0, weighted_clauses=True, clause_drop_p=0.75)` | ~89.8% at 40 epochs |
| California housing regression | `RegressionTsetlinMachine(1000, 500*10, 2.75, weighted_clauses=True)`, 30 epochs | RMSD 0.61 |

Two patterns worth noticing and passing on: `T` scales up by roughly 100x when weighted clauses are
switched on in the MNIST examples, and `s` sits between 2.75 and 10.0 across every published
example here [PyTM-README]. The CIFAR-10 Optuna study searched `s` over [1, 10] and `T` over
[1, 3000] [arXiv:2406.00704], which is a reasonable search range to offer.

## Reading the clauses

This is the part a model will get wrong without the recipe. From the interpretability demo
[PyTM-README §Interpretability Demo]:

- `tm.ta_action(class_index, clause_index, literal_index)` returns 1 when that literal is included.
- Literal indexing: `k < number_of_features` means the plain feature `x_k`; `k >= number_of_features`
  means the negated feature `¬x_{k - number_of_features}`.
- Clause polarity: the demo prints even clause indices (0, 2, 4, …) as the positive clauses for a
  class and odd indices (1, 3, 5, …) as the negative clauses.

The demo's output on noisy XOR is worth showing a user, because it makes the interpretability claim
concrete — class 0's positive clauses come out as things like `¬x0 ∧ ¬x1` and `x0 ∧ x1`, and its
negative clauses as `x0 ∧ ¬x1` and `x1 ∧ ¬x0` [PyTM-README §Interpretability Demo].

`scripts/show_clauses.py` in this skill wraps that recipe with feature names.

## Binarizer

`pyTsetlinMachine.tools.Binarizer(max_bits_per_feature=10)` with `fit` / `transform` handles
continuous input [PyTM-README §Continuous Input Demo]. See `booleanization.md`.

## Features listed

Tsetlin Machine, Convolutional TM, Regression TM, Weighted TM and Embedding TM, with support for
continuous features, multigranular clauses, clause indexing, drop clause and literal budget
[PyTM-README §intro].
