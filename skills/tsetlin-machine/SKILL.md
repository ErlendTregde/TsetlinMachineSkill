---
name: tsetlin-machine
description: "Write, debug and explain Tsetlin Machine and Graph Tsetlin Machine code (pyTsetlinMachine, TMU, GraphTsetlinMachine). Use this skill whenever the user mentions Tsetlin machines, Tsetlin automata, TMU, GraphTM, clauses/literals as a learning model, interpretable or logic-based ML, CAIR/Granmo libraries, or asks for an explainable alternative to a neural network — even if they do not name a specific library. Also use it when the user is booleanizing data for a TM, tuning T and s, building graphs for message passing, or trying to read learned clauses out of a trained model."
version: 0.1.0
license: MIT
---

# Tsetlin Machine

Helps a newcomer get a correct, interpretable Tsetlin Machine (TM) running, and helps the
assistant reason about TMs correctly instead of pattern-matching from neural-network habits.

## Citation discipline (read this first)

Every factual claim in this skill carries a source tag such as `[TM-README]` or
`[arXiv:1909.07310]`. The tags resolve in `references/sources.md`.

Claims that are **not** sourced are tagged `[heuristic]`. Those are starting points to test,
not established results. When you pass a `[heuristic]` claim on to the user, say so.

Do not add unsourced claims to this skill. If you are unsure whether something is true of a
library, read that library's README or source rather than guessing — the TM ecosystem has many
implementations with different APIs [TM-README §Other Implementations], so recall is unreliable.

## Step 1 — Route to a library before writing any code

There is no single TM package. `cair/TsetlinMachine` itself lists more than fifteen separate
implementations across Python, C, C++, CUDA, Julia, Rust, C#, F# and Node.js
[TM-README §Other Implementations], plus separate repos for the Convolutional, Graph, Regression,
Coalesced and massively-parallel architectures [TM-README §Other Architectures].

**Use a library.** That is how the field works: the published architectures — Coalesced,
Convolutional, Regression, autoencoder, drop clause, Type III feedback, literal budget — are years
of papers with C and CUDA kernels behind them, and reimplementing them is neither expected nor wise.
Default to the library and copy the shape of its demo scripts.

| User's situation | Library | Read |
|---|---|---|
| **Default for new work** — tabular/Boolean, binary or multiclass, CPU or CUDA | **TMU** | `references/tmu.md` |
| Coalesced / autoencoder / drop clause / Type III feedback / literal budget / sparse absorbing | TMU | `references/tmu.md` |
| Images with spatial structure | Convolutional TM (`patch_dim` in TMU) | `references/tmu.md` |
| Graphs, sequences, variable-sized input, multimodal, relations between parts | GraphTsetlinMachine | `references/graphtm.md` |
| Maintaining existing pyTsetlinMachine code, or following a paper that used it | pyTsetlinMachine | `references/pytsetlinmachine.md` |
| Continuous features anywhere in the above | still need booleanization | `references/booleanization.md` |
| Asking *how a TM works* rather than for code | — | `references/concepts.md` |

**TMU is the default, not pyTsetlinMachine.** TMU is the "unified" successor — one codebase for the
whole family, actively developed, with CUDA support and far better clause introspection
(`clause_precision`, `clause_recall`, `get_weight`) than pyTsetlinMachine exposes
[TMU-README §Features, TMU-InterpretabilityDemo]. pyTsetlinMachine is older, its last PyPI release
was 2024-03-05, and it ships **source-only with POSIX-specific C** — it declares Ubuntu or macOS as
its requirements and genuinely cannot build on Windows (`fatal error C1083: Cannot open include
file: 'sys/time.h'`) [PyTM-README §Requirements, verified-2026-09-14]. Prefer it only when the user
already has pyTM code or is reproducing a paper that used it.

### Platform reality — check this before promising anything

Install `tmu` **from git, never from PyPI**: `pip install git+https://github.com/cair/tmu.git`.
The PyPI release (0.8.3, 2024-03-25) predates numpy 2.0 and dies with
`OverflowError: Python integer -1 out of bounds for uint32` while building the clause bank. `main`
fixed it but has never been released to PyPI. On Ubuntu, `sudo apt install libffi-dev` first; on
Windows, MSVC build tools [TMU-README §Installation, verified-2026-09-14].

Verified 2026-09-14, git install of TMU, trains noisy XOR to 100%:

| | Windows 11 + MSVC | Debian 13 |
|---|---|---|
| TMU from git | works (numpy 2.2.6) | works (numpy 2.4.6) |
| TMU from PyPI | `OverflowError` on numpy>=2 | same |
| pyTsetlinMachine | **cannot build** (`sys/time.h`) | works (per README) |
| green-tsetlin | **cannot build** (`cpuid.h`) | not tested |

So **TMU works on Windows** — it is pyTsetlinMachine and green-tsetlin that do not, because both
ship POSIX/GCC-only C. A Windows user needs MSVC build tools for the git install; if they lack them
and will not install ~6-7 GB of tooling, point them at WSL or the from-scratch option below.

**Never report a platform as broken without checking your own call first** — see the `seed=0` trap
in `references/tmu.md`, which produced exactly this false conclusion during skill development.

**Never substitute one library for another.** If the user said GraphTsetlinMachine, do not write
TMU code because its API feels more familiar. The constructors are genuinely different: e.g.
pyTsetlinMachine takes clauses, T and s as bare positional arguments [PyTM-README §Multiclass Demo],
TMU's `TMClassifier` takes them as keyword arguments and has no `epochs` parameter on `fit`
[TMU-PPExample], and GraphTsetlinMachine requires a `Graphs` object built through a fixed call
sequence [GTM-README §Tutorial].

Read only the reference file for the chosen library. Loading all of them wastes context.

### When writing it from scratch is the right call

Reaching for the library first is correct, but a vanilla TM is unusual among ML models: it is small
enough to write. Published research repos do this routinely rather than as a fallback —
`AnwarDebes/THGTM` depends on nothing but numpy/scipy/matplotlib, and `AnwarDebes/Multi-HGTM` ships
its own 253-line `MultiClassTsetlinMachine` in numpy, using GraphTM only for its GPU graph towers
[Multi-HGTM-tsetlin]. Its header states the reasons plainly: the CPU path must run and be
unit-testable without a GPU, and they kept the ±1 vote with no clause weights *"because the
resulting clauses read directly as logical rules"* [Multi-HGTM-tsetlin].

Even CAIR's original `cair/TsetlinMachine` is not a package — it is `TsetlinMachine.pyx` and
`MultiClassTsetlinMachine.pyx`, reference code meant to be read [TM-README].

Write it from scratch when:

- the library will not install on the user's platform and WSL/Linux is not available to them;
- the user is learning how a TM works, where implementing Type I and Type II feedback teaches more
  than any explanation;
- the learning rule itself needs modifying, which is the usual reason in research code;
- maximum clause readability matters more than accuracy, so dropping clause weights is a feature.

Expect roughly 70 lines for two-class and 250–350 for multiclass with clause extraction. Say plainly
that this is a reference implementation: it will be slower than the C/CUDA kernels and will not
carry Convolutional, Coalesced or autoencoder variants. Do **not** hand-roll a TM when the user
asked for a published architecture — that is what the libraries are for. [heuristic]

**Read `references/from-scratch.md` before writing a line of it.** It carries the algorithm as read
out of CAIR's own reference implementation: TA state machine and initialisation, both feedback
tables with their exact probabilities, the clipped vote sum, the feedback-selection formula, and the
pairwise multiclass update. It also documents the mistake that costs the most time — an empty clause
must output 1 while learning and 0 at prediction, which fails silently if you get it wrong.

## Step 2 — Smoke-test before touching the user's data

Run the library's own Noisy XOR demo first and confirm it reaches the accuracy the README reports.
This is not ceremony: TMU's signature failure is a silent infinite hang rather than an exception,
so without a known-good baseline you cannot tell a broken environment from a slow one
[verified-2026-09-14].
Both pyTsetlinMachine [PyTM-README §Multiclass Demo] and GraphTsetlinMachine
[GTM-README §Execution] ship one, and `scripts/smoke_test.py` here contains a minimal version of
each.

This separates install/CUDA/version problems from modelling problems. Doing it after the fact
costs the user an hour of confusing results. [heuristic]

## Step 3 — Booleanize explicitly, and show the result

A basic TM takes a vector of **Boolean** features [TM-README §Classification]. Continuous data must
be converted first. Always print how many Boolean features came out, and show one example row
before and after. See `references/booleanization.md`.

Use `scripts/booleanize.py` rather than writing another thermometer encoder — it fits cuts on train
only, drops duplicate thresholds, and gives low-cardinality columns one cut per distinct value.
Measured on the readmission data, that changes accuracy very little (mean AUC 0.731 vs 0.735 for
naive quantiles, well inside split-to-split noise) — it is tidiness, not a fix
[verified-2026-09-14]. Print its `describe()` output anyway: the feature count, bit density and
constant-bit warnings are cheap, and a degenerate encoding is obvious there and invisible in an
accuracy score.

## Step 4 — Set T and s deliberately, never silently

`s` controls the frequency (granularity) of the patterns produced [TM-README §Learning].
`T` is the target for the clause vote sum, and governs how clauses are allocated across patterns
[TM-README §Learning].

These are the parameters that matter most. A hyperparameter study on CIFAR-10 with TM Composites
found the convolution window, T and s to be the most important for accuracy, and noted this was
consistent with earlier MNIST findings that T and s dominate [arXiv:2406.00704]. The Multigranular
TM paper exists precisely because the standard TM needs a three-dimensional hyperparameter search
and good values are not trivial to find, with the search space shifting as the clause count changes
[arXiv:1909.07310].

So: state the starting values, say out loud that they are starting values, and offer a small search
over T and s rather than presenting a single run as a result. Published starting points per task
are listed in the per-library reference files.

## Step 5 — Always extract the clauses

Interpretability is the reason to choose a TM. `cair/TsetlinMachine` describes the model as solving
pattern recognition with easy-to-interpret propositional formulas [TM-README §intro]. A script that
only calls `fit` and `predict` gives the user a worse gradient-boosted tree.

So never deliver a TM script without also printing the learned clauses, and then explain a few of
them in plain language using the user's own feature names. Per-library recipes are in the reference
files; the conventions are not guessable and **differ between libraries in a way that fails
silently**:

- pyTsetlinMachine: `ta_action(class, clause, literal)` — first argument is the **class**. Literal
  index `k` below the feature count means `x_k`, at or above means `¬x_{k-features}`; clauses are
  printed with even indices as positive and odd as negative for a given class
  [PyTM-README §Interpretability Demo].
- GraphTsetlinMachine: `ta_action(layer, clause, literal)` — first argument is the **depth/layer**,
  not the class. Class information comes from `tm.get_state()[1].reshape(n_classes, -1)`
  [GTM-NoisyXORDemo].
- TMU: `get_ta_action(clause, ta, the_class=…, polarity=…)` — the **clause comes first**, and
  polarity is an explicit argument (`0` positive, `1` negative) rather than pyTM's even/odd clause
  index. Iterate `range(number_of_clauses // 2)` per polarity
  [TMU-InterpretabilityDemo].

All three take three-ish arguments and none of them raise when you use the wrong order, so a
mismatched call produces a plausible clause listing that means something entirely different. Read
the right reference file before writing the extraction loop.

With TMU, prefer ranking clauses by `clause_precision` / `clause_recall` and showing the user the
top handful with their support, rather than dumping every clause — a domain expert wants the few
rules carrying the decision, and precision makes that selection defensible rather than arbitrary
[TMU-InterpretabilityDemo].

`scripts/smoke_test.py` has working clause extraction for GraphTM, and `scripts/show_clauses.py`
for pyTsetlinMachine. For TMU, follow the `InterpretabilityDemo.py` shape quoted in
`references/tmu.md`.

## Diagnostic table

Use this when a run produces a disappointing number. Sourced rows first.

| Symptom | Mechanism | Action |
|---|---|---|
| Accuracy plateaus but train accuracy is still climbing | Normal for a TM: on binarized MNIST, test accuracy kept improving alongside training accuracy up to 98.2% at 400 epochs, unlike backprop where test accuracy eventually degrades from overfitting [TM-README §Learning Behaviour] | Train longer before concluding anything |
| Clauses are too specific / memorising | Type I rule 1 rewards *Include* with probability (s−1)/s; large s makes inclusion overwhelmingly likely, producing fine-grained patterns [TM-README §Learning] | Lower `s` |
| Clauses are too coarse / near-empty | Type I rule 2 rewards *Exclude* with probability 1/s; small s strips literals out [TM-README §Learning] | Raise `s` |
| Model stops improving early | Once the vote sum `v` reaches the target `T`, no clauses are reinforced at all [TM-README §Learning] | Raise `T` |
| Graph has edges but they change nothing | The number of message rounds sets the depth of reasoning; with depth 1, clauses only ever see a node's own properties [GTM-README §Deeper Logical Reasoning]. Confirmed: GraphTM reduces to disconnected/convolutional CoTM with no edges [arXiv:2507.14874 §3.2], and the CIFAR-10 demo ships depth=1 for exactly that disconnected encoding [GTM-CIFAR10Demo] | Raise depth; published configs go to depth 12 on long-dependency tasks — see `references/graphtm.md` |
| GraphTM runs but accuracy is at chance | `graphs.encode()` was likely never called. It is required on **both** train and test graph sets after all properties are added, and the README tutorial does not mention it [GTM-NoisyXORDemo, GTM-CIFAR10Demo] | Call `encode()` before `fit` |
| GraphTM test accuracy nonsensical vs train | Test graphs must be built with `Graphs(n, init_with=graphs_train)` so they inherit the symbol/hypervector mapping [GTM-NoisyXORDemo] | Rebuild test graphs with `init_with` |
| GraphTM on a plain tabular problem feels pointless | With a single node you get a Coalesced Vanilla TM, and with many nodes but no edges a Coalesced Convolutional TM [GTM-README §Vanilla MNIST, §Convolutional MNIST], confirmed experimentally on MNIST/Fashion-MNIST [arXiv:2507.14874 §3.2] | Route to TMU or pyTsetlinMachine instead |
| A node cannot tell left from right in a sequence | With only one edge type a node cannot distinguish an 'A' on its left from one on its right [GTM-README §Sequence Classification] | Use distinct edge types per direction |
| Too many clauses, training is slow | The Weighted TM matched plain TM accuracy on MNIST, IMDb and Connect-4 using 1/4, 1/3 and 1/50 of the clauses respectively [arXiv:1911.12607] | Enable weighted clauses |
| TM far below a random forest / logistic regression on the same split | Could be the Boolean encoding (a clause can only combine literals that exist) or the training/selection loop. A TM's test score drifts across epochs, which mimics an encoding ceiling: one run read 0.649 vs a forest's 0.90 and the cause was reporting the final epoch while selecting on the best one, not the encoder [verified-2026-09-14] | Change one thing at a time. Check a single literal can express the rule you suspect is missing (three lines of numpy), and check epoch selection, before rebuilding the encoder |
| Training slow after booleanization | Feature count drives the number of Tsetlin Automata, since each clause has automata over all literals including negations [TM-README §Classification, §Learning] | Reduce bits per feature — see `references/booleanization.md` |
| Underfitting a task with several distinct sub-patterns | Clauses distribute themselves across frequent patterns via resource allocation [TM-README §Learning]; too few clauses cannot cover them all [heuristic] | Raise clause count first, before touching T or s [heuristic] |

## Explaining TMs to the user

When the user wants to understand the method, read `references/concepts.md` and work from the
mechanism, not from analogies to neural networks. If they are reading the book, check the vocabulary
map in `references/from-scratch.md` first: Chapter 1 says Memorize/Forget, Recognize/Erase/Reject
Feedback and Vote Margin where the papers and code say Include/Exclude, Type Ia/Ib/II and `T`. Same
algorithm, different words — and a user quoting one while you answer in the other will not realise
you agree [TM-Book-Ch1]. The one analogy that *is* sourced: the number of
clauses roughly corresponds to the number of hidden nodes in a neural network layer
[TextTM-README]. Everything else — gradients, learning rates, backprop intuitions — does not
transfer.

Point the user at the book, which is free and aimed at undergraduates [TM-Book]. Note that only
chapters 1, 2, 4 and 7 are currently downloadable; regression, pattern weighing, auto-encoding,
hierarchical representation, graphs and NLP/image processing are listed without links [TM-Book].
Chapter 1 has an accompanying Jupyter notebook [TM-Book].

## Files in this skill

- `references/concepts.md` — how a TM classifies and learns, fully sourced
- `references/from-scratch.md` — the implementable algorithm, verified against CAIR's reference
  Cython: TA states, feedback tables, empty-clause semantics, pairwise multiclass, vocabulary map
- `references/pytsetlinmachine.md` — API, published configs, clause extraction
- `references/tmu.md` — features, install prerequisites
- `references/graphtm.md` — construction sequence, symbols, depth, CUDA config
- `references/booleanization.md` — turning real data into Boolean features
- `references/sources.md` — citation tags resolved to URLs
- `scripts/smoke_test.py` — minimal Noisy XOR for TMU, pyTsetlinMachine and GraphTsetlinMachine
  (`python smoke_test.py tmu|pytm|graphtm`); the TMU path carries a hang watchdog
- `scripts/booleanize.py` — thermometer booleanization that avoids the low-cardinality cut trap,
  reports feature count, bit density and constant bits; `python booleanize.py` self-checks
- `scripts/show_clauses.py` — clause extraction for pyTsetlinMachine

## Maintenance warning

These libraries move. GraphTsetlinMachine's roadmap still lists rewriting `graphs.py` in C or numba,
plus adding an autoencoder, regression, multi-output and adjacency-matrix initialisation
[GTM-README §Roadmap]; TMU lists a multi-task classifier, one-vs-one classifier, incremental clause
evaluation and TMComposites as upcoming or in progress [TMU-README §Features]. Re-verify the
reference files against the repos before trusting them on a new project. [heuristic]
