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

| User's situation | Library | Read |
|---|---|---|
| Boolean or tabular data, binary/multiclass, CPU, wants something running fast | pyTsetlinMachine | `references/pytsetlinmachine.md` |
| Wants Coalesced / autoencoder / drop clause / Type III feedback / literal budget | TMU | `references/tmu.md` |
| Images with spatial structure | Convolutional TM (in pyTsetlinMachine or TMU) | `references/pytsetlinmachine.md` |
| Graphs, sequences, variable-sized input, multimodal, relations between parts | GraphTsetlinMachine | `references/graphtm.md` |
| Continuous features anywhere in the above | still need booleanization | `references/booleanization.md` |
| Asking *how a TM works* rather than for code | — | `references/concepts.md` |

**Never substitute one library for another.** If the user said GraphTsetlinMachine, do not write
TMU code because its API feels more familiar. The constructors are genuinely different: e.g.
pyTsetlinMachine takes clauses, T and s as bare positional arguments [PyTM-README §Multiclass Demo],
TMU's `TMClassifier` takes them as keyword arguments and has no `epochs` parameter on `fit`
[TMU-PPExample], and GraphTsetlinMachine requires a `Graphs` object built through a fixed call
sequence [GTM-README §Tutorial].

Read only the reference file for the chosen library. Loading all of them wastes context.

## Step 2 — Smoke-test before touching the user's data

Run the library's own Noisy XOR demo first and confirm it reaches the accuracy the README reports.
Both pyTsetlinMachine [PyTM-README §Multiclass Demo] and GraphTsetlinMachine
[GTM-README §Execution] ship one, and `scripts/smoke_test.py` here contains a minimal version of
each.

This separates install/CUDA/version problems from modelling problems. Doing it after the fact
costs the user an hour of confusing results. [heuristic]

## Step 3 — Booleanize explicitly, and show the result

A basic TM takes a vector of **Boolean** features [TM-README §Classification]. Continuous data must
be converted first. Always print how many Boolean features came out, and show one example row
before and after. See `references/booleanization.md`.

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

Carrying one library's call into the other produces plausible output that means something entirely
different. Read the right reference file.

`scripts/smoke_test.py` has working clause extraction for both.

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
| Training slow after booleanization | Feature count drives the number of Tsetlin Automata, since each clause has automata over all literals including negations [TM-README §Classification, §Learning] | Reduce bits per feature — see `references/booleanization.md` |
| Underfitting a task with several distinct sub-patterns | Clauses distribute themselves across frequent patterns via resource allocation [TM-README §Learning]; too few clauses cannot cover them all [heuristic] | Raise clause count first, before touching T or s [heuristic] |

## Explaining TMs to the user

When the user wants to understand the method, read `references/concepts.md` and work from the
mechanism, not from analogies to neural networks. The one analogy that *is* sourced: the number of
clauses roughly corresponds to the number of hidden nodes in a neural network layer
[TextTM-README]. Everything else — gradients, learning rates, backprop intuitions — does not
transfer.

Point the user at the book, which is free and aimed at undergraduates [TM-Book]. Note that only
chapters 1, 2, 4 and 7 are currently downloadable; regression, pattern weighing, auto-encoding,
hierarchical representation, graphs and NLP/image processing are listed without links [TM-Book].
Chapter 1 has an accompanying Jupyter notebook [TM-Book].

## Files in this skill

- `references/concepts.md` — how a TM classifies and learns, fully sourced
- `references/pytsetlinmachine.md` — API, published configs, clause extraction
- `references/tmu.md` — features, install prerequisites
- `references/graphtm.md` — construction sequence, symbols, depth, CUDA config
- `references/booleanization.md` — turning real data into Boolean features
- `references/sources.md` — citation tags resolved to URLs
- `scripts/smoke_test.py` — minimal Noisy XOR for pyTsetlinMachine and GraphTsetlinMachine
- `scripts/show_clauses.py` — clause extraction for pyTsetlinMachine

## Maintenance warning

These libraries move. GraphTsetlinMachine's roadmap still lists rewriting `graphs.py` in C or numba,
plus adding an autoencoder, regression, multi-output and adjacency-matrix initialisation
[GTM-README §Roadmap]; TMU lists a multi-task classifier, one-vs-one classifier, incremental clause
evaluation and TMComposites as upcoming or in progress [TMU-README §Features]. Re-verify the
reference files against the repos before trusting them on a new project. [heuristic]
