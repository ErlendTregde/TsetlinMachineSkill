# GraphTsetlinMachine

Source of record: `cair/GraphTsetlinMachine` README [GTM-README] and the full paper *The Tsetlin
Machine Goes Deep: Logical Learning and Reasoning With Graphs* [arXiv:2507.14874]. The README
itself lists the paper as forthcoming under a slightly different title [GTM-README §Paper]; treat
[arXiv:2507.14874] as the authoritative technical description and the README as the API tutorial.

GraphTM is explicitly built on the Coalesced Tsetlin Machine (CoTM) [arXiv:2108.07594], extended
with a depth hyperparameter D, and formalised with four hyperparameters: depth D, number of
clauses m, specificity s, and voting margin T [arXiv:2507.14874 §2 "Algorithm"].

## When to use it

Use it for directed and labeled multigraphs, vector-symbolic node properties and edge types, nested
(deep) clauses, and arbitrarily sized inputs [GTM-README §Features]. It also subsumes the Vanilla,
Multiclass, Convolutional and Coalesced TMs [GTM-README §Features].

Do **not** reach for it on plain tabular data. With a single node you obtain a Coalesced Vanilla TM
[GTM-README §Vanilla MNIST], and with many nodes but no edges a Coalesced Convolutional TM
[GTM-README §Convolutional MNIST] — both available more directly elsewhere.

## Install

```
pip3 install graphtsetlinmachine
```

or build from source with `python ./setup.py sdist` then install the produced tarball
[GTM-README §Installation].

## The construction sequence — get this wrong and nothing works

Order is load-bearing. From the tutorial [GTM-README §Tutorial]:

**1. Initialise the graph set.**

```python
graphs_train = Graphs(
    10000,
    symbols=['A', 'B'],
    hypervector_size=32,
    hypervector_bits=2
)
```

- First argument: how many graphs you will create.
- `symbols`: the vocabulary used to assign properties to nodes. You can define as many as you like.
  They need not be strings — the CIFAR-10 demo mixes string symbols (`"C:0"`, `"R:3"`) with plain
  integers for pixel positions in one symbol list [GTM-CIFAR10Demo].
- `hypervector_size` / `hypervector_bits`: larger hypervectors hold more symbols. With size 32 and
  2 bits you get 32*31/2 = 496 unique bit pairs — plenty for two symbols.
- Hypervector generation and compilation happen automatically during initialisation, using sparse
  distributed codes.
- Optional flags `double_hashing` and `one_hot_encoding` are also accepted here, and should be
  passed identically to the machine [GTM-NoisyXORDemo, GTM-CIFAR10Demo].

**Sizing the hypervector — real datapoints, not a guess.** The CIFAR-10 demo has
`2*dim + patch_size*patch_size*3` symbols (with default patch size 8: 50 positional symbols plus
192 pixel symbols, 242 total) and uses `hypervector_size=128`, `hypervector_bits=2`
[GTM-CIFAR10Demo]. The paper's CIFAR-10 configuration goes further still — 2,500 symbols at
hypervector size 128 [arXiv:2507.14874 Table 16]. So symbol count can substantially exceed
hypervector size; capacity comes from bit *combinations*, not one bit per symbol. Collisions are
tolerated by design, and `double_hashing` exists to mitigate them. Scale up if accuracy stalls, and
note the paper observed that too many clauses can cause message-hypervector collisions that
actually hurt accuracy [arXiv:2507.14874 §3.8].

**2. Declare node counts, then prepare.**

```python
for graph_id in range(10000):
    graphs_train.set_number_of_graph_nodes(graph_id, 2)

graphs_train.prepare_node_configuration()
```

**3. Add nodes, with their outgoing edge count.**

```python
for graph_id in range(10000):
    number_of_outgoing_edges = 1
    graphs_train.add_graph_node(graph_id, 'Node 1', number_of_outgoing_edges)
    graphs_train.add_graph_node(graph_id, 'Node 2', number_of_outgoing_edges)
```

The outgoing-edge count is declared **here**, not when you add the edge. Node identifiers can be
strings (`'Node 1'`) or integers — the CIFAR-10 demo uses integer `node_id` values and passes `0`
outgoing edges for a disconnected encoding [GTM-CIFAR10Demo].

**4. Prepare edges, then add them.**

```python
graphs_train.prepare_edge_configuration()

for graph_id in range(10000):
    edge_type = "Plain"
    graphs_train.add_graph_node_edge(graph_id, 'Node 1', 'Node 2', edge_type)
    graphs_train.add_graph_node_edge(graph_id, 'Node 2', 'Node 1', edge_type)
```

Graphs are **directed**, so two edges are needed to connect a pair in both directions
[GTM-README §Adding Edges].

**5. Add node properties.**

```python
graphs_train.add_graph_node_property(graph_id, 'Node 1', x1)
```

A node can carry many properties — in the MNIST example every white pixel becomes its own symbol
`W(x,y)` attached to the node [GTM-README §Vanilla MNIST].

So the sequence is: `Graphs(...)` → `set_number_of_graph_nodes` → `prepare_node_configuration()` →
`add_graph_node` → `prepare_edge_configuration()` → `add_graph_node_edge` →
`add_graph_node_property` → **`encode()`**.

`encode()` is the step the README tutorial never mentions but both demos require
[GTM-NoisyXORDemo, GTM-CIFAR10Demo]. Note that `prepare_edge_configuration()` is called even in the
zero-edge CIFAR-10 case [GTM-CIFAR10Demo] — it is not optional.

## Edge types matter

In the sequence-classification demo the task is counting consecutive 'A's. From a single node's
view, all classes look identical — each node only sees an 'A' or a space. By considering the nodes
to its Left and Right, a node can gather information about the sequence
[GTM-README §Sequence Classification].

The README is explicit that with only a single edge type a node could not distinguish an 'A' to its
left from one to its right, making the task harder, so two edge types are beneficial
[GTM-README §Sequence Classification].

## Depth is what makes edges do anything

Inference works by **clause-driven message passing**: a pool of clauses examines each node, and
whenever a clause matches a node's properties it sends a message along that node's outgoing edges.
A receiving node appends the message to its properties, so messages supplement node properties with
contextual information [GTM-README §Clause-Driven Message Passing].

Each node is described by Boolean features such as `[A, Right⊗C, Left⊗C]`, where `⊗` binds an edge
type to a clause — an explainable way of naming the feature. The message features are initialised
to False and updated by arriving messages; truth values default to False to minimise message
passing [GTM-README §Logical Reasoning With Nested Clauses].

In the first round the clause can only consider node properties, because the truth values to the
left and right are not yet computed. After message passing, the message literals activate and the
clause is matched in full. The per-node truth values are finally ORed together to give the
classification [GTM-README §Logical Reasoning With Nested Clauses].

**The number of message rounds decides the depth of the reasoning.** Three layers of reasoning
consist of local reasoning followed by two rounds of message passing [GTM-README §Deeper Logical
Reasoning]. In the paper's notation, D is the number of layers; layer 0 is the node layer and
layers 1 to D−1 are message layers [arXiv:2507.14874 §A.3]. As layers increase, the receptive
field of a node expands to neighbors further away — at layer 2, a node's messages can already
encode information from neighbors two hops away [arXiv:2507.14874 §A.3 "The receptive field"].

**This is confirmed experimentally, not just architecturally.** The paper ran a "disconnected
nodes" experiment specifically to verify that GraphTM reduces to the disconnected/convolutional
variant of CoTM when there are no edges between nodes, testing this on MNIST, Fashion-MNIST and
CIFAR-10 [arXiv:2507.14874 §3.2]. Results on MNIST and Fashion-MNIST were indeed comparable between
GraphTM and CoTM, confirming the expectation [arXiv:2507.14874 §3.2] — which is exactly the
`GTM-README §Vanilla MNIST / §Convolutional MNIST` "no edges → Coalesced [Convolutional] TM" point
made from the tutorial side.

So: if a user builds a careful multigraph and gets mediocre accuracy, check the depth before
anything else. With too few message rounds, distant structure the user encoded into edges may
never reach the clauses that need it.

Because clauses can draw on the outcomes of other clauses, the automata teams build **nested
clauses** — hierarchical clause structures centred on the nodes [GTM-README §Logical Learning With
Nested Clauses].

## Train/test graphs — verified

Confirmed from the repo's own demos [GTM-NoisyXORDemo, GTM-CIFAR10Demo]:

- Test graphs are constructed with **`Graphs(n, init_with=graphs_train)`** — no `symbols`, no
  `hypervector_size`, no `hypervector_bits`. Those are inherited from the training graphs.
- **`graphs.encode()` must be called on both** train and test graph sets, after all properties are
  added and before the machine touches them. Forgetting it is a silent-failure class of bug.
- The full node/edge construction sequence is then repeated identically for the test set.

## The machine: MultiClassGraphTsetlinMachine

Verified signature [GTM-NoisyXORDemo, GTM-CIFAR10Demo]:

```python
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

tm = MultiClassGraphTsetlinMachine(
    number_of_clauses,          # positional
    T,                          # positional
    s,                          # positional
    number_of_state_bits=8,
    depth=2,
    message_size=256,
    message_bits=2,
    max_included_literals=4,
    double_hashing=False,
    one_hot_encoding=False,
)

for i in range(epochs):
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()
```

Points that matter:

- The first three arguments are **positional**, same shape as pyTsetlinMachine's
  `MultiClassTsetlinMachine(clauses, T, s)` [PyTM-README §Multiclass Demo]. The rest are keywords.
- **`fit` takes the `Graphs` object, not an X array** — and uses the `epochs=1, incremental=True`
  loop idiom, matching pyTsetlinMachine [GTM-NoisyXORDemo; PyTM-README §MNIST Demo].
- `depth` is where message rounds are set. The Noisy XOR demo defaults to 2; the CIFAR-10 demo
  defaults to **1**, consistent with it being a disconnected-node (no-edge) encoding
  [GTM-NoisyXORDemo, GTM-CIFAR10Demo].
- `message_size` / `message_bits` control the message hypervector space, separate from
  `hypervector_size` / `hypervector_bits` which are set on `Graphs`. Both demos default to
  message_size=256, message_bits=2.
- `double_hashing` and `one_hot_encoding` are flags on **both** `Graphs` and the machine, and the
  demos pass the same value to both. Keep them consistent [GTM-NoisyXORDemo, GTM-CIFAR10Demo].
- **Neither demo passes `grid` or `block`.** The README documents those for DGX-2/A100 and DGX H100
  [GTM-README §CUDA Configurations], but they are optional — do not add them by default.

### Demo defaults, as published

| Parameter | Noisy XOR demo | CIFAR-10 demo |
|---|---|---|
| epochs | 10 | 250 |
| number-of-clauses | 10 | 40,000 |
| T | 100 | 7,500 |
| s | 1.0 | 20.0 |
| depth | 2 | 1 |
| hypervector-size | 32 | 128 |
| hypervector-bits | 2 | 2 |
| message-size | 256 | 256 |
| message-bits | 2 | 2 |
| max-included-literals | 4 | 32 |
| number-of-state-bits | 8 | 8 |

[GTM-NoisyXORDemo, GTM-CIFAR10Demo]

Note the CIFAR-10 demo's s=20.0 and hypervector size 128 match the paper's CIFAR-10 row
[arXiv:2507.14874 Table 16], though clause count and T differ (40,000/7,500 in the demo vs
80,000/15,000 in the paper) — the demo is a smaller configuration than the published result.

## Building image graphs (the CIFAR-10 pattern)

Worth showing a user who wants images-as-graphs [GTM-CIFAR10Demo]:

- Binarize first, per colour channel, with `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C,
  THRESH_BINARY, 11, 2)`.
- Node count is `dim*dim` where `dim = 32 - patch_size + 1` — one node per sliding patch position.
- Symbols are built as a list combining **column symbols `"C:%d"`, row symbols `"R:%d"`, and one
  integer symbol per pixel position in the patch** (`patch_size*patch_size*3` of them for RGB).
- Nodes are added with **`add_graph_node(graph_id, node_id, 0)`** — zero outgoing edges, and
  `node_id` is an integer, not a string name. Node names can be ints or strings.
- Patches are extracted with `skimage.util.view_as_windows`, and only nonzero (white) pixels are
  added as properties via `patch.nonzero()[0]`.
- Row/column position is encoded **cumulatively** — for a node at `(q, r)`, properties `C:0..C:q`
  and `R:0..R:r` are all added, a thermometer-style positional encoding rather than a single
  coordinate symbol.
- `prepare_edge_configuration()` is still called even with zero edges.

## Published hyperparameter configurations

Real starting points, not `[heuristic]` guesses. All from the paper's hyperparameter appendix
[arXiv:2507.14874 §A.4], which reports the exact settings behind every experiment in the results
section. Use these as the initial guess for a similar task, then search from there.

**Disconnected nodes (no edges — the "reduces to CoTM" case)** [arXiv:2507.14874 Table 16]:

| Dataset | Clauses | Symbols | Conv. window | HV size | Depth | Epochs | T | s | Max literals |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | 2,500 | 138 | 10×10 | 128 | 1 | 30 | 3,125 | 10.0 | — |
| Fashion-MNIST | 40,000 | 124 | 3×3 | 128 | 1 | 30 | 15,000 | 10.0 | — |
| CIFAR-10 | 80,000 | 2,500 | 8×8 | 128 | 1 | 30 | 15,000 | 20.0 | 32 |

Depth 1 here means no message passing — node-only, matching the disconnected-graph case.

**Genuinely graph-structured tasks**, where depth is greater than 1 and edges carry information
[arXiv:2507.14874 Tables 17–22]:

| Task | Clauses | HV size | Msg HV size | Depth | Epochs | T | s | Max literals |
|---|---|---|---|---|---|---|---|---|
| MNIST Superpixel (k-NN graph) | 60,000 | 128 | 256 | 35 | 50 | 100,000 | 0.5 | 32 |
| IMDB / Yelp / MPQA sentiment | 10,000 | 2,048 | 1,024 | 2 | 40 | 100,000 | 15 | — |
| Tangram action coreference, 3-utterance | 850 | 256 | 256 | 6 | 50 | 9,000 | 1 | — |
| Tangram action coreference, 5-utterance | 800 | 512 | 256 | 12 | 50 | 9,000 | 1 | — |
| MovieLens top-N recommendation | 4,096 | 2,096 (64 bits) | — | 1 | 10 | 500 | 5.0 | 256 |
| Viral genome, 2–5 classes | 500–2,000 | 512 | 512 | 2 | 10 | 2,000 | 1 | 200 |
| Multivalue noisy XOR | 4–2,000 | 2,048 | 2,048–8,196 | 2 | — | 10× clauses | 2.2 | — |

Patterns worth pointing out to a user:

- **Depth scales with how far information needs to travel.** The Tangram task, where actions form
  long dependency chains, uses depth up to 12; sentiment classification on fully-connected sentence
  graphs uses depth 2; disconnected-node image tasks use depth 1
  [arXiv:2507.14874 Tables 16–19].
- **s tends to be small (near 1) on tasks with deep, highly structured dependencies** (Tangram: s=1;
  viral genome: s=1) **and larger on flatter, noisier tasks** (MNIST/F-MNIST: s=10; CIFAR-10: s=20)
  [arXiv:2507.14874 Tables 16, 19, 21].
- **T scales with clause count** — the multivalue noisy XOR experiments fixed T at 10× the clause
  count for every configuration tested [arXiv:2507.14874 Table 22].
- On the recommendation-system task with three interconnected entity types (customers, products,
  categories), the paper used 2,000 clauses, T=10,000, s=10.0, hypervector size 4,096, 256
  hypervector bits, message size 256, message bits 2 [arXiv:2507.14874 §A.4.5] — GraphTM
  outperformed a standard TM by a wide margin at every noise level tested on this task
  [arXiv:2507.14874 §3.6, Table 7].

## The formal algorithm, if you need to explain it precisely

Given an input graph G, per training/inference step [arXiv:2507.14874 §2 "Algorithm"]:

1. Evaluate every layer-zero clause component against every node's properties.
2. If a component matches a node, submit a message (bound to the edge type) to that node's
   neighbors per the graph's edges.
3. For each subsequent layer up to depth D−1: evaluate that layer's clause components against the
   accumulated messages, and again submit messages to neighbors on a match.
4. The full clause's truth value is the conjunction of all its per-layer components.
5. Perform standard Coalesced TM voting and updates using the full clauses.

A clause matching **any single node** in the graph counts as the clause matching the whole graph —
per-node results are combined with OR before voting [arXiv:2507.14874 §3.1, worked "BAAAE" example].

## CUDA configuration

`MultiClassGraphTsetlinMachine` takes explicit grid and block sizes [GTM-README §CUDA
Configurations]:

- DGX-2 and A100: `grid=(16*13,1,1)`, `block=(128,1,1)`
- DGX H100: `grid=(16*13*4,1,1)`, `block=(128,1,1)`

If the user is on different hardware, say that the README only documents these two configurations
rather than inventing a third.

## Worked demos in the repo

Four, all in `examples/` [GTM-README §Demos]:

- **Noisy XOR** — two nodes, one property each (A or B), label 0 if the properties match else 1,
  with labels randomly inverted with probability 0.01 to add noise [GTM-README §Tutorial]. Expected
  output starts around 99.15% test accuracy [GTM-README §Execution].
- **Vanilla MNIST** — one node per image, white pixels as `W(x,y)` symbols.
- **Convolutional MNIST** — each image broken into a 19x19 grid of 10x10 patches, with row `R(y)`
  and column `C(x)` symbols added so the machine can reason about pixel patterns *and* their
  location.
- **Noisy XOR with MNIST images** — handwritten 0s and 1s in place of the symbols A and B, so the
  machine must learn digit appearance and the XOR relation simultaneously under noisy labels.

## Reading GraphTM clauses

Interpretation works differently here than in pyTsetlinMachine, and there are **two levels**: what
the code actually gives you, and the readable form the paper presents.

### What the code gives you

Verified from both demos [GTM-NoisyXORDemo, GTM-CIFAR10Demo]:

```python
weights = tm.get_state()[1].reshape(2, -1)   # [class, clause]
for i in range(tm.number_of_clauses):
    print("Clause #%d W:(%d %d)" % (i, weights[0, i], weights[1, i]), end=' ')
    l = []
    for k in range(graphs_train.hypervector_size * 2):
        if tm.ta_action(0, i, k):
            if k < graphs_train.hypervector_size:
                l.append("x%d" % k)
            else:
                l.append("NOT x%d" % (k - graphs_train.hypervector_size))
    print(" AND ".join(l))
```

**Critical difference from pyTsetlinMachine: the first argument to `ta_action` is the DEPTH/LAYER
index, not the class index.** `tm.ta_action(0, clause, literal)` reads the layer-zero (node)
component. The Noisy XOR demo has a commented-out block showing the message layer is read with
`tm.ta_action(1, clause, literal)`, iterating over `message_size * 2` instead of
`hypervector_size * 2` [GTM-NoisyXORDemo].

This is an easy and silent mistake: in pyTsetlinMachine, `ta_action(0, j, k)` means class 0
[PyTM-README §Interpretability Demo]; in GraphTsetlinMachine it means layer 0. Never carry the
call across libraries.

Class information instead comes from the **weight matrix**: `tm.get_state()[1].reshape(2, -1)`
gives per-class weights per clause (reshape's `2` is the class count — adjust for multiclass).
Both demos print `W:(w0 w1)` alongside each clause [GTM-NoisyXORDemo, GTM-CIFAR10Demo].

Useful attributes exposed for inspection: `graphs.hypervectors`, `tm.hypervectors`,
`graphs.edge_type_id`, `graphs.hypervector_size`, `graphs.number_of_graph_nodes`,
`tm.number_of_clauses` [GTM-NoisyXORDemo, GTM-CIFAR10Demo].

### The readable form the paper presents

The raw output above is **hypervector bit indices** (`x37 AND NOT x12`), not symbol names. The
paper's much more readable notation is a decoded view of the same thing
[arXiv:2507.14874 §3.1]:

```
C0 = ¬A ∧ r1:0 ∧ r1:1;  [3, -3]
C2 = A ∧ r1:2 ∧ r1:3 ∧ ¬r1:0 ∧ ¬l1:0;  [-5, 6]
```

Reading convention:

- A bare property like `A` or `¬A` is a layer-zero (node) literal.
- `r1:0` means: "a message was received via the **r**ight edge, in message layer **1**, reporting
  that the neighbor matched clause **0** in the previous layer." `l1:2` is the mirror for the left
  edge. Number before the colon is the layer; number after is the referenced clause index.
- Every message term traces back, layer by layer, to node-level properties — the paper identifies
  this traceability as key to GraphTM's interpretability [arXiv:2507.14874 §3.1].
- The bracketed vector is the per-class weight vector, matching `get_state()[1]` above.
- A clause matches the whole graph if it matches at **any** node — per-node results are ORed
  before the weighted vote [arXiv:2507.14874 §3.1].

**Going from bit indices to symbol names is not done for you.** Mapping raw `x37` output back to
symbols requires the hypervector table (`graphs.hypervectors`) and, since symbols occupy
`hypervector_bits` bits each and may collide, the mapping is not always clean. If a user wants
paper-style readable clauses, expect to write that decoder — neither demo ships one. [heuristic]

For image tasks the paper takes a different route: aggregate active clauses over an input image and
render a heatmap — red where white-pixel symbols activate the clause, blue for absence
[arXiv:2507.14874 §3.2, Figure 3]. With the cumulative `C:`/`R:` positional symbols from the
CIFAR-10 demo, patches can be located back in the image.

## Roadmap (things that do not exist yet)

Listed as future work [GTM-README §Roadmap]: rewriting `graphs.py` in C or numba for much faster
graph construction, adding an autoencoder, adding regression, adding multi-output, and graph
initialisation from an adjacency matrix.

If a user asks for regression or an autoencoder in GraphTM, tell them it is on the roadmap and not
present, rather than writing code against an API that does not exist.

## Example use case worth citing to a user

The README sketches a hospital setting: nodes capture ECG data and the medical narrative in
electronic health records, while edge types encode relationships — *Measurement* edges relate
medical tests to a patient, *Condition* edges relate diseases to patients. Tasks include
forecasting, alerting, decision-making, situation assessment, risk mitigation, knowledge discovery
and optimization [GTM-README §Example Use Case].
