"""
Smoke test: confirm a Tsetlin Machine install actually works before touching real data.

Run this FIRST. If it fails, the problem is the environment, not the model.

    python smoke_test.py pytm     # pyTsetlinMachine, CPU
    python smoke_test.py graphtm  # GraphTsetlinMachine, needs CUDA

Both tasks are Noisy XOR, the demo each library ships.

Expected results:
  pytm     -> accuracy near 100%
             (the pyTsetlinMachine README reports 100% on its Noisy XOR demo, with
              MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0),
              200 epochs)
  graphtm  -> test accuracy near 99%
             (the GraphTsetlinMachine README's example output starts at 99.15%)

The graphtm path mirrors examples/NoisyXORDemo.py from the GraphTsetlinMachine repo
and uses its verified API and default hyperparameters.

Sources: see ../references/sources.md
  [PyTM-README], [GTM-README], [GTM-NoisyXORDemo], [GTM-CIFAR10Demo]
"""

import sys
import random
import numpy as np


def make_noisy_xor(n_samples=5000, n_features=12, noise=0.1, seed=0):
    """Two informative features, the rest are distractors. Labels flipped with prob `noise`."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.uint32)
    Y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.uint32)
    Y_noisy = np.where(rng.random(n_samples) <= noise, 1 - Y, Y).astype(np.uint32)
    return X, Y, Y_noisy


def smoke_pytm():
    from pyTsetlinMachine.tm import MultiClassTsetlinMachine

    X_train, _, Y_train = make_noisy_xor(seed=0)
    X_test, Y_test, _ = make_noisy_xor(seed=1)  # clean labels for evaluation

    # Positional order is (clauses, T, s) -- see ../references/pytsetlinmachine.md
    tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)
    tm.fit(X_train, Y_train, epochs=200)

    acc = 100.0 * (tm.predict(X_test) == Y_test).mean()
    print(f"pyTsetlinMachine Noisy XOR accuracy: {acc:.2f}%")
    print("Expected: near 100%. Much lower means something is wrong with the install.")
    return acc


def smoke_graphtm():
    """Noisy XOR on graphs. Mirrors examples/NoisyXORDemo.py from the repo.

    Requires CUDA. Expected test accuracy near 99% (README example output starts at 99.15%).
    """
    from GraphTsetlinMachine.graphs import Graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
    from time import time

    n = 10000
    noise = 0.01

    def build(graphs):
        """Run the fixed construction sequence and return labels. Order is load-bearing."""
        for gid in range(n):
            graphs.set_number_of_graph_nodes(gid, 2)
        graphs.prepare_node_configuration()

        for gid in range(n):
            # outgoing edge count is declared HERE, not at edge-creation time
            graphs.add_graph_node(gid, "Node 1", 1)
            graphs.add_graph_node(gid, "Node 2", 1)
        graphs.prepare_edge_configuration()

        for gid in range(n):
            # directed graphs: two edges to connect a pair both ways
            graphs.add_graph_node_edge(gid, "Node 1", "Node 2", "Plain")
            graphs.add_graph_node_edge(gid, "Node 2", "Node 1", "Plain")

        Y = np.empty(n, dtype=np.uint32)
        for gid in range(n):
            x1 = random.choice(["A", "B"])
            x2 = random.choice(["A", "B"])
            graphs.add_graph_node_property(gid, "Node 1", x1)
            graphs.add_graph_node_property(gid, "Node 2", x2)
            Y[gid] = 0 if x1 == x2 else 1
            if np.random.rand() <= noise:
                Y[gid] = 1 - Y[gid]

        graphs.encode()   # REQUIRED. Easy to forget; fails quietly if omitted.
        return Y

    print("Creating training data")
    graphs_train = Graphs(n, symbols=["A", "B"], hypervector_size=32, hypervector_bits=2)
    Y_train = build(graphs_train)

    print("Creating testing data")
    # test graphs inherit symbols/hypervector config -- do NOT re-specify them
    graphs_test = Graphs(n, init_with=graphs_train)
    Y_test = build(graphs_test)

    # first three args are POSITIONAL: clauses, T, s
    tm = MultiClassGraphTsetlinMachine(
        10,        # number_of_clauses
        100,       # T
        1.0,       # s
        number_of_state_bits=8,
        depth=2,           # message rounds; 1 = node-only, edges do nothing
        message_size=256,
        message_bits=2,
        max_included_literals=4,
    )
    # note: no grid=/block= here. Both repo demos omit them; the README documents
    # them only for DGX-2/A100 and DGX H100.

    for i in range(10):
        start = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        elapsed = time() - start
        acc_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
        acc_train = 100 * (tm.predict(graphs_train) == Y_train).mean()
        print("%d train=%.2f test=%.2f (%.2fs)" % (i, acc_train, acc_test, elapsed))

    print("\nExpected: test accuracy near 99%. Much lower means check depth and encode().")
    show_graphtm_clauses(tm, graphs_train)
    return tm, graphs_train


def show_graphtm_clauses(tm, graphs, n_classes=2):
    """Print layer-zero clause components with per-class weights.

    WARNING: ta_action's FIRST argument here is the DEPTH/LAYER index, not the class
    index. That is the opposite of pyTsetlinMachine, where it is the class. Use
    ta_action(1, ...) over range(message_size * 2) to read the message layer.

    Output is raw hypervector bit indices, not symbol names. Decoding back to symbols
    requires graphs.hypervectors and is not provided by the repo demos.
    """
    weights = tm.get_state()[1].reshape(n_classes, -1)
    for i in range(tm.number_of_clauses):
        parts = []
        for k in range(graphs.hypervector_size * 2):
            if tm.ta_action(0, i, k):
                if k < graphs.hypervector_size:
                    parts.append("x%d" % k)
                else:
                    parts.append("NOT x%d" % (k - graphs.hypervector_size))
        w = " ".join(str(weights[c, i]) for c in range(n_classes))
        print("Clause #%d W:(%s) %s" % (i, w, " AND ".join(parts)))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pytm"
    if which == "pytm":
        smoke_pytm()
    elif which == "graphtm":
        smoke_graphtm()
    else:
        print(__doc__)
