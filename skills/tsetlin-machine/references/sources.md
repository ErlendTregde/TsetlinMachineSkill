# Sources

Every citation tag used in this skill resolves here. Section names in citations (e.g.
`[TM-README §Learning]`) refer to headings in the linked document.

Anything tagged `[heuristic]` in this skill has **no** source. It is a working suggestion to be
tested, and should be presented to the user as such.

## Repositories and documentation

| Tag | Document | URL |
|---|---|---|
| `TM-README` | Tsetlin Machine — code, datasets, and the Basics write-up | https://github.com/cair/TsetlinMachine/blob/master/README.md |
| `PyTM-README` | pyTsetlinMachine — high-level Python API with C extensions | https://github.com/cair/pyTsetlinMachine |
| `TMU-README` | Tsetlin Machine Unified | https://github.com/cair/tmu/blob/main/README.md |
| `GTM-README` | GraphTsetlinMachine | https://github.com/cair/GraphTsetlinMachine |
| `GTM-NoisyXORDemo` | `examples/NoisyXORDemo.py` — verified GraphTM API reference | https://github.com/cair/GraphTsetlinMachine/blob/master/examples/NoisyXORDemo.py |
| `GTM-CIFAR10Demo` | `examples/CIFAR10Demo.py` — verified image-graph construction | https://github.com/cair/GraphTsetlinMachine/blob/master/examples/CIFAR10Demo.py |
| `TM-Book` | *An Introduction to Tsetlin Machines*, Ole-Christoffer Granmo — free chapters | https://tsetlinmachine.org/ |
| `TM-Book-Code` | Notebooks accompanying the book, incl. `Chapter_1.ipynb` (car/plane worked example) | https://github.com/cair/TsetlinMachineBook |
| `TMU-PPExample` | `CIFAR10ColorThermometerScoring.py`, a worked TMU `TMClassifier` script in the CAIR repo for the TMComposites paper [arXiv:2309.04801] | https://github.com/cair/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR10ColorThermometerScoring.py |
| `IEEE-TM-accelerator-2025` | *An All-digital 8.6-nJ/Frame 65-nm Tsetlin Machine Image Classification Accelerator* — cites `tmu/models/classification/coalesced_classifier.py` directly | https://arxiv.org/html/2501.19347 |
| `TextTM-README` | TextUnderstandingTsetlinMachine — hyperparameter notes | https://github.com/cair/TextUnderstandingTsetlinMachine |
| `CAIR` | Centre for Artificial Intelligence Research, University of Agder | https://github.com/cair |

Book chapters currently downloadable: 1 (Your First Tsetlin Machine, with a Jupyter notebook),
2 (Classification), 4 (Convolution), 7 (Confidence, Trustworthiness, and Composites). Listed
without links: 3 (Regression, marked coming soon), 5 (Pattern Weighing and Sharing),
6 (Auto-Encoding), 8 (Hierarchical Representation), 9 (Graphs), 10 (Natural Language and Image
Processing) [TM-Book].

## Papers

| Tag | Paper | URL |
|---|---|---|
| `arXiv:1804.01508` | Granmo, *The Tsetlin Machine — A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic* | https://arxiv.org/abs/1804.01508 |
| `arXiv:1905.09688` | Granmo et al., *The Convolutional Tsetlin Machine* | https://arxiv.org/abs/1905.09688 |
| `arXiv:1905.04199` | Abeyrathna et al., scheme for continuous input to the Tsetlin Machine | https://arxiv.org/abs/1905.04199 |
| `arXiv:1909.07310` | Gorji et al., *A Tsetlin Machine with Multigranular Clauses* | https://arxiv.org/abs/1909.07310 |
| `arXiv:1911.12607` | Phoulady et al., *The Weighted Tsetlin Machine* | https://arxiv.org/abs/1911.12607 |
| `arXiv:2105.14506` | Sharma et al., *Human Interpretable AI: Enhancing Tsetlin Machine Stochasticity with Drop Clause* | https://arxiv.org/abs/2105.14506 |
| `arXiv:2108.07594` | Glimsdal & Granmo, *Coalesced Multi-Output Tsetlin Machines with Clause Sharing* | https://arxiv.org/abs/2108.07594 |
| `arXiv:2301.00709` | Tsetlin Machine autoencoder | https://arxiv.org/abs/2301.00709 |
| `arXiv:2301.08190` | Literal budget | https://arxiv.org/abs/2301.08190 |
| `arXiv:2309.04801` | TMComposites — plug-and-play collaboration between specialized Tsetlin Machines | https://arxiv.org/abs/2309.04801 |
| `arXiv:2309.06315` | Type III feedback | https://arxiv.org/abs/2309.06315 |
| `arXiv:2310.11481` | Sparse computation with absorbing actions | https://arxiv.org/abs/2310.11481 |
| `arXiv:2406.00704` | *An Optimized Toolbox for Advanced Image Processing with Tsetlin Machine Composites* | https://arxiv.org/abs/2406.00704 |
| `arXiv:2407.09162` | *Exploring State Space and Reasoning by Elimination in Tsetlin Machines* | https://arxiv.org/abs/2407.09162 |
| `arXiv:2507.14874` | Granmo et al., *The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs* — full paper, including Appendix A.4 hyperparameter tables | https://arxiv.org/abs/2507.14874 |
| `arXiv:2508.08350` | Hnilov, *Fuzzy-Pattern Tsetlin Machine* | https://arxiv.org/abs/2508.08350 |
| `RSTA-2019-0165` | Abeyrathna et al., *The Regression Tsetlin Machine* | https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165 |
| `IEEE-9316190` | Abeyrathna et al., integer-weighted clauses | https://ieeexplore.ieee.org/document/9316190 |
| `IEEE-9923859` | Focused negative sampling | https://ieeexplore.ieee.org/document/9923859 |
| `Springer-s10844-021-00682-5` | Saha et al., Relational Tsetlin Machine | https://link.springer.com/article/10.1007/s10844-021-00682-5 |
| `Electronics-10-2107` | Abeyrathna et al., adaptive sparse representation of continuous input | https://doi.org/10.3390/electronics10172107 |
| `SSCI-2020-adaptive` | Abeyrathna et al., adaptive continuous feature binarization (dengue forecasting), SSCI 2020 | listed in [TM-README §Tsetlin Machine Papers] |

The `TM-README` and `PyTM-README` files both carry long BibTeX bibliographies of the wider TM
literature — use them as the canonical citation source rather than reconstructing references from
memory.

## Community

- International Symposium on the Tsetlin Machine (ISTM): https://istm.no —
  proceedings for 2022 and 2023 are on IEEE Xplore [TM-README §Conferences]
- Curated research list: https://github.com/cair/awesome-tsetlin-machine
- CAIR Claude Code marketplace (a possible home for this skill):
  https://github.com/cair/claude-marketplace

## Licensing note

`cair/TsetlinMachine`, `cair/pyTsetlinMachine`, `cair/tmu` and `cair/GraphTsetlinMachine` are MIT
licensed, copyright Ole-Christoffer Granmo and the University of Agder
[TM-README §Licence, PyTM-README §Licence, GTM-README §Licence].
