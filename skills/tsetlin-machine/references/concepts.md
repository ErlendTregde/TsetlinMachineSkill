# How a Tsetlin Machine works

Everything here is sourced. Tags resolve in `sources.md`. The canonical write-up is the `Basics`
section of the `cair/TsetlinMachine` README [TM-README], which condenses the original paper
[arXiv:1804.01508].

## Contents

1. Classification
2. Learning: the Tsetlin Automaton, Type I and Type II feedback
3. Resource allocation and T
4. What the learning curve looks like
5. The architecture family
6. What does *not* transfer from neural networks

---

## 1. Classification

**Input.** A basic TM takes a vector of Boolean features and classifies it into one of two classes,
y=0 or y=1 [TM-README §Classification].

**Literals.** Each feature is paired with its negation. Features plus negations form the literal
set [TM-README §Classification]. This is why feature count and automata count are linked: a clause
reasons over twice the number of input features.

**Clauses.** A pattern is a conjunctive clause — an AND over a subset of the literals. For example
a clause made of `x1` and `¬x2` outputs 1 exactly when x1=1 and x2=0 [TM-README §Classification].

**Polarity.** The number of clauses is a user-set parameter. Half the clauses are assigned positive
polarity and half negative [TM-README §Classification].

**Decision.** Clause outputs are summed — positives minus negatives — and thresholded with the unit
step function. Classification is therefore a majority vote, with positive clauses voting for y=1
and negative clauses for y=0 [TM-README §Classification]. For more than two classes there is one
such vote sum per class and the prediction is the `argmax`; note that the reference implementation
*trains* pairwise, picking one contrasting class at random per example rather than updating all of
them [TM-Reference-Impl:236-238].

**Worked example (XOR).** The classifier `x1·¬x2 + ¬x1·x2 − x1·x2 − ¬x1·¬x2` captures the XOR
relation [TM-README §Classification]. This is the single most useful *formula-level* example to
show a newcomer, because it shows the whole model as readable propositional logic.

**Worked example (plain-language, for a first-time explanation).** The book's own Chapter 1
notebook uses a more approachable setup than XOR: classifying `Car` vs (implicitly) other vehicle
types from five Boolean features — *Four Wheels*, *Transports People*, *Wings*, *Yellow*, *Blue*.
A car in the dataset is `{'Four Wheels': True, 'Transports People': True, 'Wings': False,
'Yellow': False, 'Blue': True}` [TM-Book-Code, `Chapter_1.ipynb`]. This is a better first example
for someone who has never seen a TM, since the clauses that fall out (something like "Four Wheels
AND Transports People AND NOT Wings") read as ordinary English rather than as algebra. Prefer this
framing over XOR when a user asks "explain this to me" rather than "show me the math."

## 2. Learning

### What a Tsetlin Automaton actually is

A Tsetlin Automaton is a **finite-state counter** — not a neuron, not a weight.

It has `2N` states. States `1..N` mean **Exclude**, states `N+1..2N` mean **Include**
[TM-Reference-Impl:168-172]. Reward pushes it deeper into its current half; penalty pushes it toward
the boundary and eventually across. That is the entire mechanism — a literal enters or leaves a
clause by a counter crossing the midpoint.

Two consequences worth understanding before writing any code:

- **Depth is confidence.** An automaton at state `2N` and one at `N+1` both say *Include*, but the
  first needs `N` penalties to change its mind and the second needs one. This is why `number_of_state_bits`
  is a real hyperparameter: it sets how much evidence a decision accumulates before it becomes hard
  to overturn. Chapter 2 of the book analyses single-literal learning as a stochastic process and
  shows the learning outcome gets more accurate as memory depth increases [TM-Book-Ch2].
- **Saturation is a feature.** States clamp at 1 and `2N`, so a firmly-learned literal stops
  responding to further reinforcement rather than drifting.

Each clause has one automaton **per literal** — so `2 x n_features` of them, since every feature is
paired with its negation. Automata are initialised straddling the boundary, at `N` or `N+1`
[TM-Reference-Impl:71]; the book notes the choice does not matter, because the machine is
self-correcting in a way neural-network initialisation is not [TM-Book-Ch1].

Learning means these automata moving between Include and Exclude states based on reinforcement.
There is no gradient, no loss surface, and nothing to differentiate.

The exact feedback tables and the rest of the implementable algorithm are in
`from-scratch.md`.

A TM learns **online**, one training example at a time [TM-README §Learning].

Two feedback types:

- **Type I feedback produces frequent patterns.**
- **Type II feedback increases the discrimination power of the patterns.** [TM-README §Learning]

### Type I

Given stochastically to positive-polarity clauses when y=1, and to negative-polarity clauses when
y=0 [TM-README §Learning]. Two rules, applied independently per automaton:

1. *Include* rewarded and *Exclude* penalised with probability **(s−1)/s**, whenever the clause
   output is 1 **and** the literal is 1. This reinforcement is strong, and makes the clause
   remember and refine the pattern it recognises [TM-README §Learning].
2. *Include* penalised and *Exclude* rewarded with probability **1/s**, if the clause output is 0
   **or** the literal is 0. This reinforcement is weak, and coarsens infrequent patterns to make
   them frequent [TM-README §Learning].

`s` is the hyperparameter controlling the frequency of the patterns produced [TM-README §Learning].
Read the two probabilities together and the direction is obvious: raising `s` pushes (s−1)/s toward
1 and 1/s toward 0, so literals get included more readily and clauses get more specific.

### Type II

Given stochastically to positive-polarity clauses when y=0, and to negative-polarity clauses when
y=1 — the mirror of Type I [TM-README §Learning]. It penalises *Exclude* whenever the clause output
is 1 **and** the literal is 0. This feedback is strong and produces candidate literals for
discriminating between y=0 and y=1 [TM-README §Learning].

Intuition worth passing to a user: Type I is "you were right, sharpen this", Type II is "you fired
when you shouldn't have, find a literal that would have stopped you".

## 3. Resource allocation and T

For any input, the probability of reinforcing a clause gradually drops to zero as the clause vote
sum `v` approaches a user-set target `T` for y=1, or −T for y=0 [TM-README §Learning]. Concretely,
`v` is first clipped to `[-T, T]` and a clause of the target class is then given feedback with
probability `(T - v) / 2T` [TM-Reference-Impl:128-131,262]. The book calls `T` the **Vote Margin**
and works the same formula through with `T = 2` [TM-Book-Ch1].

An unreinforced clause gives no feedback to its automata, so they are left unchanged. In the
extreme, when `v` meets or exceeds `T` — the machine has successfully recognised the input — no
clauses are reinforced at all, leaving them free to learn new patterns and naturally balancing the
pattern representation resources [TM-README §Learning].

This is the mechanism that stops all clauses collapsing onto the same easy pattern. It is also why
`T` behaves nothing like a learning rate.

Empirically, `T` and `s` pull in opposite directions on which literals end up included
[arXiv:2407.09162].

## 4. What the learning curve looks like

On a binarized but otherwise unenhanced MNIST, averaged over 50 runs, both test and training
accuracy increase almost monotonically across epochs. Even as training accuracy approaches 99.9%,
test accuracy keeps rising, reaching 98.2% after 400 epochs. This differs from backpropagation on a
neural network, where test accuracy starts to drop at some point due to overfitting without proper
regularization [TM-README §Learning Behaviour].

Practical consequence: do not early-stop a TM on the reflex that a widening train/test gap means
overfitting. Train it longer.

## 5. The architecture family

Beyond the vanilla TM, the family includes the Convolutional TM [arXiv:1905.09688], the Regression
TM [RSTA-2019-0165], the Weighted TM [IEEE-9316190], the Coalesced multi-output TM
[arXiv:2108.07594] and the Graph TM [GTM-README], among others [TM-README §Other Architectures,
TMU-README §Features].

The Graph TM is the most general of these in input terms: it processes directed and labeled
multigraphs with vector-symbolic node properties and edge types, supports nested (deep) clauses and
arbitrarily sized inputs, and incorporates the Vanilla, Multiclass, Convolutional and Coalesced TMs
as cases [GTM-README §Features].

There is a dedicated conference, the International Symposium on the Tsetlin Machine
[TM-README §Conferences], and a hardware effort [TM-README §Hardware] — useful context when a user
asks whether this is a serious research line.

## 6. What does not transfer from neural networks

Sourced points of difference:

- Learning is reinforcement over automata states, not gradient descent [TM-README §Learning].
- The model is a set of propositional formulas, readable as logic
  [TM-README §intro, §Classification].
- Test accuracy does not degrade the way backprop's does without regularization
  [TM-README §Learning Behaviour].
- Input must be Boolean [TM-README §Classification].

One correspondence that *is* sourced: the clause count roughly translates to the number of hidden
nodes in a neural network layer [TextTM-README].

Everything else — learning rate schedules, batch sizes, weight initialisation, dropout as usually
understood — has no direct counterpart, and inventing one is how wrong code gets written.
(A "drop clause" mechanism does exist, but it is its own technique with its own paper
[arXiv:2105.14506], not a port of dropout. [heuristic])
