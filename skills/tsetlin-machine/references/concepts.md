# How a Tsetlin Machine works

Everything here is sourced. Tags resolve in `sources.md`. The canonical write-up is the `Basics`
section of the `cair/TsetlinMachine` README [TM-README], which condenses the original paper
[arXiv:1804.01508].

## Contents

1. Classification
2. Learning: Type I and Type II feedback
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
and negative clauses for y=0 [TM-README §Classification].

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

Each clause is composed by a **team of Tsetlin Automata**, one per literal, each deciding to
*Include* or *Exclude* that literal in the clause [TM-README §Learning]. Learning means these
automata moving between Include and Exclude states based on reinforcement. There is no gradient.

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
sum `v` approaches a user-set target `T` for y=1, or −T for y=0 [TM-README §Learning].

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
