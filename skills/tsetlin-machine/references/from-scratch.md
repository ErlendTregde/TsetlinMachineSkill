# Implementing a Tsetlin Machine from scratch

Every rule below was read out of CAIR's own reference implementation,
`cair/TsetlinMachine/MultiClassTsetlinMachine.pyx` [TM-Reference-Impl], cross-checked against
Chapter 1 of the book [TM-Book-Ch1]. Where the two use different words for the same thing, both are
given. Line references are to the `.pyx`.

Read this when writing a TM directly rather than calling a library — see SKILL.md Step 1 for when
that is the right call. For the conceptual picture first, read `concepts.md`.

## 1. The Tsetlin Automaton

One automaton per literal per clause. It is a counter, not a neuron.

- States run `1 .. 2N`, where `N` is `number_of_states` (the "memory depth"). The book draws
  `N = 5`, so positions 1-10 [TM-Book-Ch1].
- **Action:** `state <= N` -> **exclude**; `state > N` -> **include** [TM-Reference-Impl:168-172].
  The book calls these *Forgotten* and *Memorized*.
- **Reward** moves the state deeper into its current action; **penalty** moves it toward the
  boundary and eventually across it. In code this is just `+= 1` / `-= 1`, clamped to `[1, 2N]`.
- **Initialisation:** every automaton starts at `N` or `N+1`, chosen at random — straddling the
  decision boundary [TM-Reference-Impl:71]. The book initialises all literals at position 5, "barely
  Forgotten", and notes the choice does not matter because the machine is self-correcting
  [TM-Book-Ch1].

The clamping matters. An automaton driven to 1 or 2N is *saturated*: further reinforcement in that
direction is a no-op, which is what makes a learned pattern stable rather than oscillating.

## 2. Clause output — and the bug everyone writes

A clause is an AND over its included literals. With `X[k]` the feature and `¬X[k]` its negation:

```
clause_output = 1
all_exclude   = 1
for k in features:
    if include(k) or include_negated(k): all_exclude = 0
    if (include(k) and X[k] == 0) or (include_negated(k) and X[k] == 1):
        clause_output = 0; break
if predict and all_exclude:      # <-- THE IMPORTANT LINE
    clause_output = 0
```

[TM-Reference-Impl:95-115]

**An empty clause — one with every literal excluded — outputs 1 during learning and 0 during
prediction.** Both halves are deliberate:

- During **learning** it must output 1, or a freshly initialised machine could never receive the
  Type Ia feedback that starts memorisation. The book puts it plainly: *"An empty condition does not
  specify any literal requirements so it is always True"* [TM-Book-Ch1].
- During **prediction** it must output 0, or every untrained clause votes for its class on every
  input, and accuracy collapses toward the class prior.

Get this wrong and the model trains fine and predicts badly, with no error anywhere. It is the
single most common from-scratch bug.

## 3. Voting, and the vote sum is clipped

Half the clauses per class are positive polarity, half negative. Sum `+1` per firing positive clause
and `-1` per firing negative clause; that is `v`, the class sum.

**Clip `v` to `[-T, +T]` before using it for feedback** [TM-Reference-Impl:128-131], where `T` is
`threshold`, which the book calls the **Vote Margin** [TM-Book-Ch1]. Classification itself is
`argmax` over the per-class sums.

## 4. Which clauses get feedback

Not all of them. Each clause is selected independently:

```
P(feedback | clause belongs to the target class)   = (T - clip(v, -T, T)) / (2T)
P(feedback | clause belongs to a contrasting class) = (T + clip(v, -T, T)) / (2T)
```

[TM-Reference-Impl:262,273]. The book states the same thing concretely with Vote Margin 2, writing
`(2 - v)/4`, and spells out the three landmarks [TM-Book-Ch1]:

| vote sum `v` | P(update) | meaning |
|---|---|---|
| `-T` or lower | 1.0 | badly wrong — update every clause |
| 0 | 0.5 | undecided |
| `+T` or higher | 0.0 | already correct by the required margin — update nothing |

**This is the whole resource-allocation mechanism.** Once the machine is right by margin `T`,
learning stops for that example, leaving clauses free for patterns not yet covered. It is why `T`
behaves nothing like a learning rate, and why raising `T` lets more clauses stay active per example.

## 5. The two feedback types

### Type I (combats false negatives)

Given to positive-polarity clauses when `y=1`, negative-polarity clauses when `y=0`.

**If the clause output is 0** — the book's *Erase Feedback*, Type Ib — decrement **every** automaton
with probability `1/s`, regardless of literal values [TM-Reference-Impl:293-300]. This coarsens a
clause that failed to recognise an example it should have.

**If the clause output is 1** — the book's *Recognize Feedback*, Type Ia — for each feature
[TM-Reference-Impl:302-320]:

| literal value | automaton for `x_k` | automaton for `¬x_k` |
|---|---|---|
| `X[k] == 1` | increment w.p. `(s-1)/s` | decrement w.p. `1/s` |
| `X[k] == 0` | decrement w.p. `1/s` | increment w.p. `(s-1)/s` |

So the **true** literals are memorised and the **false** ones forgotten. The book's *Memorize Value*
is `(s-1)/s` and its *Forget Value* is `1/s`, which is why it says the two usually sum to 1.0
[TM-Book-Ch1].

`boost_true_positive_feedback=1` replaces the `(s-1)/s` draw with an unconditional increment
[TM-Reference-Impl:306] — the book's "fix the Memorize Value to 1.0".

### Type II (combats false positives)

Given to positive-polarity clauses when `y=0`, negative-polarity clauses when `y=1`. **Only acts
when the clause output is 1** — a clause that fired when it should not have.

For each feature, if the literal is **false** and its automaton is currently **excluding**,
increment it [TM-Reference-Impl:326-336]:

```
if X[k] == 0 and action(state[j,k,0]) == 0: state[j,k,0] += 1
if X[k] == 1 and action(state[j,k,1]) == 0: state[j,k,1] += 1
```

Three things to get right, all easy to miss:

- **There is no probability here.** Type II is deterministic; no `s`, no coin flip. The book is
  explicit: *"this time there is no randomization – the increment is always performed"*
  [TM-Book-Ch1].
- **It only touches automata that are currently excluding.** Already-included literals are left
  alone.
- **It never decrements.** Type II only pushes literals toward inclusion — specifically, a false
  literal whose inclusion would have stopped the clause from firing.

The book calls this *Reject Feedback*, because the clause learns to reject the object
[TM-Book-Ch1].

## 6. Multiclass

The reference implementation does **pairwise** updates, not one-vs-all. Per training example it
picks **one** contrasting class uniformly at random and updates only that class's clauses alongside
the target class's [TM-Reference-Impl:236-238]. Updating every other class on every example is a
different (slower, and differently-behaved) algorithm — do not silently substitute it.

## 7. Vocabulary map

Chapter 1 of the book renames everything for teachability, and its own footnote confirms the
mapping [TM-Book-Ch1]. A user quoting the book and a user quoting the papers are describing the
same algorithm:

| Book (Chapter 1) | Papers / code |
|---|---|
| Memorized / Forgotten | Include / Exclude |
| Memory position (1-10) | TA state (`1..2N`) |
| Memorize Value | `(s-1)/s` |
| Forget Value | `1/s` |
| Recognize Feedback | Type Ia |
| Erase Feedback | Type Ib |
| Reject Feedback | Type II |
| Vote Margin | `T` / `threshold` |
| Rule | Clause |

## 8. Minimal correctness check

XOR is the right smoke test for a hand-written TM, because it is not linearly separable — a TM that
reaches ~100% on XOR has working Type I *and* Type II feedback, while a broken sign or a missing
Type II typically plateaus near 50-75%.

```python
X = rng.integers(0, 2, (500, 8)).astype(np.uint32)
y = (X[:, 0] ^ X[:, 1]).astype(np.uint32)
# fit, then:
assert (tm.predict(X) == y).mean() > 0.95
```

If XOR passes but real data underperforms, the model is fine and the problem is upstream — see
`booleanization.md`.

This document was validated by implementing a TM from it and nothing else: ~55 lines of numpy
following sections 1-6 in order reached 1.000 train and 1.000 held-out accuracy on noisy-free XOR
with 20 clauses, T=15, s=3.9, 40 epochs [verified-2026-09-14]. If you follow it and XOR does not
converge, suspect your code rather than the spec — the three things that most often break it are
the empty-clause rule in section 2, forgetting to clip `v` in section 3, and adding a probability
to Type II in section 5.
