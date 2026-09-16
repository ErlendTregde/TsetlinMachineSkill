# Booleanization

A basic TM takes a vector of **Boolean** features [TM-README §Classification]. Everything real-valued
has to be converted first, and this conversion decides how much signal the TM can possibly see.

The Graph TM paper is blunt about why this matters historically: the Boolean representation of input
data hindered the TM's widespread application, and hypervectors were introduced to expand the range
of data a TM can process [arXiv:2507.14874].

## Methods that appear in the source material

**1. Fixed threshold.** The MNIST demos threshold raw pixel values, keeping pixels above 75
[PyTM-README §MNIST Demo]. Simple, and adequate for high-contrast data.

**2. Adaptive thresholding.** The Fashion-MNIST demo uses OpenCV adaptive Gaussian thresholding
rather than a fixed cut [PyTM-README §Fashion MNIST Demo]. The CIFAR-10 composites study also used
adaptive Gaussian thresholding among its image-processing techniques [arXiv:2406.00704].

**3. Binarizer with a bit budget.** `pyTsetlinMachine.tools.Binarizer(max_bits_per_feature=10)`,
then `fit` and `transform`, used for the breast cancer and California housing demos
[PyTM-README §Continuous Input Demo, §Regression Demo].

**4. Thermometer encoding.** The CIFAR-10 study's best-performing specialist was a 5x5 colour
thermometer trained with 64,000 clauses, reaching 75.4% [arXiv:2406.00704]. Thermometer codes are a
standard way to preserve ordering across bits.

**5. Edge-detection and concatenation.** One study thresholded convolutional edge-detection feature
maps (3x3, 5x5, 7x7 kernels) by pixel histogram and concatenated them into a single Boolean vector
[arXiv:2508.08350].

**6. Fixed-width numeric encoding.** The Multigranular TM paper used five bits per real number on
Iris — three for the integer part and two for the fractional part [arXiv:1909.07310].

**7. N-gram bit positions for text.** The IMDb demo builds a vocabulary of 1- and 2-grams with a
minimum frequency of 10, assigns each a bit position, then selects the top 5000 features by chi²
[PyTM-README §IMDb Demo].

**8. Hypervector symbols (GraphTM).** Rather than a flat Boolean vector, GraphTM assigns symbols to
sparse distributed hypervector codes automatically at initialisation
[GTM-README §Initialization]. Images become sets of white-pixel symbols `W(x,y)`, optionally with
row `R(y)` and column `C(x)` symbols to encode location [GTM-README §Vanilla MNIST,
§Convolutional MNIST].

## Low-cardinality columns: a real but small effect, and a cautionary tale

Quantile cuts on a small-integer column collapse. On the 500-row readmission dataset,
`n_prior_admissions` (values 0-5) gives:

```
np.percentile(n_prior_admissions, [20, 40, 60, 80])  ->  [0, 1, 1, 2]
```

`1` appears twice, so one of the four bits is pure waste — it is identical to its neighbour and
carries no information. Passing cuts through `np.unique` costs nothing and fixes that.

**What this does NOT do is make thresholds inexpressible, and it is worth being precise about why,
because the opposite is easy to believe.** Thermometer bits use `x > cut`, so a cut at 2 yields the
literal `n_prior_admissions > 2`, which is exactly `>= 3`. Measured directly: under naive quantile
cuts, deduplicated cuts, and one-cut-per-distinct-value alike, a single bit equals
`(n_prior_admissions >= 3)`. The decisive threshold survives all three encodings
[verified-2026-09-14].

And the accuracy effect is negligible. Same model, same seed, three train/test splits:

| cuts | Boolean features | test ROC-AUC per split | mean |
|---|---|---|---|
| naive quantiles | 120 | 0.730, 0.694, 0.781 | **0.735** |
| one cut per distinct value (<=12) | 125 | 0.735, 0.759, 0.698 | **0.731** |

The split-to-split spread (0.694-0.781) is an order of magnitude larger than the difference between
encodings [verified-2026-09-14]. Use `np.unique` on your cuts because duplicate bits are free to
remove, not because the threshold would otherwise be lost.

One caveat on that table, in the interest of not overcorrecting: it used 4 cut points per feature.
A separate run using 10 cut points reported that plain quantiles cost it about 3 accuracy points on
the same data, but did not keep the comparison, so the two results cannot be reconciled here. The
plausible mechanism is that asking for 10 quantiles of a 6-value column yields mostly duplicates and
wastes most of that feature's bit budget, which matters more as the budget grows. Treat the size of
this effect as unsettled and configuration-dependent; what is settled is that the threshold remains
expressible either way. Measure it on your own data if it matters to you. [heuristic]

### The cautionary tale is the real lesson

This section originally claimed the opposite — that the missing cut at 3 made the most predictive
fact "not expressible by any clause, no matter how much tuning." That came from a run which saw
ROC-AUC jump 0.649 -> 0.938 after changing the encoder and attributed the jump to the encoding.
The attribution was wrong: a later run reproduced the same low starting AUC and traced it to
selecting the best-epoch validation score but reporting the final epoch. TM vote sums genuinely
drift across epochs, so that gap is large and looks exactly like an encoding ceiling.

Two habits follow, and they matter more than the cut lists:

- **A plausible mechanism plus a big number is not a cause.** Change one thing at a time and
  measure. An encoder change that coincides with a rewrite of the training loop has told you
  nothing about the encoder.
- **Before blaming the encoding, check that a single literal can express the rule you think is
  missing.** It is three lines of numpy, and it would have prevented this entire detour.

The underlying principle is still sound — booleanization sets a hard ceiling on what a clause can
possibly combine, so a TM far below a tree or logistic regression is worth checking there. Just
verify the specific ceiling before asserting it. [heuristic]

## Cost

Every Boolean feature is paired with its negation to form the literal set
[TM-README §Classification], and each clause carries a Tsetlin Automaton per literal
[TM-README §Learning]. So 30 continuous features at 10 bits each is 300 Boolean features and 600
literals per clause. Multiply by the clause count to see where training time goes. [heuristic]

This is why `max_bits_per_feature` is a real modelling decision and not a formality, and why the
first thing to try when a TM is slow is fewer bits per feature. [heuristic]

## What the assistant should always do

1. State which method was chosen and why.
2. Print the feature count before and after.
3. Print one example row before and after.
4. If the user's data is categorical, one-hot rather than ordinal-encode it — an integer category
   code thresholded into bits invents an ordering that does not exist. [heuristic]
5. Sanity-check that the Boolean matrix is not nearly all zeros or all ones. A threshold that keeps
   almost nothing gives the TM nothing to include. [heuristic]
6. Check the cut list of any low-cardinality or integer column before training — see the trap above.
   `scripts/booleanize.py` in this skill does this for you and prints what it did.
7. If the TM lands far below a quick random forest or logistic regression on the same split, check
   the encoding AND the epoch/selection logic. Verify which one by changing one at a time -- a TM's
   test score can swing 0.05+ AUC between epochs, which mimics an encoding ceiling. [heuristic]

## Alternatives to hand-tuning

A scheme for continuous input exists [arXiv:1905.04199], as does adaptive continuous feature
binarization applied to forecasting dengue incidence [SSCI-2020-adaptive], and an adaptive sparse
representation of continuous input based on stochastic searching on the line
[Electronics-10-2107]. These are worth naming to a user who is spending a lot of effort on
thresholds by hand.
