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

## Alternatives to hand-tuning

A scheme for continuous input exists [arXiv:1905.04199], as does adaptive continuous feature
binarization applied to forecasting dengue incidence [SSCI-2020-adaptive], and an adaptive sparse
representation of continuous input based on stochastic searching on the line
[Electronics-10-2107]. These are worth naming to a user who is spending a lot of effort on
thresholds by hand.
