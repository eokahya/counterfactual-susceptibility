# Research Specification v0.1

## 1. Working title

**Counterfactual Susceptibility: Discovering Inactive and Inhibitory Circuits in Language Models**

The title is provisional. We will narrow it if experiments support only feature-level gate-crossing prediction rather than full circuit discovery.

## 2. Problem statement

Sparse attribution graphs are usually built from features active on a baseline prompt. This omits an important causal possibility: an inactive feature may be close to its activation threshold and held below it by one or more active inhibitory features. Suppressing an upstream feature can cause the inactive feature to turn on and alter downstream behavior.

The project asks:

> Can we identify, before running an exhaustive intervention search, which inactive features are likely to activate under a specified counterfactual intervention and which of those activations materially affect model behavior?

## 3. Scope of the first paper

The first paper focuses on a controlled intervention family:

- choose an active upstream feature `j`;
- continuously suppress it by a fraction `alpha in [0, 1]`;
- predict whether an inactive target feature `i` crosses its activation threshold;
- measure the critical suppression strength;
- test whether the newly activated target mediates an output change.

Multi-source interventions are an extension, not a prerequisite for the first validated result.

## 4. Notation and operational definitions

At a fixed prompt, token position, and layer, let target feature `i` have preactivation

\[
z_i
\]

and threshold

\[
\tau_i.
\]

Let its activation function be the exact thresholded nonlinearity used by the loaded transcoder,

\[
a_i = \phi_i(z_i).
\]

The feature is inactive at baseline when `a_i = 0`. Define its positive activation margin

\[
m_i = \tau_i-z_i > 0.
\]

Let `j` be an active upstream feature with activation `a_j`. Let

\[
J_{ij}=\frac{\partial z_i}{\partial a_j}
\]

be the prompt-local response of target preactivation to the source activation under the declared linearization convention.

Suppressing source `j` by a fraction `alpha` gives the first-order prediction

\[
\widehat z_i(\alpha)=z_i-\alpha a_jJ_{ij}.
\]

For an inhibitory source-target relation, `J_ij < 0`, so source suppression raises the target preactivation. Define

\[
q_{j\to i}=-a_jJ_{ij}.
\]

Only `q_{j->i} > 0` can activate the target under source suppression.

### 4.1 Critical intervention

The predicted critical suppression fraction is

\[
\widehat\alpha^{\star}_{j\to i}
=
\frac{m_i}{q_{j\to i}},
\qquad q_{j\to i}>0.
\]

A full ablation is predicted to activate the target when

\[
\widehat\alpha^{\star}_{j\to i}\le 1.
\]

More precisely, `alpha_hat_star < 1` predicts that full suppression moves the
preactivation strictly past the threshold.  The equality case reaches the
threshold exactly, so whether the feature activates there depends on the
loaded transcoder's strict (`>`) or non-strict (`>=`) gate convention and must
be treated as a boundary case.

### 4.2 Pairwise counterfactual susceptibility

Define the dimensionless score

\[
S_{j\to i}
=
\frac{q_{j\to i}}{m_i+\varepsilon}
=
\frac{1}{\widehat\alpha^{\star}_{j\to i}+\varepsilon/q_{j\to i}},
\qquad q_{j\to i}>0.
\]

Thus an equivalent `epsilon_prime` notation is valid only when
`epsilon_prime = epsilon / q_{j->i}`; it is pair-dependent rather than a fixed
global constant.

Interpretation:

- `S > 1`: with positive `epsilon`, full source ablation is predicted to move
  strictly past the target threshold by more than the numerical buffer;
- larger `S`: a smaller source suppression should suffice;
- `S <= 0`: the intervention does not move the target toward activation in the local model.

When `epsilon = 0` and the pair is away from tolerance boundaries, `S > 1`
is equivalent to `alpha_hat_star < 1`.  The case `q = m_i` is a gate-semantic
boundary, not a definite crossing.

### 4.3 Predicted counterfactual activation

For a proposed intervention strength `alpha`, use the exact loaded feature nonlinearity:

\[
\widehat a_i(\alpha)
=
\phi_i\!\left(z_i-\alpha a_jJ_{ij}\right).
\]

#### 4.3.1 Numerical worked example

Let `z_i = 0.20`, `tau_i = 0.50`, `a_j = 2.0`, and `J_ij = -0.25`.
Then

\[
m_i=0.30,
\qquad
q_{j\to i}=0.50,
\qquad
\widehat\alpha^\star_{j\to i}=0.60.
\]

The local prediction is `z_hat_i(alpha) = 0.20 + 0.50 alpha`, so it reaches
the threshold at `alpha = 0.60` and gives `z_hat_i(1) = 0.70` under full
suppression.  With `epsilon = 0.01`, the susceptibility is
`S = 0.50 / 0.31 = 50/31`, approximately `1.612903`.  Whether activation is
nonzero at the exact equality point `alpha = 0.60` follows the loaded
transcoder's gate convention; the full suppression prediction is strictly
above threshold under either convention.

### 4.4 Behavioral salience

Let `T` be a declared scalar behavior metric, such as a target-minus-contrast logit difference. Let

\[
g_i=\frac{\partial T}{\partial a_i}
\]

under the declared local response convention. A first-order behavior-weighted ranking score is

\[
B_{j\to i}(\alpha)
=
\left|g_i\right|
\left|\widehat a_i(\alpha)-a_i\right|.
\]

`S` and `B` answer different questions:

- `S`: how easy is the feature to activate under this intervention?
- `B`: if activated as predicted, how behaviorally consequential might it be?

They must be evaluated separately.

### 4.5 Mediation test

For a verified pair `(j, i)`, compare:

1. baseline;
2. source suppression `do(j down)`;
3. source suppression plus target clamp `do(j down, i = 0)`.

Let the corresponding behavior changes from baseline be `Delta T_j` and `Delta T_{j,i0}`. A signed mediated component is

\[
\Delta T^{\mathrm{med}}_{j\to i}
=
\Delta T_j-\Delta T_{j,i0}.
\]

When the denominator is sufficiently far from zero, report the mediated fraction

\[
\mathrm{MF}_{j\to i}
=
\frac{\Delta T^{\mathrm{med}}_{j\to i}}
{\Delta T_j}.
\]

The paper must report the signed quantities as primary results; ratios are secondary because they can be unstable near zero.

## 5. Central hypotheses

### H1 — Gate-crossing discrimination

The susceptibility score ranks true intervention-induced gate crossings better than:

- random inactive features;
- activation margin alone;
- inhibitory influence alone;
- downstream gradient alone.

### H2 — Critical-strength calibration

The predicted critical intervention `alpha_hat_star` correlates with the critical intervention measured by an actual suppression sweep.

### H3 — Behavioral prediction

Behavior-weighted susceptibility predicts the direction and magnitude of the observed target metric change better than gate proximity or downstream gradient alone.

### H4 — Causal mediation

For high-ranked verified pairs, clamping the counterfactually activated target back to zero removes a nontrivial, correctly signed component of the source-ablation effect.

### H5 — Added explanatory coverage

At a matched node or edge budget, adding verified inactive targets and inhibitory relations explains more observed intervention effects than an active-only graph.

H5 is a full-paper objective, not required for the first pilot.

## 6. Candidate generation

Exhaustively testing every inactive feature is infeasible. Candidate generation proceeds in stages.

### Stage A — Near-threshold filter

At each selected layer and token position, retain the `K_margin` inactive features with the smallest positive margins.

### Stage B — Inhibitory-response filter

For selected active sources, compute or approximate `q_{j->i}` and retain pairs with positive predicted movement toward the threshold.

### Stage C — Behavioral filter

For a declared target metric, estimate `|g_i|` and rank by `B`.

Candidate-generation recall must be measured against a smaller brute-force intervention set. Efficiency cannot be claimed without this audit.

## 7. Ground truth

For each selected source feature:

1. run a suppression sweep over `alpha`;
2. record target feature preactivations and activations;
3. detect gate crossings using the exact transcoder activation convention;
4. estimate the observed critical `alpha_star` by bracketing and binary search where feasible;
5. record the target metric and downstream states;
6. perform mediation tests on a preregistered subset of high-ranked and control pairs.

Ground truth should be recorded both in the replacement-model setting and, where the tooling permits, through interventions whose consequences are measured in the underlying model.

## 8. Baselines

- Random inactive target.
- Margin-only: `1 / (m_i + eps)`.
- Influence-only: `max(0, q_{j->i})`.
- Downstream-only: `|g_i|`.
- Product baseline without threshold nonlinearity: `max(0, q) * |g|`.
- Contrastive-prompt discovery, when a natural contrastive prompt is available.
- Oracle exhaustive sweep on a small subset; this is an upper-bound discovery procedure, not a deployable baseline.

## 9. Metrics

### Gate crossing

- AUPRC and AUROC.
- Precision@k and recall@k per prompt.
- Candidate-generation recall.
- False-crossing rate.

### Critical intervention

- Spearman correlation between predicted and observed `alpha_star`.
- Median absolute error.
- Calibration plots in bins of predicted susceptibility.

### Activation magnitude

- Correlation and error between predicted and observed post-intervention target activation.

### Behavior

- Sign accuracy for `Delta T`.
- Spearman correlation and normalized error.
- Targeted-versus-off-target effect.

### Mediation

- Signed mediated effect.
- Mediated fraction when numerically stable.
- Comparison with matched random target clamps.

### Efficiency

- GPU time, peak memory, and number of actual forward interventions.
- Recall as a function of candidate budget.

## 10. Initial empirical ladder

### E0 — Upstream reproduction and API audit

- Pin the upstream `circuit-tracer` commit.
- Reproduce one official attribution example and one intervention example.
- Verify exact feature activation, preactivation, threshold, and intervention semantics.
- Confirm how to observe inactive feature preactivations.

### E1 — Synthetic unit model

Construct a small thresholded directed feature network with known inhibitory edges. Verify exact score behavior, crossing detection, critical-strength estimation, and mediation metrics.

### E2 — Open-model feasibility scan

Use a small open model with pretrained transcoders. On a modest prompt set:

- inventory active sources and near-threshold inactive targets;
- run single-source suppression sweeps;
- estimate the natural prevalence of crossings;
- identify numerical and API bottlenecks.

### E3 — Predictor benchmark

Evaluate susceptibility and baselines on held-out prompts and interventions.

### E4 — Behavioral and mediation study

Select behaviorally meaningful target metrics and validate top predictions with double interventions.

### E5 — Generalization

Repeat across a second dictionary size and, if feasible, a second model family.

### E6 — Attribution-graph augmentation

Add verified inactive nodes and inhibitory edges to graph exports; evaluate causal coverage at matched graph size.

## 11. Initial model choice

Use Gemma 2 2B with an available cross-layer transcoder for the first full-stack pilot because current open tooling supports attribution and feature intervention at this scale. Start with the smaller dictionary for debugging and repeat key results with the larger dictionary if feasible.

A second model is not selected until E2 exposes the actual compute and API constraints.

## 12. Failure modes and safeguards

- **Local linearization failure:** gate changes, normalization, and attention changes can invalidate `J_ij`. Measure, do not hide, this failure.
- **Replacement-model mismatch:** report replacement and underlying-model intervention effects separately.
- **Feature absorption/splitting:** interpret individual features cautiously; test feature groups where needed.
- **Threshold convention errors:** obtain thresholds and activation semantics from the loaded transcoder, never reimplement from memory.
- **Selection bias:** separate discovery prompts from held-out evaluation prompts.
- **Multiple testing:** preregister primary metrics and use bootstrap confidence intervals over prompts.
- **Large-intervention artifacts:** use continuous sweeps and report the minimum verified intervention, not only full ablation.
- **Unstable mediation ratios:** prioritize signed mediated effects and confidence intervals.

## 13. Go/no-go gates

- Do not begin the large benchmark until E0 verifies exact upstream semantics.
- Do not claim scalable discovery until candidate-generation recall is measured against a brute-force subset.
- Do not claim behavioral importance from gate crossing alone.
- Do not claim a circuit in the underlying model unless the relevant intervention is tested there.
- If single-source crossings are too rare for a useful benchmark, preserve the negative result and move to sparse multi-source suppression as a separately documented extension.

## 14. Reproducibility requirements

Every run must store:

- git commit and dirty-tree status;
- upstream package commit/version;
- model and transcoder identifiers and revisions;
- full configuration;
- random seeds;
- hardware and software metadata;
- prompt identifiers or generation recipe;
- metric tables before plotting.
