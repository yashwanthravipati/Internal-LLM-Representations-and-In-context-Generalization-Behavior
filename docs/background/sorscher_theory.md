# Sorscher, Ganguli & Sompolinsky (2022): geometric theory of few-shot learning

**Paper:** "Neural representational geometry underlies few-shot concept learning", *PNAS* 119(43): e2200800119.
**Full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9618072 · **Official code:** https://github.com/bsorsch/geometry-fewshot-learning

Equations below were checked against the paper's full text on 2026-10-25. Where a normalization convention is ambiguous, the official code is the tie-breaker, and Monte Carlo agreement (T-023) is the final check.

## What the paper shows

- Few-shot learning of novel visual concepts can be done by a single downstream neuron with a **prototype rule**: average the m training examples of each concept into prototypes x̄_a and x̄_b, then classify a test example by the nearest prototype. Equivalently, w = x̄_a − x̄_b plus a bias.
- Prototype learning on ResNet50 features gets 98.6% average accuracy on 5-shot discrimination of 1,000 novel ImageNet21k concepts.
- An analytical theory predicts the generalization error from four geometric quantities of the **concept manifolds**: the point clouds of representations of a concept's examples.
- The theory matches experiments on synthetic ellipsoids, on DNNs (1,000 × 999 concept pairs), and on **macaque IT** (168 neurons, 64 × 63 object pairs). That last result is why the theory belongs in a neuro + AI course.
- About 200 neurons suffice for near-asymptotic performance. The asymptotic dimension is D∞ ≈ 35 in a trained DNN and ≈ 12 in IT.

## Geometry estimation (paper Methods §K)

For the P examples of one concept, collect the responses in an N × P matrix X (N = number of neurons, here the residual-stream width).

- Centroid: x₀ = mean of the columns of X.
- Covariance: C = (1/P) X Xᵀ − x₀x₀ᵀ = (1/P) Σᵢ Rᵢ² uᵢuᵢᵀ (Eq. 4). So uᵢ are the eigenvectors and **Rᵢ²/P are the eigenvalues**, as stated in the paper.
- Mean squared radius: R² = (1/N) Σᵢ Rᵢ².
- Participation-ratio dimension: D = (Σᵢ Rᵢ²)² / Σᵢ Rᵢ⁴.

> **Normalization (resolved 2026-10-26, T-022).** The official code (`geometry()` in the macaque / prototype-vs-exemplar notebooks, commit fee5d47) takes `R` = singular values of the centered P × N sample matrix and normalizes the signal by **tr(C_a) = Σᵢ Rᵢ²/P** (the total variance; no 1/N factor). So signal = ‖x₀ᵃ − x₀ᵇ‖² / tr(C_a), bias uses tr(C_b)/tr(C_a), and D is the participation ratio of the Rᵢ². Our `sorscher.py` matches a verbatim port of that function to 1e-8 (`tests/test_sorscher.py::test_matches_official_code`).
>
> **Extra term.** The official code's denominator also includes a **subspace–subspace overlap** ss/m, with ss = Σᵢⱼ (uᵢᵃ·uⱼᵇ)² Rᵢᵃ² Rⱼᵇ² / (Σ Rᵃ²)², which the short form of Eq. 1 above omits. The full SI expression (as in the official k-way code) adds 1/(2D_a m²) + (R_b²/R_a²)/(2D_b m²). We implement both (`full=False/True`); `full=True` matches Monte Carlo best at m = 1–2.
>
> **Validation (synthetic Gaussian manifolds, 25 pairs per regime, P = 1000, N = 200):** high D (median D ≈ 72): mean |theory − MC| = 0.005 at m = 1, ≤ 0.002 for m ≥ 2, r = 0.999. Low D (power-law spectrum, median D ≈ 2.5): mean |error| ≈ 0.04–0.06, r ≈ 0.99; the theory **overestimates** error there (rank order preserved). Run: `results/20261026-0111_t022_synthetic_theory-vs-mc/`; figure `figures/theory_vs_prototype_synthetic.png`.

**Efficiency (P ≪ N).** Don't form the N × N covariance. Center X to get X̃, eigendecompose the P × P Gram matrix (1/P) X̃ᵀX̃ = V Λ Vᵀ, and recover uᵢ = X̃vᵢ / √(Pλᵢ). At most P − 1 eigenvalues are nonzero.

## The SNR formula (Eq. 1)

For m-shot prototype learning of the pair (a, b), the error on test examples of concept a is ε_a = H(SNR_a), where H(x) = ∫ₓ^∞ e^(−t²/2)/√(2π) dt (the Gaussian tail; `scipy.stats.norm.sf`). The dominant terms are:

```
SNR_a = ½ [ ‖Δx₀‖² + (R_b² R_a⁻² − 1)/m ]  /  √( D_a⁻¹/m + ‖Δx₀·U_b‖²/m + ‖Δx₀·U_a‖² )
```

The full expression is in SI §3C, and the multi-class extension in SI §3D.

| Term | Definition | Effect |
|---|---|---|
| **Signal** | ‖Δx₀‖² = ‖x₀ᵃ − x₀ᵇ‖² / R_a² | Farther-apart centroids → higher SNR |
| **Bias** | (R_b²/R_a² − 1)/m | Negative when manifold a is larger → asymmetric errors; vanishes as m grows |
| **Dimension** | D_a, the participation ratio | **Higher D helps few-shot learning**: the noise term scales as D⁻¹/m |
| **Signal–noise overlap** | ‖Δx₀·U_a‖², ‖Δx₀·U_b‖², with U_a = [u₁ᵃR₁ᵃ, …, u_Nᵃ R_Nᵃ]/R_a² | Variability along the signal direction hurts; the U_a term doesn't vanish as m → ∞ |

- **As m → ∞**, the bias and the first two noise terms vanish. The SNR saturates at a finite value set by the signal and the overlap ‖Δx₀·U_a‖. This is why prototype learning loses to an optimal linear classifier at large m but is near-optimal at small m.
- **Error asymmetry:** ε_a ≠ ε_b in general (e.g. "coati" is mistaken for "numbat" more often than the reverse). The bias term predicts it. This gives a testable prediction for LLMs, though LLM label bias must be controlled (counterbalancing, calibration).

## Subsampling / random projections (Eq. 2)

When only M neurons (or M random linear combinations of them) are observed:

```
SNR(M) = SNR∞ / (1 + D∞/M),      D⁻¹(M) = D∞⁻¹ + M⁻¹
```

**Checked numerically (2026-10-26, `tests/test_subspaces.py`):** for Gaussian manifolds (D ≈ 97, ambient d = 400) the form SNR∞/(1 + D∞/M) matches random orthonormal projections at **m → ∞ and M ≪ d** (M = 80: measured ratio 0.446 vs predicted 0.453; M = 160: 0.639 vs 0.623). At finite m (m = 5) the measured loss is clearly *smaller* than Eq. 2 (M = 80: 0.574 vs 0.453), and when M approaches d the formula underestimates the ratio. **Consequence for T-043:** use empirical random subspaces of matched M as the null, and report Eq. 2 only as the asymptotic reference.

**Why we care:** Exp 3 compares low-dimensional J-space representations with full hidden states. Eq. 2 gives the expected SNR loss from projecting onto a *random* M-dimensional subspace. That is the null a J-space subspace has to beat (T-043).

## Zero-shot learning from language (Eq. 3)

The paper also shows that concepts can be learned with no visual examples, using a prototype y derived from language (word embeddings mapped into the visual representation space): SNR_zero-shot = ½(‖x₀ᵃ − y_b‖² − ‖x₀ᵃ − y_a‖²) / ‖Δy·U_a‖. This is loosely analogous to our explicit-rule control condition, where the rule is stated in words instead of demonstrated.

## Mapping onto LLM in-context learning

| Sorscher | This project |
|---|---|
| Neurons (IT units / DNN feature layer) | Residual-stream dimensions at layer ℓ |
| Concept (e.g. "coati") | Category (Exp 1, e.g. "fruit") or one attribute value (Exp 2, e.g. "mammal") |
| Examples of a concept (images) | Item words in that category (e.g. apple, pear, plum …) |
| m training examples per concept | m demonstrations per category in the prompt |
| Prototype readout neuron | (Hypothesized) what ICL is doing |
| Test example | Held-out query item |
| Generalization error ε_a | LLM error on held-out queries of category a |

## Where the assumptions may break for LLMs

These are the reasons the question is interesting. Each is a possible explanation if the theory fails.

1. **ICL isn't literally prototype learning.** Attention could implement something exemplar-like (nearest demo), a learned rule, or a task vector (Hendel 2023; Todd 2024). The theory is a *model of the model*. Where it fails is informative.
2. **Representations depend on context.** The theory assumes a fixed representation, but in an LLM the query's representation depends on the demos. That's Exp 2: we compute geometry on both neutral and in-context representations.
3. **Anisotropy and outlier dimensions.** LLM residual streams have a few huge-magnitude dimensions that dominate Euclidean distances and radii. Handling is decided in T-021, with a sensitivity analysis.
4. **Few samples per manifold.** We'll have ~20–60 items per category, vs 500 images in the paper. Participation-ratio estimates are capped at P − 1 and biased downward. Use bootstrap CIs, and check sensitivity to P.
5. **Labels and format.** The LLM must also map the inferred grouping onto arbitrary label tokens. Label/recency biases add error the theory doesn't model, so counterbalance and calibrate.
6. **Which layer?** The theory needs a single representation. We sweep all layers and select on validation data only.

## Implementation checklist (T-022 / T-023)

- [ ] `manifold_stats(X)` → x₀, {Rᵢ, uᵢ}, R², D (Gram trick).
- [ ] `snr(stats_a, stats_b, m)` → SNR_a plus each term separately (signal, bias, dimension, overlap_a, overlap_b); `error = norm.sf(snr)`.
- [ ] Both directions (a vs b, b vs a); m grid; m → ∞ limit.
- [ ] Eq. 2 random-projection prediction.
- [ ] Unit tests on synthetic ellipsoids against Monte Carlo prototype learning (cf. paper Fig. 2E–H).
- [ ] Numerical cross-check against the official repository on a shared example.
