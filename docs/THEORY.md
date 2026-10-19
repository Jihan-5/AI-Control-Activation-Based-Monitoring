# Restricted Theoretical Model

> Status: the identities below are elementary linear algebra and standard results; they are **not**
> claimed as novel theorems. The research target (RP p.8) is a finite-sample characterisation of
> nuisance-subspace estimation, its overlap with failure directions, and detection power under stated
> assumptions. Code: `src/irmon/theory/restricted_model.py`, `scripts/run_theory_sim.py`.

## 1. Model

For a root with observable context *c*, a response vector in ℝᵈ (one layer, one edit) is

$$ r = \mu(c) + B z + \delta\,\mathbb{1}[\text{failure}] + \varepsilon, $$

with $B \in \mathbb{R}^{d\times k}$ orthonormal (the nuisance subspace excited by T edits),
$z \sim \mathcal N(0, \tau^2 I_k)$, $\varepsilon \sim \mathcal N(0, \sigma^2 I_d)$, and a failure shift
$\delta$. Define the overlap $\rho = \lVert B^\top \delta\rVert / \lVert \delta \rVert \in [0, 1]$.

## 2. Known-subspace detection power

Let $P_\perp = I - BB^\top$ and $\tilde r = r - \mu(c)$.

* **Projected detector** $s_\perp = \lVert P_\perp \tilde r\rVert^2/\sigma^2 \sim \chi^2_{d-k}(\lambda_\perp)$ with
  $\lambda_\perp = \lVert P_\perp\delta\rVert^2/\sigma^2 = (1-\rho^2)\lVert\delta\rVert^2/\sigma^2$.
* **Unprojected Mahalanobis** $s = \tilde r^\top \Sigma^{-1}\tilde r$ with $\Sigma = \tau^2 BB^\top + \sigma^2 I$:
  $s \sim \chi^2_d(\lambda)$, $\lambda = \delta^\top\Sigma^{-1}\delta = \lVert\delta\rVert^2\left(\frac{1-\rho^2}{\sigma^2} + \frac{\rho^2}{\tau^2+\sigma^2}\right)$.
* Power at level α: $1 - F_{\chi^2_{k'}(\lambda)}\big(q_{1-\alpha}(\chi^2_{k'})\big)$.

Consequences (all standard): projection removes B's contribution but erases any failure component
inside span(B) ($\rho \to 1 \Rightarrow \lambda_\perp \to 0$); with a correctly specified Gaussian
Mahalanobis the nuisance directions are already down-weighted, so projection helps mainly through
fewer degrees of freedom and robustness to *misspecified* nuisance scale (e.g., benign shift that
inflates $\tau$ at test time).

## 3. Finite-sample subspace estimation

$\hat B$ = top-k eigenvectors of the empirical covariance of *n* T-responses. With eigengap
$g = \tau^2$ between the nuisance and noise eigenvalues, a Davis–Kahan-type bound gives
$\lVert \sin\Theta(\hat B, B)\rVert \lesssim \lVert \hat\Sigma - \Sigma\rVert / g$, and
$\lVert\hat\Sigma-\Sigma\rVert = O_p\big((\tau^2+\sigma^2)\sqrt{d/n}\big)$. Estimated-projection power
degrades from the oracle as $\sin\Theta$ grows: residual nuisance leaks into $P_\perp$ and part of
$\delta$ is projected away or retained depending on the angle.

## 4. Questions the simulation answers (T1)

1. Power vs overlap ρ for {no projection, oracle projection, estimated projection} at fixed n.
2. Subspace error and power vs n (T-response sample size), at fixed ρ.
3. Sensitivity to nuisance-rank misspecification (k̂ ≠ k) and to test-time nuisance inflation
   (benign shift: τ_test > τ_fit).

## 5. What would be a contribution

A finite-sample power statement for the *estimated* projection under benign shift that predicts when
nuisance removal helps or hurts, validated on the real response geometry (E8). If the theory only
restates §2, it is presented as background, not as the paper's contribution (RP p.8).
