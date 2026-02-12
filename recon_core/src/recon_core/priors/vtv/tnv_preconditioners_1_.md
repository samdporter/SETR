# Smoothed TNV Preconditioners: Complete Derivation

## Citations for imported results

- **First-order inequality for concave functions** (tangent upper bound): Boyd & Vandenberghe, *Convex Optimization* ([seas.ucla.edu](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf)).
- **Hermitian dilation eigenvalues are $\pm\sigma_i$** and linearity of the dilation map: Tropp, *Matrix Analysis*, Caltech ACM 204 ([tropp.caltech.edu](https://tropp.caltech.edu/notes/Tro22-Matrix-Analysis-LN.pdf)).
- **Lewis–Sendov Hessian formula for spectral functions**: Lewis & Sendov, *Twice Differentiable Spectral Functions* ([people.orie.cornell.edu](https://people.orie.cornell.edu/aslewis/publications/01-twice.pdf)).
- **Gershgorin theorem / diagonal dominance**: Alexander, *Gershgorin Circles* ([aalexan3.math.ncsu.edu](https://aalexan3.math.ncsu.edu/articles/gershgorin-report.pdf)).

---

## A. Definitions from first principles

### A.1 Variables and discrete gradients

Let the two images be

$$x^{(1)} \in \mathbb{R}^N, \qquad x^{(2)} \in \mathbb{R}^N.$$

Let the spatial dimension be $d \in \{2, 3\}$. Fix a discrete gradient operator $G : \mathbb{R}^N \to \mathbb{R}^{dN}$, formed by stacking $d$ directional difference operators $G_1, \dots, G_d$, where $(G_p x)_i$ is the finite difference of voxel $i$ in direction $p$.

At voxel $i$, define the gradient vectors

$$g_i^{(m)} \in \mathbb{R}^d, \qquad g^{(m)} = G x^{(m)}, \qquad m \in \{1, 2\}.$$

So $g_i^{(m)}$ is the length-$d$ vector of directional derivatives of modality $m$ at voxel $i$.

### A.2 Local Jacobian / multi-channel gradient matrix

At each voxel $i$, define the $d \times 2$ matrix

$$J_i := \begin{bmatrix} g_i^{(1)} & g_i^{(2)} \end{bmatrix}.$$

Let its singular values be $\sigma_{i,1} \ge \sigma_{i,2} \ge 0$. Since there are only 2 modalities, there are at most 2 nonzero singular values.

Define its $2 \times 2$ Gram matrix

$$S_i := J_i^\top J_i \in \mathbb{S}_+^2.$$

Then the eigenvalues of $S_i$ are exactly $\lambda_{i,1} = \sigma_{i,1}^2$ and $\lambda_{i,2} = \sigma_{i,2}^2$.

### A.3 Smoothed TNV (Charbonnier on singular values)

Smooth each singular value with the Charbonnier function:

$$\phi(\sigma) := \sqrt{\sigma^2 + \varepsilon^2}, \qquad \varepsilon > 0.$$

The local TNV density is

$$\rho(J_i) := \phi(\sigma_{i,1}) + \phi(\sigma_{i,2}).$$

The global TNV regulariser is

$$R(x^{(1)}, x^{(2)}) := \sum_{i=1}^{N} \rho(J_i).$$

Everything below concerns approximating/majorising the curvature of $R$ and converting that into a practical preconditioner.

---

## B. MM/IRLS preconditioners (with closed form for 2×2 Gram)

Two MM/IRLS variants exist:

1. **MM block**: retains modality coupling (a $2 \times 2$ block per voxel/edge).
2. **MM diagonal**: drops modality coupling (two scalars per voxel/edge).

### B.1 The key inequality (no steps skipped)

Define the scalar function of a scalar variable $t \ge 0$:

$$g(t) := \sqrt{t + \varepsilon^2}.$$

Its derivatives are:

$$g'(t) = \frac{1}{2\sqrt{t + \varepsilon^2}}, \qquad g''(t) = -\frac{1}{4}(t + \varepsilon^2)^{-3/2} < 0.$$

So $g$ is **concave** on $[0, \infty)$.

A standard characterisation of concavity for differentiable functions is the **global first-order upper bound**: for all $t, t_0$ in the domain,

$$g(t) \le g(t_0) + g'(t_0)(t - t_0).$$

This is exactly the concave version of the first-order condition given in Boyd & Vandenberghe, *Convex Optimization* ([seas.ucla.edu](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf)).

Apply this to $t = \sigma^2$ and $t_0 = \bar{\sigma}^2$ (where $\bar{\sigma}$ is a lag/previous singular value — this is the MM step):

$$\sqrt{\sigma^2 + \varepsilon^2} \le \sqrt{\bar{\sigma}^2 + \varepsilon^2} + \frac{1}{2\sqrt{\bar{\sigma}^2 + \varepsilon^2}} \left(\sigma^2 - \bar{\sigma}^2\right).$$

Rearrange into "constant + weight × $\sigma^2$":

$$\sqrt{\sigma^2 + \varepsilon^2} \le \underbrace{\left(\sqrt{\bar{\sigma}^2 + \varepsilon^2} - \frac{\bar{\sigma}^2}{2\sqrt{\bar{\sigma}^2 + \varepsilon^2}}\right)}_{\text{constant in } \sigma} + \underbrace{\frac{1}{2\sqrt{\bar{\sigma}^2 + \varepsilon^2}}}_{=: w(\bar{\sigma})} \cdot \sigma^2.$$

Define the IRLS/MM weight:

$$w(\bar{\sigma}) := \frac{1}{2\sqrt{\bar{\sigma}^2 + \varepsilon^2}}.$$

### B.2 Apply to the TNV sum (two singular values)

At voxel $i$, apply the inequality to each $\sigma_{i,k}$ with lag singular values $\bar{\sigma}_{i,k}$, for $k = 1, 2$:

$$\rho(J_i) = \sum_{k=1}^{2} \sqrt{\sigma_{i,k}^2 + \varepsilon^2} \le c_i + \sum_{k=1}^{2} w_{i,k} \, \sigma_{i,k}^2$$

where

$$w_{i,k} := \frac{1}{2\sqrt{\bar{\sigma}_{i,k}^2 + \varepsilon^2}}, \qquad c_i := \sum_{k=1}^{2} \left(\sqrt{\bar{\sigma}_{i,k}^2 + \varepsilon^2} - \frac{\bar{\sigma}_{i,k}^2}{2\sqrt{\bar{\sigma}_{i,k}^2 + \varepsilon^2}}\right).$$

(The constant $c_i$ is never needed for preconditioning; it drops from gradients/Hessians.)

So up to a constant, the MM surrogate is a **weighted sum of squared singular values**.

### B.3 Turn $\sum_k w_k \sigma_k^2$ into a quadratic form in $J$

Let $\bar{J}_i$ have SVD

$$\bar{J}_i = U_i \bar{\Sigma}_i V_i^\top, \qquad \bar{\Sigma}_i = \operatorname{diag}(\bar{\sigma}_{i,1}, \bar{\sigma}_{i,2}).$$

Define the $2 \times 2$ weight matrix in modality space:

$$W_i := V_i \, \operatorname{diag}(w_{i,1}, w_{i,2}) \, V_i^\top.$$

Now define the surrogate quadratic term:

$$Q_i(J_i) := \operatorname{tr}(J_i \, W_i \, J_i^\top).$$

**Claim**: $Q_i(J_i) \ge \sum_{k=1}^{2} w_{i,k} \, \sigma_{i,k}^2$, with equality when $J_i = \bar{J}_i$.

**Proof**: Write $S_i = J_i^\top J_i$ (current Gram matrix). Then

$$Q_i(J_i) = \operatorname{tr}(J_i W_i J_i^\top) = \operatorname{tr}(W_i J_i^\top J_i) = \operatorname{tr}(W_i S_i).$$

The matrix $W_i$ has eigenvalues $w_{i,1} \le w_{i,2}$ (since $\bar{\sigma}_{i,1} \ge \bar{\sigma}_{i,2}$ implies $w_{i,1} \le w_{i,2}$), and $S_i$ has eigenvalues $\sigma_{i,1}^2 \ge \sigma_{i,2}^2$. By the **trace rearrangement inequality** (von Neumann's trace inequality for commuting eigenvalue orderings: if $A, B \in \mathbb{S}^n_+$ with eigenvalues $\alpha_1 \ge \dots \ge \alpha_n$ and $\beta_1 \le \dots \le \beta_n$ respectively, then $\operatorname{tr}(AB) \ge \sum_k \alpha_k \beta_k$):

$$\operatorname{tr}(W_i S_i) \ge w_{i,1} \sigma_{i,1}^2 + w_{i,2} \sigma_{i,2}^2 = \sum_{k=1}^{2} w_{i,k} \sigma_{i,k}^2.$$

At the lag point $J_i = \bar{J}_i$, the matrices $W_i$ and $S_i = \bar{S}_i$ share the eigenbasis $V_i$, so the inequality holds with equality.

Therefore, combining with Section B.2:

$$\rho(J_i) \le c_i + \sum_{k=1}^{2} w_{i,k} \sigma_{i,k}^2 \le c_i + Q_i(J_i).$$

The quadratic surrogate $Q_i$ is a valid majoriser (composed of two inequalities: the scalar MM concavity bound, and the trace rearrangement bound). It is slightly looser than the purely scalar bound, but has the crucial advantage of being a quadratic form in the entries of $J_i$.

**Expansion of $Q_i$ in directional components.** Write $J_i$ row-wise:

$$J_i = \begin{bmatrix} b_{i,1}^\top \\ \vdots \\ b_{i,d}^\top \end{bmatrix}, \qquad b_{i,p} = \begin{bmatrix} (g_i^{(1)})_p \\ (g_i^{(2)})_p \end{bmatrix} \in \mathbb{R}^2.$$

Then

$$J_i W_i J_i^\top = \begin{bmatrix} b_{i,1}^\top W_i \\ \vdots \\ b_{i,d}^\top W_i \end{bmatrix} \begin{bmatrix} b_{i,1} & \cdots & b_{i,d} \end{bmatrix}.$$

The trace is the sum of diagonal entries; the $p$-th diagonal entry is $b_{i,p}^\top W_i \, b_{i,p}$. Hence:

$$Q_i(J_i) = \operatorname{tr}(J_i W_i J_i^\top) = \sum_{p=1}^{d} b_{i,p}^\top W_i \, b_{i,p}.$$

So $Q_i$ is a quadratic form in the directional modality-derivative pairs, and the MM surrogate is:

$$\rho(J_i) \le c_i + Q_i(J_i), \qquad Q_i(J_i) = \sum_{p=1}^{d} b_{i,p}^\top W_i \, b_{i,p}.$$

---

## C. Closed form for the MM block weight ($W_i$) using the 2×2 Gram matrix

### C.1 Gram matrix entries from gradients

Recall $S_i = J_i^\top J_i$. Write

$$S_i = \begin{bmatrix} a & b \\ b & c \end{bmatrix}, \qquad a = |g_i^{(1)}|^2, \quad c = |g_i^{(2)}|^2, \quad b = \langle g_i^{(1)}, g_i^{(2)} \rangle.$$

### C.2 Eigenvalues of a 2×2 symmetric matrix (derivation)

Eigenvalues solve $\det(S - \lambda I) = 0$:

$$\det \begin{bmatrix} a - \lambda & b \\ b & c - \lambda \end{bmatrix} = (a - \lambda)(c - \lambda) - b^2 = \lambda^2 - (a + c)\lambda + (ac - b^2) = 0.$$

By the quadratic formula:

$$\lambda = \frac{(a + c) \pm \sqrt{(a + c)^2 - 4(ac - b^2)}}{2}.$$

Simplify the discriminant:

$$(a + c)^2 - 4(ac - b^2) = a^2 + 2ac + c^2 - 4ac + 4b^2 = (a - c)^2 + 4b^2.$$

Define

$$\tau := a + c, \qquad \delta := \sqrt{(a - c)^2 + 4b^2}.$$

Then

$$\lambda_1 = \frac{\tau + \delta}{2}, \qquad \lambda_2 = \frac{\tau - \delta}{2}.$$

These are $\sigma_1^2$ and $\sigma_2^2$ respectively.

### C.3 MM scalar weights ($w_1, w_2$)

Using lagged eigenvalues $\bar{\lambda}_k = \bar{\sigma}_k^2$:

$$w_k = \frac{1}{2\sqrt{\bar{\lambda}_k + \varepsilon^2}}, \quad k = 1, 2.$$

### C.4 Avoid eigenvectors: express $W = h(\bar{S})$ as $W = uI + v\bar{S}$

Define the scalar function

$$h(\lambda) = \frac{1}{2\sqrt{\lambda + \varepsilon^2}}.$$

Then the MM weight matrix is the matrix function

$$W = h(\bar{S}),$$

meaning: if $\bar{S} = V \operatorname{diag}(\bar{\lambda}_1, \bar{\lambda}_2) V^\top$, then

$$W = V \operatorname{diag}(h(\bar{\lambda}_1), h(\bar{\lambda}_2)) V^\top = V \operatorname{diag}(w_1, w_2) V^\top.$$

For any $2 \times 2$ matrix function $h(\bar{S})$ (with distinct eigenvalues), $h(\bar{S})$ is an affine polynomial in $\bar{S}$:

$$W = uI + v\bar{S}.$$

**Reason**: The space of polynomials modulo the characteristic polynomial of a $2 \times 2$ matrix has dimension 2, so any analytic matrix function reduces to degree $\le 1$ in that quotient (standard functional calculus + Cayley–Hamilton argument).

To find $u, v$: enforce that $W$ has eigenvalues $w_1, w_2$ in the eigenbasis of $\bar{S}$. In that basis, $\bar{S}$ is $\operatorname{diag}(\bar{\lambda}_1, \bar{\lambda}_2)$, so

$$uI + v\bar{S} \;\mapsto\; \operatorname{diag}(u + v\bar{\lambda}_1, \; u + v\bar{\lambda}_2).$$

We need

$$u + v\bar{\lambda}_1 = w_1, \qquad u + v\bar{\lambda}_2 = w_2.$$

Subtract:

$$v(\bar{\lambda}_1 - \bar{\lambda}_2) = w_1 - w_2 \quad \Rightarrow \quad v = \frac{w_1 - w_2}{\bar{\lambda}_1 - \bar{\lambda}_2}.$$

Then

$$u = w_1 - v\bar{\lambda}_1 = \frac{\bar{\lambda}_1 w_2 - \bar{\lambda}_2 w_1}{\bar{\lambda}_1 - \bar{\lambda}_2}.$$

### C.5 Closed-form MM block weight (summary)

$$\boxed{W = uI + v\bar{S}}, \qquad v = \frac{w_1 - w_2}{\bar{\lambda}_1 - \bar{\lambda}_2}, \qquad u = \frac{\bar{\lambda}_1 w_2 - \bar{\lambda}_2 w_1}{\bar{\lambda}_1 - \bar{\lambda}_2}.$$

Writing $\bar{S} = \begin{bmatrix} \bar{a} & \bar{b} \\ \bar{b} & \bar{c} \end{bmatrix}$, the entries are explicitly:

$$W_{11} = u + v\bar{a}, \qquad W_{22} = u + v\bar{c}, \qquad W_{12} = W_{21} = v\bar{b}.$$

**Degenerate case** ($\bar{\lambda}_1 = \bar{\lambda}_2$, i.e. $\delta = 0$): then $w_1 = w_2 = w$ and the limit gives

$$W = wI, \qquad w = \frac{1}{2\sqrt{\bar{\lambda}_1 + \varepsilon^2}}.$$

This fully specifies the MM block weight with no eigendecomposition.

### C.6 Explicit 2×2 inverse of $W$ (needed later)

Since $W = \begin{bmatrix} W_{11} & W_{12} \\ W_{12} & W_{22} \end{bmatrix}$ is $2 \times 2$ SPD:

$$W^{-1} = \frac{1}{W_{11} W_{22} - W_{12}^2} \begin{bmatrix} W_{22} & -W_{12} \\ -W_{12} & W_{11} \end{bmatrix}.$$

The determinant satisfies $W_{11} W_{22} - W_{12}^2 = w_1 w_2 > 0$ (product of eigenvalues), which can also be computed as

$$w_1 w_2 = \frac{1}{4\sqrt{(\bar{\lambda}_1 + \varepsilon^2)(\bar{\lambda}_2 + \varepsilon^2)}}.$$

---

## D. The four preconditioners (local definitions)

Define preconditioners at the **edge / directional derivative level**, since that is where TNV lives most naturally.

Let the set of directed edges be $e = (i, p)$ meaning "voxel $i$, direction $p$". Define the two-modality directional difference vector

$$d_e := \begin{bmatrix} (G_p x^{(1)})_i \\ (G_p x^{(2)})_i \end{bmatrix} \in \mathbb{R}^2.$$

### Preconditioner 1: MM/IRLS block

Per voxel $i$, build $W_i \in \mathbb{S}_{++}^2$ from the lag Gram matrix $\bar{S}_i$ using the closed form in Section C.5. Then for every direction $p$:

$$B^{\text{MM-block}}_{(i,p)} := W_i.$$

### Preconditioner 2: MM/IRLS diagonal (Gershgorin row-sum of $W_i$)

Drop cross-modality coupling from the MM block weight by applying the **Gershgorin row-sum majoriser** to the $2 \times 2$ matrix $W_i$:

$$B^{\text{MM-diag}}_{(i,p)} := \operatorname{diag}\big((W_i)_{11} + |(W_i)_{12}|, \; (W_i)_{22} + |(W_i)_{12}|\big).$$

**Why not simply extract the diagonal of $W_i$?** The plain diagonal $\operatorname{diag}((W_i)_{11}, (W_i)_{22})$ does not satisfy $\operatorname{diag}(W_i) \succeq W_i$ in Loewner order when $(W_i)_{12} \ne 0$: the difference $\operatorname{diag}(W_i) - W_i = \begin{bmatrix} 0 & -(W_i)_{12} \\ -(W_i)_{12} & 0 \end{bmatrix}$ has eigenvalues $\pm|(W_i)_{12}|$, so it is indefinite. The MM-block surrogate gives $\rho(J_i) \le c_i + \operatorname{tr}(J_i W_i J_i^\top)$; to chain a diagonal relaxation $\operatorname{tr}(J_i W_i J_i^\top) \le \operatorname{tr}(J_i \tilde{W}_i J_i^\top)$ for all $J_i$, one needs $\tilde{W}_i \succeq W_i$. The Gershgorin row-sum diagonal provides exactly this (by the same Gershgorin argument used in Section H, applied to the $2 \times 2$ matrix $W_i$ itself).

**Explicit entries.** From Section C.5, $(W_i)_{12} = v\bar{b}$, so:

$$(B^{\text{MM-diag}}_{(i,p)})_{11} = u + v\bar{a} + |v\bar{b}|, \qquad (B^{\text{MM-diag}}_{(i,p)})_{22} = u + v\bar{c} + |v\bar{b}|.$$

**Degenerate case** ($\bar{\lambda}_1 = \bar{\lambda}_2$, i.e. $v = 0$): then $(W_i)_{12} = 0$ and the row-sum reduces to $B^{\text{MM-diag}}_{(i,p)} = \operatorname{diag}(w, w) = wI$, coinciding with the plain diagonal.

### Preconditioner 3: Lewis–Sendov block-Gershgorin majoriser

Compute the **true** Hessian of the smoothed TNV density $\rho(J_i)$ with respect to the $2d$ entries of $J_i$ (via Lewis–Sendov on the Hermitian dilation, derived in Sections E–G below). Represent this as a $2d \times 2d$ matrix $H_i$ and partition it into $d \times d$ blocks of size $2 \times 2$, indexed by direction:

$$H_i = \begin{bmatrix} H_{11} & H_{12} & \cdots \\ H_{21} & H_{22} & \cdots \\ \vdots & & \ddots \end{bmatrix}, \qquad H_{pq} \in \mathbb{R}^{2 \times 2},$$

where $H_{pq}$ couples the modality entries of directions $p$ and $q$.

Simply extracting the diagonal block $H_{pp}$ would be unprincipled: it satisfies neither $\operatorname{block\text{-}diag}(H_{pp}) \succeq H_i$ nor $\operatorname{block\text{-}diag}(H_{pp}) \preceq H_i$ in general. Instead, define the **block-Gershgorin majoriser**:

$$B^{\text{LS-bG}}_{(i,p)} := H_{pp} + \left(\sum_{q \ne p} \|H_{pq}\|_2\right) I_2,$$

where $\|\cdot\|_2$ denotes the spectral norm (largest singular value) of the $2 \times 2$ off-diagonal block. The proof that $\operatorname{block\text{-}diag}(B^{\text{LS-bG}}_{(i,p)}) \succeq H_i$ is given in Section G.4. The explicit computation of $H_{pq}$ from Lewis–Sendov coefficients is given in Section G.

### Preconditioner 4: Lewis–Sendov row-sum diagonal (Gershgorin majoriser)

Start from the (full) Lewis–Sendov Hessian matrix representation $H_i \in \mathbb{S}_+^{2d}$ of $\nabla^2 \rho(J_i)$ in the basis of entries of $J_i$. Then define the diagonal majoriser:

$$D^{\text{RS}}_{i,\alpha\alpha} := H_{i,\alpha\alpha} + \sum_{\beta \ne \alpha} |H_{i,\alpha\beta}|.$$

Then

$$B^{\text{LS-RS}}_{(i,p)} := \operatorname{diag}\Big(D^{\text{RS}}_{i,(p,1)}, \; D^{\text{RS}}_{i,(p,2)}\Big),$$

i.e. keep only the two diagonal entries belonging to direction $p$ and modalities 1 and 2.

---

## E. Lewis–Sendov machinery + explicit coefficients for TNV smoothing

### E.1 Why Hermitian dilation is used

We want second derivatives of a function of **singular values** of $J$. Lewis–Sendov gives second derivatives for functions of **eigenvalues of a symmetric matrix**. So we need a symmetric matrix whose eigenvalues are the singular values (up to sign).

Define the Hermitian dilation (Jordan–Wielandt matrix):

$$A(J) := \begin{bmatrix} 0 & J \\ J^\top & 0 \end{bmatrix} \in \mathbb{S}^{d+2}.$$

**Imported properties** (Tropp, *Matrix Analysis*, [tropp.caltech.edu](https://tropp.caltech.edu/notes/Tro22-Matrix-Analysis-LN.pdf)):

1. The mapping $J \mapsto A(J)$ is **real-linear**.
2. The eigenvalues of $A(J)$ are $\pm \sigma_k(J)$ (and zeros if $d > 2$).

Property (1) is crucial: linearity means $A(\Delta J)$ introduces no second-derivative terms from the map itself, so the chain rule for the Hessian involves only $\nabla^2 F$.

### E.2 Turn smoothed TNV into a symmetric-eigenvalue spectral function

Define the *even* smooth scalar function

$$\psi(t) := \sqrt{t^2 + \varepsilon^2}.$$

Its derivatives:

$$\psi'(t) = \frac{t}{\sqrt{t^2 + \varepsilon^2}}, \qquad \psi''(t) = \frac{\varepsilon^2}{(t^2 + \varepsilon^2)^{3/2}}.$$

Define the symmetric spectral function

$$F(A) := \operatorname{tr} \psi(A) = \sum_{j=1}^{d+2} \psi(\lambda_j(A)).$$

Apply this to $A(J)$. Its eigenvalues are $\pm\sigma_1, \pm\sigma_2$ and $d - 2$ zeros (for $d = 3$; in $d = 2$ there are no extra zeros). Using the fact that $\psi$ is even:

$$F(A(J)) = 2\psi(\sigma_1) + 2\psi(\sigma_2) + (d - 2)\psi(0) = 2\rho(J) + (d - 2)\varepsilon.$$

Therefore

$$\rho(J) = \frac{1}{2} F(A(J)) - \frac{d - 2}{2}\varepsilon.$$

The additive constant plays no role in derivatives.

### E.3 Lewis–Sendov: Hessian of spectral functions (statement)

Let $A \in \mathbb{S}^n$ with eigendecomposition $A = Q \operatorname{diag}(\lambda) Q^\top$. Let $f : \mathbb{R}^n \to \mathbb{R}$ be a symmetric $C^2$ function and consider the spectral function $f \circ \lambda$.

Lewis & Sendov give an explicit formula for the Hessian operator $\nabla^2 (f \circ \lambda)(A)[H]$ in terms of $\nabla^2 f(\lambda)$ and a "divided difference" matrix applied via Hadamard product. (Lewis & Sendov, *Twice Differentiable Spectral Functions*, [people.orie.cornell.edu](https://people.orie.cornell.edu/aslewis/publications/01-twice.pdf).)

For the separable case $f(\lambda) = \sum_j \psi(\lambda_j)$, the formula simplifies dramatically.

### E.4 Specialise Lewis–Sendov to $F(A) = \operatorname{tr} \psi(A)$

Here $f(\lambda) = \sum_j \psi(\lambda_j)$. Then:

- $\nabla f(\lambda) = (\psi'(\lambda_1), \dots, \psi'(\lambda_n))$,
- $\nabla^2 f(\lambda) = \operatorname{diag}(\psi''(\lambda_1), \dots, \psi''(\lambda_n))$.

Lewis–Sendov's operator formula for this separable case:

Given symmetric perturbation $H$, define $\tilde{H} := Q^\top H Q$. Then

$$\nabla^2 F(A)[H] = Q \Big( \operatorname{diag}\big(\psi''(\lambda) \odot \operatorname{diag}(\tilde{H})\big) + \mathcal{C}(\lambda) \circ \tilde{H} \Big) Q^\top,$$

where:

- $\odot$ is elementwise product of vectors,
- $\circ$ is Hadamard product of matrices,
- $\mathcal{C}(\lambda)$ is the off-diagonal coefficient matrix with entries:

$$\mathcal{C}_{ij}(\lambda) = \begin{cases} 0, & i = j, \\[4pt] \dfrac{\psi'(\lambda_i) - \psi'(\lambda_j)}{\lambda_i - \lambda_j}, & i \ne j, \; \lambda_i \ne \lambda_j, \\[10pt] \psi''(\lambda_i), & i \ne j, \; \lambda_i = \lambda_j. \end{cases}$$

This is precisely the Lewis–Sendov divided-difference structure specialised to separable $f$ (Lewis & Sendov, Theorem 3.3, [people.orie.cornell.edu](https://people.orie.cornell.edu/aslewis/publications/01-twice.pdf)).

---

## F. Explicit LS coefficients for the dilation eigenvalues $\{\pm\sigma_1, \pm\sigma_2, 0, \dots\}$

Let the eigenvalues of $A(J)$ be:

$$\lambda \in \{+\sigma_1, +\sigma_2, -\sigma_1, -\sigma_2, \underbrace{0, \dots, 0}_{d-2 \text{ times}}\}.$$

Define the shorthand

$$r(\sigma) := \sqrt{\sigma^2 + \varepsilon^2}.$$

Then for $t = \pm\sigma$:

$$\psi'(t) = \frac{t}{r(\sigma)}, \qquad \psi''(t) = \frac{\varepsilon^2}{r(\sigma)^3}.$$

And at $t = 0$:

$$\psi'(0) = 0, \qquad \psi''(0) = \frac{1}{\varepsilon}.$$

For any pair $(\lambda, \mu)$, define the LS off-diagonal coefficient:

$$c(\lambda, \mu) := \begin{cases} \dfrac{\psi'(\lambda) - \psi'(\mu)}{\lambda - \mu}, & \lambda \ne \mu, \\[10pt] \psi''(\lambda), & \lambda = \mu. \end{cases}$$

This is exactly the $\mathcal{C}_{ij}$ entry from Section E.4.

Now enumerate all cases that arise:

### F.1 Same magnitude, opposite sign: $(\sigma, -\sigma)$

$$c(\sigma, -\sigma) = \frac{\psi'(\sigma) - \psi'(-\sigma)}{\sigma - (-\sigma)} = \frac{\frac{\sigma}{r(\sigma)} - \left(-\frac{\sigma}{r(\sigma)}\right)}{2\sigma} = \frac{2\sigma / r(\sigma)}{2\sigma} = \frac{1}{r(\sigma)}.$$

### F.2 Pair with zero: $(\sigma, 0)$ or $(-\sigma, 0)$

$$c(\sigma, 0) = \frac{\psi'(\sigma) - \psi'(0)}{\sigma - 0} = \frac{\sigma / r(\sigma)}{\sigma} = \frac{1}{r(\sigma)}.$$

Similarly:

$$c(-\sigma, 0) = \frac{-\sigma / r(\sigma) - 0}{-\sigma} = \frac{1}{r(\sigma)}.$$

### F.3 Distinct positive singular values: $(\sigma_1, \sigma_2)$

$$c(\sigma_1, \sigma_2) = \frac{\dfrac{\sigma_1}{r(\sigma_1)} - \dfrac{\sigma_2}{r(\sigma_2)}}{\sigma_1 - \sigma_2}.$$

### F.4 Mixed sign, distinct magnitudes: $(\sigma_1, -\sigma_2)$

$$c(\sigma_1, -\sigma_2) = \frac{\dfrac{\sigma_1}{r(\sigma_1)} - \left(-\dfrac{\sigma_2}{r(\sigma_2)}\right)}{\sigma_1 + \sigma_2} = \frac{\dfrac{\sigma_1}{r(\sigma_1)} + \dfrac{\sigma_2}{r(\sigma_2)}}{\sigma_1 + \sigma_2}.$$

### F.5 Repeated zero block: $(0, 0)$ (only matters when $d = 3$)

For $\lambda = \mu = 0$:

$$c(0, 0) = \psi''(0) = \frac{1}{\varepsilon}.$$

In the LS formula, this fills the off-diagonal entries within the repeated-eigenvalue block (Lewis & Sendov, [people.orie.cornell.edu](https://people.orie.cornell.edu/aslewis/publications/01-twice.pdf)).

### F.6 Repeated singular values: $\sigma_1 = \sigma_2 = \sigma$

When the two positive eigenvalues coincide, the divided differences involving that pair become:

$$c(\sigma, \sigma) = \psi''(\sigma) = \frac{\varepsilon^2}{r(\sigma)^3}.$$

The pairs $(\sigma, -\sigma)$ and $(\sigma, 0)$ are unchanged from F.1 and F.2. This is handled by the LS formula's convention for repeated eigenvalues, but it is worth noting since the dilation eigenstructure changes: the $+\sigma$ eigenspace becomes two-dimensional, so the eigenvector matrix $Q$ is no longer unique (but the Hessian operator is still well-defined).

All other pairs reduce to one of the above by symmetry ($c(\lambda, \mu) = c(\mu, \lambda)$).

---

## G. Computing the LS Hessian blocks and the block-Gershgorin majoriser

### G.1 What the LS Hessian gives you (as an operator)

At a fixed voxel $i$, define $A := A(J_i) \in \mathbb{S}^{d+2}$. Lewis–Sendov gives an operator:

$$H \mapsto \nabla^2 F(A)[H] \in \mathbb{S}^{d+2}.$$

From Section E.2, $\rho(J) = \frac{1}{2} F(A(J)) + \text{const}$. By the chain rule, the bilinear form for $\rho$ satisfies:

$$\nabla^2 \rho(J)[\Delta J_1, \Delta J_2] = \frac{1}{2} \nabla^2 F(A)[A(\Delta J_1), A(\Delta J_2)].$$

We now show that the Riesz representative (i.e. the matrix $Y$ such that $\nabla^2 \rho[\Delta J_1, \Delta J_2] = \langle Y, \Delta J_2 \rangle_F$ for all $\Delta J_2$) is obtained **without** the factor $\frac{1}{2}$.

**Derivation of the cancellation.** Given $\Delta J_1$, let $\nabla^2 F(A)[A(\Delta J_1)] = QMQ^\top$ and write this in block form:

$$QMQ^\top = \begin{bmatrix} X & Y \\ Y^\top & Z \end{bmatrix}, \qquad X \in \mathbb{R}^{d \times d}, \; Y \in \mathbb{R}^{d \times 2}, \; Z \in \mathbb{R}^{2 \times 2}.$$

Then evaluate the bilinear form:

$$\nabla^2 F(A)[A(\Delta J_1), A(\Delta J_2)] = \operatorname{tr}\!\left(QMQ^\top \cdot \begin{bmatrix} 0 & \Delta J_2 \\ \Delta J_2^\top & 0 \end{bmatrix}\right).$$

Expanding the block matrix product:

$$= \operatorname{tr}\!\left(\begin{bmatrix} X & Y \\ Y^\top & Z \end{bmatrix} \begin{bmatrix} 0 & \Delta J_2 \\ \Delta J_2^\top & 0 \end{bmatrix}\right) = \operatorname{tr}\!\begin{bmatrix} Y \Delta J_2^\top & X \Delta J_2 \\ Z \Delta J_2^\top & Y^\top \Delta J_2 \end{bmatrix}.$$

The trace is the sum of diagonal block traces:

$$= \operatorname{tr}(Y \Delta J_2^\top) + \operatorname{tr}(Y^\top \Delta J_2) = 2 \langle Y, \Delta J_2 \rangle_F.$$

Substituting into the chain rule:

$$\nabla^2 \rho[\Delta J_1, \Delta J_2] = \frac{1}{2} \cdot 2 \langle Y, \Delta J_2 \rangle_F = \langle Y, \Delta J_2 \rangle_F.$$

So the Riesz representative is $Y = \Pi(QMQ^\top)$, where $\Pi(\cdot)$ denotes extraction of the top-right $d \times 2$ block. Therefore:

$$\boxed{\nabla^2 \rho(J)[\Delta J] = \Pi\!\big(\nabla^2 F(A)[A(\Delta J)]\big).}$$

The factor of $\frac{1}{2}$ from $\rho = \frac{1}{2}F \circ A$ is exactly cancelled by the factor of 2 arising from the dilation's off-diagonal symmetry.

### G.2 The full Hessian matrix and its block partition by direction

The $d \times 2$ matrix $J_i$ has $2d$ scalar entries. Index these entries as $(p, m)$ where $p \in \{1, \dots, d\}$ is the direction and $m \in \{1, 2\}$ is the modality. The full Hessian matrix $H_i \in \mathbb{S}_+^{2d}$ of $\nabla^2 \rho(J_i)$ in this coordinate basis has entries:

$$H_{i, (p,\alpha)(q,\beta)} = \big\langle E_{p,\alpha}, \; \nabla^2 \rho(J_i)[E_{q,\beta}] \big\rangle_F,$$

where $E_{p,m} \in \mathbb{R}^{d \times 2}$ is the basis matrix with a 1 at entry $(p, m)$ and zeros elsewhere.

Partition $H_i$ into $d \times d$ blocks of size $2 \times 2$, indexed by direction pair $(p, q)$:

$$H_i = \begin{bmatrix} H_{11} & H_{12} & \cdots \\ H_{21} & H_{22} & \cdots \\ \vdots & & \ddots \end{bmatrix}, \qquad (H_{pq})_{\alpha\beta} = H_{i,(p,\alpha)(q,\beta)}.$$

The diagonal block $H_{pp}$ captures the coupling between the two modalities within direction $p$. The off-diagonal block $H_{pq}$ ($p \ne q$) captures the cross-direction coupling.

### G.3 How to compute the Hessian blocks using the explicit LS coefficients

To compute any entry $H_{i,(p,\alpha)(q,\beta)}$, perform a **Hessian-vector product** with the basis perturbation $\Delta J = E_{q,\beta}$:

**Step 1.** Eigendecompose $A = Q \operatorname{diag}(\lambda) Q^\top$.

**Step 2.** Form the dilation perturbation $\Delta A = A(E_{q,\beta})$ and rotate:

$$\tilde{H} := Q^\top \Delta A \, Q.$$

**Step 3.** Build the coefficient matrix $\mathcal{C}(\lambda)$ using the explicit formulae in Section F:

- For $i \ne j$: $\mathcal{C}_{ij} = c(\lambda_i, \lambda_j)$.
- $\mathcal{C}_{ii} = 0$.

**Step 4.** Form:

$$M := \operatorname{diag}\big(\psi''(\lambda) \odot \operatorname{diag}(\tilde{H})\big) + \mathcal{C}(\lambda) \circ \tilde{H}.$$

**Step 5.** Rotate back:

$$\nabla^2 F(A)[\Delta A] = Q M Q^\top.$$

**Step 6.** Extract the top-right block (**no** factor of $\frac{1}{2}$, as derived in G.1):

$$Y^{(q,\beta)} := \nabla^2 \rho(J)[\Delta J] = \Pi(QMQ^\top).$$

**Step 7.** Read off entries: since $E_{p,\alpha}$ has a single nonzero entry, $\langle E_{p,\alpha}, Y^{(q,\beta)} \rangle_F = Y^{(q,\beta)}_{p,\alpha}$.

Steps 3–5 are exactly Lewis–Sendov's Theorem 3.3 specialised to separable $f = \sum \psi$ ([people.orie.cornell.edu](https://people.orie.cornell.edu/aslewis/publications/01-twice.pdf)).

To build the full $2d \times 2d$ Hessian requires $2d$ Hessian-vector products (one per basis element $E_{q,\beta}$). To build only the diagonal block $H_{pp}$ requires 2 Hessian-vector products. To build the block-Gershgorin majoriser $\tilde{B}_p = H_{pp} + (\sum_{q \ne p} \|H_{pq}\|_2) I_2$ for all $p$, the **full $2d$ HVPs are needed**, since the spectral norms $\|H_{pq}\|_2$ require the off-diagonal blocks. This is the main computational cost of the block-Gershgorin approach relative to the scalar row-sum (which requires the same $2d$ HVPs but avoids the $2 \times 2$ spectral norm computation) and relative to the MM constructions (which require no HVPs at all).

### G.4 Block-Gershgorin majorisation (proof)

**Claim**: Define the block-diagonal matrix $\tilde{D} := \operatorname{block\text{-}diag}(\tilde{B}_1, \dots, \tilde{B}_d)$ where

$$\tilde{B}_p := H_{pp} + \left(\sum_{q \ne p} \|H_{pq}\|_2\right) I_2.$$

Then $\tilde{D} \succeq H_i$.

**Proof.** Let $v = (v_1, \dots, v_d) \in \mathbb{R}^{2d}$ with $v_p \in \mathbb{R}^2$. Then:

$$v^\top H_i \, v = \sum_p v_p^\top H_{pp} \, v_p + \sum_{p \ne q} v_p^\top H_{pq} \, v_q.$$

For the cross terms, the submultiplicativity of the spectral norm gives:

$$|v_p^\top H_{pq} \, v_q| \le \|H_{pq}\|_2 \, \|v_p\| \, \|v_q\|.$$

By the AM–GM inequality, $\|v_p\| \, \|v_q\| \le \frac{1}{2}(\|v_p\|^2 + \|v_q\|^2)$. Sum over all ordered pairs $(p, q)$ with $p \ne q$:

$$\sum_{p \ne q} \|H_{pq}\|_2 \, \|v_p\| \, \|v_q\| \le \sum_{p \ne q} \|H_{pq}\|_2 \cdot \frac{\|v_p\|^2 + \|v_q\|^2}{2}.$$

Since $H_i$ is symmetric, $H_{qp} = H_{pq}^\top$, so $\|H_{qp}\|_2 = \|H_{pq}\|_2$. Relabelling the $\|v_q\|^2$ terms by swapping the summation indices $p \leftrightarrow q$ in the second half:

$$= \sum_p \left(\sum_{q \ne p} \|H_{pq}\|_2\right) \|v_p\|^2.$$

Now compute the quadratic form of $\tilde{D}$:

$$v^\top \tilde{D} \, v = \sum_p v_p^\top \tilde{B}_p \, v_p = \sum_p v_p^\top H_{pp} \, v_p + \sum_p \left(\sum_{q \ne p} \|H_{pq}\|_2\right) \|v_p\|^2.$$

The second term is exactly the upper bound on $\big|\sum_{p \ne q} v_p^\top H_{pq} v_q\big|$ established above. Therefore:

$$v^\top \tilde{D} \, v \ge \sum_p v_p^\top H_{pp} \, v_p + \sum_{p \ne q} v_p^\top H_{pq} \, v_q = v^\top H_i \, v$$

for all $v \in \mathbb{R}^{2d}$, i.e. $\tilde{D} \succeq H_i$. $\square$

### G.5 Comparing block-Gershgorin and scalar row-sum

Both $\tilde{D}$ and $D^{\text{RS}}$ independently majorise $H_i$:

$$H_i \preceq \tilde{D}, \qquad H_i \preceq D^{\text{RS}}.$$

However, there is **no universal Loewner ordering between $\tilde{D}$ and $D^{\text{RS}}$**. The matrix $\tilde{D}$ is block-diagonal with $2 \times 2$ blocks that retain the within-direction off-diagonal entries $(H_{pp})_{12}$, while $D^{\text{RS}}$ is scalar-diagonal. Even if the diagonal entries of $D^{\text{RS}}$ exceed those of $\tilde{D}$ entrywise, this does not imply $\tilde{D} \preceq D^{\text{RS}}$ in PSD order, because a scalar-diagonal matrix cannot automatically Loewner-dominate a non-diagonal block.

**Why block-Gershgorin is typically less conservative.** Two independent effects contribute:

1. The spectral norm bound $\|H_{pq}\|_2 \le \sum_{a,b} |(H_{pq})_{ab}|$ means the inflation added per direction is generically smaller for block-Gershgorin than for the scalar row-sum.
2. Block-Gershgorin retains the $2 \times 2$ cross-modality structure of $H_{pp}$, whereas the scalar row-sum discards it entirely.

These are empirical/structural observations, not a Loewner-order theorem. In practice, the block-Gershgorin preconditioner will be tighter for most voxel configurations, but the comparison is not universal.

### G.6 Spectral norm of a 2×2 matrix (closed form)

For the off-diagonal blocks $H_{pq} \in \mathbb{R}^{2 \times 2}$, the spectral norm has a closed form. Let

$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}.$$

Then $\|A\|_2^2$ is the largest eigenvalue of $A^\top A$. Compute:

$$A^\top A = \begin{bmatrix} a_{11}^2 + a_{21}^2 & a_{11} a_{12} + a_{21} a_{22} \\ a_{11} a_{12} + a_{21} a_{22} & a_{12}^2 + a_{22}^2 \end{bmatrix}.$$

Define

$$s := \operatorname{tr}(A^\top A) = a_{11}^2 + a_{12}^2 + a_{21}^2 + a_{22}^2 = \|A\|_F^2,$$

$$\Delta := \sqrt{s^2 - 4 \det(A^\top A)}.$$

By the $2 \times 2$ eigenvalue formula (Section C.2 applied to $A^\top A$):

$$\|A\|_2^2 = \frac{s + \Delta}{2}.$$

Since $d \in \{2, 3\}$, each block $\tilde{B}_p$ requires at most $d - 1 \in \{1, 2\}$ such spectral norm computations — negligible cost.

---

## H. LS scalar row-sum diagonal majoriser (and why it majorises)

The scalar row-sum majoriser is the simplest of the LS-based options. It independently majorises $H_i$ (as does the block-Gershgorin from Section G.4), but discards all cross-modality coupling. See Section G.5 for a comparison of the two.

Let $H_i \in \mathbb{S}_+^{2d}$ be the (local) Hessian matrix representation of $\nabla^2 \rho(J_i)$ in the coordinate basis of the entries of $J_i$.

Define the diagonal matrix $D_i$ by

$$(D_i)_{\alpha\alpha} := (H_i)_{\alpha\alpha} + \sum_{\beta \ne \alpha} |(H_i)_{\alpha\beta}|.$$

Consider $M_i := D_i - H_i$. Then:

- $M_i$ is symmetric.
- For each row $\alpha$:

$$(M_i)_{\alpha\alpha} = \sum_{\beta \ne \alpha} |(H_i)_{\alpha\beta}| \ge 0,$$

and

$$\sum_{\beta \ne \alpha} |(M_i)_{\alpha\beta}| = \sum_{\beta \ne \alpha} |{-}(H_i)_{\alpha\beta}| = \sum_{\beta \ne \alpha} |(H_i)_{\alpha\beta}| = (M_i)_{\alpha\alpha}.$$

So $M_i$ is (weakly) **diagonally dominant** with nonnegative diagonal.

Now apply Gershgorin's theorem: for a real symmetric matrix, eigenvalues lie in the union of real Gershgorin intervals $[a_{\alpha\alpha} - r_\alpha, \; a_{\alpha\alpha} + r_\alpha]$ (Alexander, *Gershgorin Circles*, [aalexan3.math.ncsu.edu](https://aalexan3.math.ncsu.edu/articles/gershgorin-report.pdf)).

Here $a_{\alpha\alpha} = (M_i)_{\alpha\alpha}$ and $r_\alpha = \sum_{\beta \ne \alpha} |(M_i)_{\alpha\beta}| = (M_i)_{\alpha\alpha}$, so each interval is $[0, \; 2(M_i)_{\alpha\alpha}] \subseteq [0, \infty)$. Therefore all eigenvalues of $M_i$ are $\ge 0$, i.e.:

$$M_i = D_i - H_i \succeq 0 \quad \Longrightarrow \quad H_i \preceq D_i.$$

That is the precise PSD-order statement that the row-sum diagonal is a **majoriser** of curvature.

The **LS row-sum preconditioner** is the diagonal $D_i$ (or its restriction to the two entries belonging to each direction $p$).

---

## I. Mapping local edge blocks to an actual block-diagonal image-space preconditioner

### I.1 Write the surrogate as a quadratic form in the image variables

Let the stacked unknown be

$$x := \begin{bmatrix} x^{(1)} \\ x^{(2)} \end{bmatrix} \in \mathbb{R}^{2N}.$$

For each directed edge $e = (i, p)$, define the linear operator $D_e : \mathbb{R}^{2N} \to \mathbb{R}^2$ by

$$D_e x := \begin{bmatrix} (G_p x^{(1)})_i \\ (G_p x^{(2)})_i \end{bmatrix}.$$

This is linear in $x$.

**Practical note.** In implementations, $D_e$ typically includes additional spatial weight and directional sensitivity factors (e.g. voxel-size scaling, anatomical weighting), so the actual edge variable is of the form $(\text{weight}) \cdot D_e x$ rather than the bare finite difference. The edge blocks $B_e$ are then applied to these *scaled* differences. The algebraic structure of the mapping below is unchanged; only the definition of $D_e$ absorbs the extra factors.

All four preconditioners provide a $2 \times 2$ **edge block** $B_e$ (either $W_i$, $\operatorname{diag}(W_i)$, $B^{\text{LS-bG}}_{(i,p)}$, or a diagonal from row-sum).

The generic quadratic surrogate (dropping constants) is:

$$\widetilde{R}(x) = \sum_{e} (D_e x)^\top B_e (D_e x).$$

Because $D_e$ is linear:

$$(D_e x)^\top B_e (D_e x) = x^\top (D_e^\top B_e D_e) x,$$

so

$$\widetilde{R}(x) = x^\top \left(\sum_e D_e^\top B_e D_e\right) x.$$

Hence the surrogate Hessian in image space is

$$H_{\text{img}} := 2 \sum_e D_e^\top B_e D_e$$

(the factor 2 comes from $\nabla^2(x^\top A x) = 2A$; for preconditioning this global constant can be absorbed).

### I.2 Expand one edge contribution explicitly (this gives the diagonal blocks)

Consider one edge connecting voxel $i$ to voxel $j = i + e_p$ (forward difference). Then for modality $m$:

$$(G_p x^{(m)})_i = x^{(m)}_j - x^{(m)}_i.$$

So

$$D_e x = x_j - x_i \in \mathbb{R}^2, \qquad \text{where } x_i = \begin{bmatrix} x^{(1)}_i \\ x^{(2)}_i \end{bmatrix}.$$

Then the edge energy is:

$$(x_j - x_i)^\top B_e (x_j - x_i).$$

Expand fully:

$$(x_j - x_i)^\top B_e (x_j - x_i) = x_j^\top B_e x_j + x_i^\top B_e x_i - x_j^\top B_e x_i - x_i^\top B_e x_j.$$

Because $B_e$ is symmetric, $x_j^\top B_e x_i = x_i^\top B_e x_j$, so:

$$(x_j - x_i)^\top B_e (x_j - x_i) = x_i^\top B_e x_i + x_j^\top B_e x_j - 2 x_i^\top B_e x_j.$$

From this single-edge expansion, contributions to the global Hessian are:

- The **block-diagonal** contribution at voxel $i$ gets $+B_e$.
- At voxel $j$ gets $+B_e$.
- The off-diagonal coupling between $i$ and $j$ is $-B_e$ (and symmetric).

### I.3 The block-Jacobi (per-voxel) preconditioner

A standard practical preconditioner is to take only the **block diagonal** of $H_{\text{img}}$. From the expansion above, the block diagonal at voxel $i$ is:

$$P_i := \sum_{e \text{ incident to } i} B_e,$$

i.e. **sum the $2 \times 2$ edge blocks of all edges touching voxel $i$**.

For a regular grid with forward differences, the incident edges are:

- Edges starting at $i$: $(i, p)$ for $p = 1, \dots, d$.
- Edges ending at $i$: $(i - e_p, p)$ for $p = 1, \dots, d$.

So explicitly:

$$\boxed{P_i = \sum_{p=1}^{d} \left(B_{(i,p)} + B_{(i-e_p, p)}\right),}$$

with appropriate boundary conventions (edges outside the domain are omitted or set to zero).

This is the mapping: **edge-space curvature blocks $\to$ voxel-space block-diagonal preconditioner**.

### I.4 Inverting the 2×2 blocks (what the preconditioned solver uses)

#### Case 1: MM-block or LS-block-Gershgorin ($P_i$ is $2 \times 2$ SPD)

Write

$$P_i = \begin{bmatrix} \alpha & \beta \\ \beta & \gamma \end{bmatrix}, \qquad \alpha\gamma - \beta^2 > 0.$$

Then

$$\boxed{P_i^{-1} = \frac{1}{\alpha\gamma - \beta^2} \begin{bmatrix} \gamma & -\beta \\ -\beta & \alpha \end{bmatrix}.}$$

**Application to a gradient vector**: given the stacked gradient at voxel $i$,

$$\nabla_i = \begin{bmatrix} (\nabla R)^{(1)}_i \\ (\nabla R)^{(2)}_i \end{bmatrix},$$

the preconditioned direction is

$$P_i^{-1} \nabla_i = \frac{1}{\alpha\gamma - \beta^2} \begin{bmatrix} \gamma (\nabla R)^{(1)}_i - \beta (\nabla R)^{(2)}_i \\ -\beta (\nabla R)^{(1)}_i + \alpha (\nabla R)^{(2)}_i \end{bmatrix}.$$

**Numerical safeguard**: in practice, add a small diagonal shift $\mu > 0$ to ensure positive definiteness:

$$P_i \leftarrow P_i + \mu I.$$

This guarantees $\alpha\gamma - \beta^2 > 0$ even if the sum of edge blocks is nearly singular (e.g. in flat regions).

#### Case 2: MM-diag or LS-row-sum ($P_i$ is diagonal)

Write

$$P_i = \begin{bmatrix} \alpha & 0 \\ 0 & \gamma \end{bmatrix}.$$

Then

$$P_i^{-1} = \begin{bmatrix} 1/\alpha & 0 \\ 0 & 1/\gamma \end{bmatrix},$$

and the preconditioned direction is simply componentwise division:

$$P_i^{-1} \nabla_i = \begin{bmatrix} (\nabla R)^{(1)}_i / \alpha \\ (\nabla R)^{(2)}_i / \gamma \end{bmatrix}.$$

### I.5 Combining with data-term curvature

In practice, the preconditioner for the full objective $\Phi = \Phi_{\text{data}} + \lambda R$ is:

$$P_i^{\text{total}} = P_i^{\text{data}} + \lambda \, P_i^{\text{prior}},$$

then apply $(P_i^{\text{total}})^{-1}$ to the (stochastic) gradient. The mapping above gives $P_i^{\text{prior}}$ in a principled way for all four constructions. The inversion formula in I.4 applies to $P_i^{\text{total}}$ identically (it is still $2 \times 2$ SPD).

---

## Summary: fully explicit components

| Preconditioner | Edge block $B_e$ | Closed form? | Coupling | Justification |
|---|---|---|---|---|
| **MM-block** | $W_i = uI + v\bar{S}_i$ (Section C.5) | Yes — no eigendecomposition | Cross-modality | Majorises $\rho$ (function value) at lag point; yields PSD quadratic surrogate |
| **MM-diag** | $\operatorname{diag}((W_i)_{11} + |(W_i)_{12}|, \; (W_i)_{22} + |(W_i)_{12}|)$ | Yes | None | Majorises $\rho$ (function value) at lag point via Gershgorin row-sum of $W_i$; yields PSD diagonal surrogate |
| **LS-block-Gershgorin** | $H_{pp} + (\sum_{q \ne p} \|H_{pq}\|_2) I_2$ (Section G.4) | Semi-closed (requires dilation eigendecomposition) | Cross-modality | Majorises $H_i$ in Loewner order (Section G.4) |
| **LS-row-sum** | Absolute row sums of full LS Hessian (Section H) | Semi-closed | None | Majorises $H_i$ in Loewner order (Section H) |

**Note on majorisation type.** The MM constructions prove $\rho(J) \le c + \operatorname{tr}(JWJ^\top)$ for all $J$, i.e. majorisation of the function value at the lag point. This yields a valid PSD quadratic surrogate (and hence a valid SPD preconditioner), but does **not** imply the Loewner-order bound $\nabla^2 \rho(J) \preceq \nabla^2 Q(J)$ for all $J$. The LS constructions directly majorise the Hessian matrix in Loewner order; the block-Gershgorin and scalar row-sum independently satisfy $H_i \preceq \tilde{D}$ and $H_i \preceq D^{\text{RS}}$ respectively, but are Loewner-incomparable with each other (Section G.5).

**Edge blocks $\to$ voxel blocks**: $P_i = \sum_{\text{incident edges}} B_e$ (Section I.3).

**Inversion**: $2 \times 2$ SPD inverse via $(\alpha\gamma - \beta^2)^{-1} \begin{bmatrix} \gamma & -\beta \\ -\beta & \alpha \end{bmatrix}$, or componentwise for diagonal variants (Section I.4).

**Key corrections from prior versions**:

1. **Factor-of-2 cancellation** (Section G.1): The Riesz representative of $\nabla^2\rho$ is $\Pi(QMQ^\top)$, not $\frac{1}{2}\Pi(QMQ^\top)$. The $\frac{1}{2}$ from $\rho = \frac{1}{2}F \circ A$ is cancelled by the factor of 2 from the dilation's off-diagonal symmetry.

2. **Trace rearrangement inequality** (Section B.3): The MM quadratic surrogate $Q_i(J) = \operatorname{tr}(JWJ^\top)$ majorises $\sum_k w_k \sigma_k^2$ via two composed inequalities: the scalar MM concavity bound and the von Neumann trace inequality. These are now stated separately and explicitly.

3. **Block-Gershgorin replaces naive LS-block** (Sections D, G.4): The previous Preconditioner 3 simply extracted the diagonal block $H_{pp}$, which is not a majoriser. The block-Gershgorin construction $H_{pp} + (\sum_{q \ne p} \|H_{pq}\|_2) I_2$ is a provable Loewner majoriser that retains cross-modality coupling.

4. **MM vs LS majorisation type** (Summary): MM constructions majorise the function value $\rho$ (valid surrogate), not the Hessian matrix in Loewner order. LS constructions majorise the Hessian directly. These are logically independent properties.

5. **Block-Gershgorin vs scalar row-sum** (Section G.5): The two LS majorisers are Loewner-incomparable in general, not ordered as previously claimed. Both independently majorise $H_i$, but there is no universal $\tilde{D} \preceq D^{\text{RS}}$.

6. **MM-diag requires Gershgorin row sums** (Section D, Preconditioner 2): The previous definition $\operatorname{diag}((W_i)_{11}, (W_i)_{22})$ is not a valid Loewner majoriser of $W_i$ when $(W_i)_{12} \ne 0$, so the chained bound $\rho \le c + \operatorname{tr}(JWJ^\top) \le c + \operatorname{tr}(J\tilde{W}J^\top)$ fails. The corrected definition uses the Gershgorin row-sum diagonal of $W_i$, adding $|(W_i)_{12}|$ to each diagonal entry.
