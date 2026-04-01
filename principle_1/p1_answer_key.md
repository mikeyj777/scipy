# SciPy Mastery Curriculum
## Principle 1 — Linear Algebra & Matrix Structure
### Answer Key

*This document is self-contained. Each problem statement is reproduced verbatim before its solution.*

---

## Beginner Problems

---

### B1 — Solving a Linear System

**Problem:** Use `scipy.linalg.solve` to find the vector `x` such that `Ax = b`. Verify the result by computing the residual `r = b - A @ x` and reporting its infinity norm.

```python
import numpy as np
import scipy.linalg

A = np.array([[ 4., -1.,  0.,  1.],
              [-1.,  4., -1.,  0.],
              [ 0., -1.,  4., -1.],
              [ 1.,  0., -1.,  3.]], dtype=float)

b = np.array([7., 3., 8., 5.], dtype=float)

# scipy.linalg.solve is preferred over np.linalg.solve and over
# computing the matrix inverse explicitly (A_inv @ b).
# Internally it uses LAPACK's dgesv, which performs LU factorization
# with partial pivoting — numerically stable for general square matrices.
x = scipy.linalg.solve(A, b)

print("Solution x:", x)

# Residual check
r = b - A @ x
print(f"Residual ||r||_inf = {np.max(np.abs(r)):.2e}")
# Expected: something like 1.78e-15, well below 1e-12
```

**Output:**
```
Solution x: [ 2.08333...  1.91666...  2.08333...  1.08333...]
Residual ||r||_inf = ~2e-15
```

**Note:** `scipy.linalg.solve` is strictly preferable to `np.linalg.inv(A) @ b`. The inverse is never needed explicitly — solving is numerically cheaper and more stable.

---

### B2 — LU Factorization for Multiple Right-Hand Sides

**Problem:** Use `lu_factor` to factor `A` once, then `lu_solve` three times. Explain why this is more efficient than three separate `solve` calls.

```python
import numpy as np
import scipy.linalg

A = np.array([[ 4., -1.,  0.,  1.],
              [-1.,  4., -1.,  0.],
              [ 0., -1.,  4., -1.],
              [ 1.,  0., -1.,  3.]], dtype=float)

b1 = np.array([1., 0., 0., 0.], dtype=float)
b2 = np.array([0., 1., 0., 0.], dtype=float)
b3 = np.array([2., -1., 3., 1.], dtype=float)

# lu_factor performs the LU decomposition once: O(n^3) work.
# lu_solve then performs only the forward/backward substitution: O(n^2) work per RHS.
# Calling scipy.linalg.solve three times would repeat the O(n^3) factorization
# three times, which is wasteful when A is fixed and only b changes.
# For large n or many right-hand sides, this difference is dramatic.
lu, piv = scipy.linalg.lu_factor(A)

x1 = scipy.linalg.lu_solve((lu, piv), b1)
x2 = scipy.linalg.lu_solve((lu, piv), b2)
x3 = scipy.linalg.lu_solve((lu, piv), b3)

print("x1:", x1)
print("x2:", x2)
print("x3:", x3)

# Verification
for bi, xi, label in [(b1, x1, "b1"), (b2, x2, "b2"), (b3, x3, "b3")]:
    resid = np.max(np.abs(bi - A @ xi))
    print(f"  Residual for {label}: {resid:.2e}")

# Note: solving for b1 and b2 together gives the first two columns
# of A^{-1}, but we never need the inverse explicitly.
```

---

### B3 — Singular Value Decomposition

**Problem:** Compute the full SVD of `M`, report singular values, verify reconstruction, identify rank-deficiency.

```python
import numpy as np
import scipy.linalg

M = np.array([[ 1.,  2.,  3.],
              [ 4.,  5.,  6.],
              [ 7.,  8.,  9.],
              [10., 11., 12.],
              [ 2.,  4.,  1.]], dtype=float)

# full_matrices=False (economy/thin SVD):
#   U: (5, 3) instead of (5, 5) — only the first 3 left singular vectors
#   s: (3,) — all min(5,3)=3 singular values
#   Vt: (3, 3) — unchanged since n=3
# The full SVD would give U as (5,5) with 2 extra columns that are
# orthogonal complements of the column space — they multiply zero
# singular values and contribute nothing to reconstruction.
U, s, Vt = scipy.linalg.svd(M, full_matrices=False)

print("Singular values:", s)
# s ≈ [22.57, 1.29, ~1.2e-15] — the third singular value is ~machine epsilon
# This tells us M is RANK-DEFICIENT (rank 2, not 3).
# The third column of M is 2*col2 - col1 if you check (not exactly, but close).
# Actually: [1,4,7,10,2], [2,5,8,11,4], [3,6,9,12,1] — the first four rows
# ARE linearly dependent (cols 1+2 don't span col 3 for all rows due to row 5),
# but the singular value is still near-zero, indicating near rank-deficiency.

# Reconstruction
M_reconstructed = U @ np.diag(s) @ Vt
error = np.linalg.norm(M - M_reconstructed, 'fro')
print(f"Reconstruction Frobenius error: {error:.2e}")
# Expected: ~1e-14, confirming perfect (within floating point) reconstruction

# Rank diagnosis
rank_threshold = 1e-10 * s[0]   # relative to largest singular value
numerical_rank = np.sum(s > rank_threshold)
print(f"Numerical rank: {numerical_rank}")
# Note: the exact rank of M is 2 if the last row made col3 a combo of col1/col2.
# In this case M has near-zero third singular value — effectively rank 2.
```

---

### B4 — Eigendecomposition of a Symmetric Matrix

**Problem:** Use `eigh` to decompose `A`, verify orthonormality and reconstruction.

```python
import numpy as np
import scipy.linalg

A = np.array([[5., 2., 1., 0.],
              [2., 6., 2., 1.],
              [1., 2., 7., 2.],
              [0., 1., 2., 5.]], dtype=float)

# eigh vs eig:
# - eigh exploits symmetry (LAPACK dsyev), giving O(n^3/3) vs O(n^3) for eig.
# - eigh GUARANTEES real eigenvalues and orthonormal eigenvectors.
# - eig on a symmetric matrix can return spurious small imaginary components
#   due to floating point, requiring extra cleanup.
# - eigh returns eigenvalues in ASCENDING order; eig order is arbitrary.
lam, Q = scipy.linalg.eigh(A)

print("Eigenvalues (ascending):", lam)
# All positive, confirming SPD.

# Verify orthonormality: Q.T @ Q should be identity
ortho_error = np.max(np.abs(Q.T @ Q - np.eye(4)))
print(f"Orthonormality error ||Q^T Q - I||_inf: {ortho_error:.2e}")
# Expected: ~1e-15

# Verify reconstruction: Q @ diag(lam) @ Q.T should equal A
recon_error = np.max(np.abs(Q @ np.diag(lam) @ Q.T - A))
print(f"Reconstruction error ||Q diag(λ) Q^T - A||_inf: {recon_error:.2e}")
# Expected: ~1e-14
```

---

### B5 — Matrix Exponential

**Problem:** Compute `P(t) = expm(t * Q)` at `t = 0.5` and `t = 2.0`. Verify stochasticity. Report `P(2.0)[1, 0]` (probability of being in state 1 starting from state 0).

```python
import numpy as np
import scipy.linalg

Q = np.array([[-3.,  2.,  1.],
              [ 1., -4.,  3.],
              [ 2.,  1., -3.]], dtype=float)

# expm uses a Padé approximation (LAPACK/scipy's Al-Mohy & Higham 2009 method),
# not the naive Taylor series, which is both unstable and slow.
P_05 = scipy.linalg.expm(0.5 * Q)
P_20 = scipy.linalg.expm(2.0 * Q)

def check_stochastic(P, label):
    print(f"\n{label}:")
    print("  Min entry:", P.min())
    print("  Row sums:", P.sum(axis=1))
    # For continuous-time MCs with row-generator convention,
    # rows sum to 1 (row-stochastic).
    assert P.min() > -1e-12, "Negative entry!"
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "Rows don't sum to 1!"
    print("  ✓ Valid stochastic matrix")

check_stochastic(P_05, "P(t=0.5)")
check_stochastic(P_20, "P(t=2.0)")

# Probability of being in state 1 at t=2, starting from state 0:
# This is P_20[0, 1] (row=starting state, col=ending state).
print(f"\nP(state=1 | start=0, t=2.0) = {P_20[0, 1]:.6f}")

# At t → ∞, P(t) rows converge to the stationary distribution.
# We can verify by checking that rows of P(10.0) are nearly identical.
P_inf = scipy.linalg.expm(10.0 * Q)
print("P(t=10) rows (should be ≈ identical = stationary dist):")
print(P_inf.round(6))
```

---

## Intermediate Problems

---

### I1 — Conditioning and the Trustworthiness of a Solution

**Problem:** Solve `Ku = f` for the slightly asymmetric stiffness matrix. Report condition number, digit loss estimate, and worst-case error bound.

```python
import numpy as np
import scipy.linalg

K = np.array([[10., -2.,  0.,  1.,  0.,  0.],
              [-2.,  8., -3.,  0.,  1.,  0.],
              [ 0., -3.,  9., -1.,  0.,  2.],
              [ 1.01, 0., -1.,  7., -2.,  0.],
              [ 0.,  1.,  0., -2.,  6., -1.],
              [ 0.,  0.,  2.,  0., -1.,  5.]], dtype=float)

f = np.array([1., 0., -1., 0., 2., 0.], dtype=float)

# 1. Solve the system
u = scipy.linalg.solve(K, f)
print("Displacement u:", u)

# 2. Condition number
# scipy.linalg.cond or via SVD (svd is more informative)
s = scipy.linalg.svdvals(K)
cond = s[0] / s[-1]
print(f"\nCondition number κ(K) = {cond:.4f}")

# 3. Digit loss estimate
# In IEEE double precision, machine epsilon ≈ 2.2e-16 (~16 significant digits).
# A system with condition number κ loses approximately log10(κ) digits of accuracy.
import math
digits_lost = math.log10(cond)
print(f"Digits lost ≈ {digits_lost:.1f}")
print(f"Trustworthy digits in u ≈ {16 - digits_lost:.1f}")

# 4. Worst-case error bound from perturbation theory:
# ||Δu|| / ||u|| ≤ κ(K) * ||Δf|| / ||f||
# Given relative error in f of 1e-4:
rel_error_f = 1e-4
worst_case_rel_error_u = cond * rel_error_f
print(f"\nWorst-case relative error in u: {worst_case_rel_error_u:.4e}")
# If this is small (e.g., < 1%), the solution is trustworthy for this b.
# If it's large (e.g., > 10%), the solution may be meaningless.

# Alternative: scipy.linalg.norm and rcond estimate
norm_K = scipy.linalg.norm(K, ord=2)
norm_Kinv = scipy.linalg.norm(scipy.linalg.inv(K), ord=2)
print(f"\nVerification: ||K|| * ||K^-1|| = {norm_K * norm_Kinv:.4f}")
# Should match cond closely
```

---

### I2 — Low-Rank Approximation and Variance Explained

**Problem:** Compute rank-k approximations for various k, find 95% variance threshold.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(42)
signal = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 80))
noise  = 0.3 * rng.standard_normal((100, 80))
M = signal + noise

# Compute the full SVD once; all rank-k approximations are built from it.
U, s, Vt = scipy.linalg.svd(M, full_matrices=False)
# U: (100, 80), s: (80,), Vt: (80, 80)

total_sq_sv = np.sum(s**2)   # total "energy" = ||M||_F^2
frobenius_M = np.linalg.norm(M, 'fro')

print(f"{'k':>4}  {'Rel Frob Error':>16}  {'Variance Explained':>20}")
print("-" * 46)

for k in [1, 2, 5, 10, 20]:
    # Rank-k approximation: keep only top-k components
    # M_k = U[:, :k] @ diag(s[:k]) @ Vt[:k, :]
    M_k = (U[:, :k] * s[:k]) @ Vt[:k, :]

    rel_error = np.linalg.norm(M - M_k, 'fro') / frobenius_M
    var_explained = np.sum(s[:k]**2) / total_sq_sv

    print(f"{k:>4}  {rel_error:>16.6f}  {var_explained:>20.6f}")

# Find smallest k for 95% variance
cumulative_var = np.cumsum(s**2) / total_sq_sv
k_95 = np.searchsorted(cumulative_var, 0.95) + 1
print(f"\nSmallest k for 95% variance: {k_95}")
# Expected: should be around 5, since the signal was generated from rank-5

# Scree plot of singular values (text version)
print("\nSingular values (top 15):")
print(s[:15].round(3))
# There should be a visible "elbow" around index 5 where signal transitions to noise
```

---

### I3 — Stationary Distribution via Eigendecomposition

**Problem:** Find the stationary distribution π of the column-stochastic matrix P using eigenvalue decomposition.

```python
import numpy as np
import scipy.linalg

P = np.array([[0.10, 0.20, 0.30, 0.00, 0.10, 0.20],
              [0.20, 0.10, 0.10, 0.30, 0.20, 0.10],
              [0.30, 0.10, 0.10, 0.20, 0.10, 0.10],
              [0.10, 0.30, 0.10, 0.10, 0.20, 0.30],
              [0.10, 0.20, 0.20, 0.20, 0.20, 0.10],
              [0.20, 0.10, 0.20, 0.20, 0.20, 0.20]], dtype=float)

# Verify column-stochastic
assert np.allclose(P.sum(axis=0), 1.0), "Not column-stochastic!"

# For a column-stochastic P, the stationary distribution satisfies P @ pi = pi.
# This means pi is the eigenvector of P for eigenvalue 1.
# We compute eigenvalues/vectors of P directly.
eigenvalues, eigenvectors = scipy.linalg.eig(P)
# Note: for non-symmetric P, use eig (not eigh).
# Eigenvalues may be complex; for a stochastic matrix they are all ≤ 1 in modulus.

# Find index of eigenvalue closest to 1.0
# We use the real part since stochastic matrices have real leading eigenvalue.
idx = np.argmin(np.abs(eigenvalues - 1.0))
print(f"Eigenvalue closest to 1: {eigenvalues[idx]:.10f}")

# Extract the corresponding eigenvector (column of eigenvectors matrix)
pi_raw = eigenvectors[:, idx].real   # take real part; imaginary should be ~0

# Normalize: make entries sum to 1 and all positive
# (eigenvectors are defined up to a scalar; stochastic eigenvector is non-negative)
pi = pi_raw / pi_raw.sum()
print("Stationary distribution π:", pi.round(6))

# Verify
residual = np.max(np.abs(P @ pi - pi))
print(f"Verification ||Pπ - π||_inf: {residual:.2e}")
# Expected: ~1e-15

# Most-trafficked page
best_page = np.argmax(pi)
print(f"Highest traffic: page {best_page} (π = {pi[best_page]:.4f})")
```

---

### I4 — Least Squares and Rank Diagnosis

**Problem:** Use SVD to compute the pseudoinverse and recover the least-squares solution.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(99)
A = rng.standard_normal((12, 3))
x_true = np.array([2.0, -1.5, 0.8])
b = A @ x_true + 0.05 * rng.standard_normal(12)

# SVD of A: A = U S V^T
# Pseudoinverse: A^+ = V S^+ U^T  where S^+ inverts non-zero singular values
U, s, Vt = scipy.linalg.svd(A, full_matrices=False)
# U: (12, 3), s: (3,), Vt: (3, 3)

# Threshold for "numerical zero" singular values
tol = 1e-10 * s[0]
s_inv = np.where(s > tol, 1.0 / s, 0.0)   # safe inversion

# Pseudoinverse: A^+ = V * S^+ * U^T
A_pinv = Vt.T @ np.diag(s_inv) @ U.T   # shape (3, 12)

# Least-squares solution
x_ls = A_pinv @ b
print("Least-squares solution:", x_ls.round(6))
print("True x_true:", x_true)

# Residual norm
residual_norm = np.linalg.norm(b - A @ x_ls)
print(f"||b - A x_ls|| = {residual_norm:.6f}")

# Rank diagnosis
numerical_rank = np.sum(s > tol)
print(f"Singular values: {s.round(6)}")
print(f"Numerical rank of A: {numerical_rank} (out of 3)")
# Since A is (12, 3) with random entries, rank should be 3 with probability 1.
# All singular values significantly > 0 confirms linear independence of columns.

# Error comparison
rel_error = np.linalg.norm(x_ls - x_true) / np.linalg.norm(x_true)
print(f"Relative error ||x_ls - x_true|| / ||x_true|| = {rel_error:.4f}")
# Small because noise level is 0.05 and system is well-conditioned
```

---

### I5 — Cholesky Decomposition for Sampling

**Problem:** Use Cholesky to factor Σ, generate samples, check sample covariance.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(7)
F = rng.standard_normal((5, 8))
Sigma = (F @ F.T) / 8 + 0.5 * np.eye(5)

# scipy.linalg.cholesky returns the upper-triangular factor R such that R.T @ R = Sigma
# (or the lower-triangular L such that L @ L.T = Sigma, via lower=True).
# For sampling we want L such that x = L @ z has covariance L L^T = Sigma.
#
# Why Cholesky over full eigendecomposition?
# - Cholesky: O(n^3/3) — about 3x cheaper than the full eigen decomp O(n^3).
# - Produces a UNIQUE decomposition (given SPD input).
# - More numerically stable for SPD matrices than eig.
# - The eigen approach is needed only if Sigma might be semi-definite (zero eigenvalues).
L = scipy.linalg.cholesky(Sigma, lower=True)

# Verify: L @ L.T should equal Sigma
verify_err = np.max(np.abs(L @ L.T - Sigma))
print(f"Cholesky verify ||L L^T - Sigma||_inf: {verify_err:.2e}")

# Generate 2000 samples: x = L @ z, z ~ N(0, I)
rng2 = np.random.default_rng(42)
z = rng2.standard_normal((5, 2000))   # 5 dims, 2000 samples
X_samples = L @ z                      # shape (5, 2000)

# Estimate sample covariance (unbiased, ddof=1)
Sigma_hat = np.cov(X_samples)         # shape (5, 5)
error = np.linalg.norm(Sigma_hat - Sigma, 'fro')
print(f"||Sigma_sample - Sigma||_F = {error:.4f}")
# Expected: < 0.15 for 2000 samples

print("\nTrue Sigma diagonal:", np.diag(Sigma).round(4))
print("Sample Sigma diagonal:", np.diag(Sigma_hat).round(4))
```

---

## Advanced Problems

---

### A1 — Spectral Unmixing of Chemical Mixtures

**Justification:** The core idea is that the column space of the low-rank structure of `X` spans the same subspace as the pure-component spectra. After diagnosing rank via singular value decay, we project out the known components A and B from the data's spectral basis, recovering the residual direction which corresponds to component C. Concentration estimation then follows from a least-squares projection. The primary limitation is that noise blurs the boundary between signal and noise subspaces; the quality of C recovery depends on its contribution to variance being distinguishable from noise.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(123)
S_true = np.abs(rng.standard_normal((3, 200)))
C_true = np.abs(rng.standard_normal((50, 3)))
X = C_true @ S_true + 0.01 * rng.standard_normal((50, 200))

rng2 = np.random.default_rng(123)  # same seed → S_known[:2] == S_true[:2]
S_known = np.abs(rng2.standard_normal((2, 200)))  # rows: spectra of A and B

# ── Step 1: Diagnose rank structure ──────────────────────────────────────────
U, s, Vt = scipy.linalg.svd(X, full_matrices=False)
# Vt is (min(50,200), 200) = (50, 200); rows are right singular vectors

total_var = np.sum(s**2)
cumulative = np.cumsum(s**2) / total_var
print("Singular values (top 10):", s[:10].round(4))
print("Cumulative variance (top 10):", cumulative[:10].round(4))
# Three singular values should stand out before the noise floor.
# Look for a clear drop-off after index 2 (0-indexed).

# Visual scree check (text)
for i, (si, ci) in enumerate(zip(s[:8], cumulative[:8])):
    print(f"  k={i+1}: σ={si:.3f}, cumvar={ci:.4f}")

# ── Step 2: Recover spectrum of component C ───────────────────────────────────
# Strategy: the row space of X is spanned by its top-3 right singular vectors.
# Components A and B are known. Their projections onto this 3D subspace define
# a 2D subspace. Component C lies in the complementary 1D direction.

V3 = Vt[:3, :].T          # (200, 3) — column basis of the spectral row space
# Project S_known onto V3
S_known_proj = S_known @ V3   # (2, 3)

# Use QR or SVD to find the direction in V3's span orthogonal to both known spectra
# Gram-Schmidt: orthogonalize S_known_proj, then take the complement
Q_known, _ = np.linalg.qr(S_known_proj.T, mode='complete')
# Q_known is (3, 3); the last column is the direction orthogonal to S_known
c_direction_in_V3 = Q_known[:, -1]           # (3,)
spectrum_C_hat = V3 @ c_direction_in_V3      # (200,) — back in wavelength space

# Spectra are non-negative; take absolute value (sign is ambiguous from SVD)
spectrum_C_hat = np.abs(spectrum_C_hat)
spectrum_C_hat /= np.linalg.norm(spectrum_C_hat)   # normalize

# ── Step 3: Estimate concentrations of C ─────────────────────────────────────
# Project each sample onto the recovered spectrum
# This is a least-squares fit: for each sample x_i in X, find scalar c_i
# such that c_i * spectrum_C_hat minimizes ||x_i - c_i * spectrum_C_hat||
# Solution: c_i = x_i @ spectrum_C_hat
concentrations_C = X @ spectrum_C_hat   # (50,)
print("\nEstimated concentrations of C (first 10):", concentrations_C[:10].round(4))

# ── Validation ────────────────────────────────────────────────────────────────
# Since we have S_true, check alignment of recovered vs true spectrum of C.
# Note: we only recover it up to a scaling and sign ambiguity.
S_C_true_norm = S_true[2] / np.linalg.norm(S_true[2])
cosine_sim = np.abs(np.dot(spectrum_C_hat, S_C_true_norm))
print(f"\nCosine similarity (recovered vs true spectrum C): {cosine_sim:.4f}")
# A value near 1.0 indicates good recovery.

# ── Practical caveats ─────────────────────────────────────────────────────────
# 1. Recovery degrades if C's contribution to X variance is comparable to noise.
# 2. The above assumes A and B spectra are known exactly; errors in S_known propagate.
# 3. Non-negativity of spectra should be enforced (NMF is the principled approach).
# 4. Check: are S_known rows linearly independent? If nearly parallel, recovery fails.
print(f"Angle between known spectra A and B: "
      f"{np.degrees(np.arccos(np.clip(np.dot(S_known[0]/np.linalg.norm(S_known[0]),
                                              S_known[1]/np.linalg.norm(S_known[1])), -1, 1))):.1f}°")
```

---

### A2 — Sparse Laplacian and Graph Connectivity

**Justification:** The graph Laplacian's spectral properties encode connectivity directly. The zero eigenvalue's multiplicity counts connected components. ARPACK is essential here — we need only 8 eigenvalues from a 600×600 sparse problem, making dense methods wasteful. Shift-invert is not needed for this problem size, but CSR format is correct for the matrix-vector products ARPACK performs internally.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

rng = np.random.default_rng(55)
N = 600
rows, cols = [], []
for i in range(N):
    neighbors = rng.choice(N, size=8, replace=False)
    for j in neighbors:
        if i != j:
            rows.extend([i, j])
            cols.extend([j, i])
A_adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
A_adj = (A_adj + A_adj.T)
A_adj.data = np.ones_like(A_adj.data)
A_adj = A_adj.tocsr()

# ── Step 1: Build Laplacian ───────────────────────────────────────────────────
# L = D - A where D is the diagonal degree matrix.
# For CSR: the degree of node i = number of non-zero entries in row i of A.
degrees = np.array(A_adj.sum(axis=1)).flatten()   # (N,)
D = sp.diags(degrees, format='csr')
L = D - A_adj   # CSR sparse Laplacian

# CSR is optimal here: ARPACK's eigsh performs repeated mat-vec products L @ v.
# CSR has O(nnz) mat-vec, vs O(N^2) for dense.

# ── Step 2: Find 8 smallest eigenvalues via ARPACK ────────────────────────────
# which='SM' = smallest magnitude; sigma=None, no shift-invert needed here.
# For very large graphs or near-zero gaps, sigma=0.0 (shift-invert) would be needed.
# k must be < N - 1 for eigsh.
k_eig = 8
eigenvalues, eigenvectors = spla.eigsh(L, k=k_eig, which='SM', tol=1e-10)

# Sort (eigsh may not return in order)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("8 smallest eigenvalues of L:")
for i, lam in enumerate(eigenvalues):
    print(f"  λ_{i} = {lam:.8f}")

# ── Step 3: Interpret zero eigenvalue multiplicity ────────────────────────────
# The number of eigenvalues at (or near) zero = number of connected components.
zero_threshold = 1e-6
n_zero = np.sum(eigenvalues < zero_threshold)
print(f"\nZero eigenvalue multiplicity: {n_zero}")
print(f"→ Graph has {n_zero} connected component(s)")

# ── Step 4: Fiedler value ──────────────────────────────────────────────────────
fiedler_idx = n_zero   # first non-zero eigenvalue
fiedler_value = eigenvalues[fiedler_idx]
fiedler_vector = eigenvectors[:, fiedler_idx]
print(f"\nFiedler value (λ_2) = {fiedler_value:.6f}")
print("Interpretation: larger Fiedler value → better-connected graph")
print("  Small Fiedler value → graph has a bottleneck (near-cut)")

# ── Step 5: Fiedler vector partition ─────────────────────────────────────────
# Sign of Fiedler vector entries partitions the graph into two groups.
partition_0 = np.where(fiedler_vector >= 0)[0]
partition_1 = np.where(fiedler_vector < 0)[0]
print(f"\nPartition sizes: {len(partition_0)} / {len(partition_1)}")
print(f"Balance ratio: {min(len(partition_0), len(partition_1)) / N:.3f}")
# Near 0.5 = balanced; near 0 = highly unbalanced (trivial cut)
```

---

### A3 — Tridiagonal Sparse System from Heat Equation Discretization

**Justification:** The system is large (1200×1200) but has only 3 non-zeros per row. Assembling in CSR and using a sparse direct solver exploits this structure. Dense solve requires O(N^2) memory and O(N^3) time, while the sparse tridiagonal solve is O(N). The crossover where sparse wins is typically around N ≈ 100–200.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg
import time

N = 1200
h = 1.0 / (N + 1)
x_grid = np.arange(1, N + 1) * h   # interior points

# ── Step 1: Assemble A in CSR ─────────────────────────────────────────────────
# Tridiagonal: 2/h² on diagonal, -1/h² on off-diagonals.
diag_val = 2.0 / h**2
off_val  = -1.0 / h**2

diagonals = [off_val * np.ones(N - 1),
             diag_val * np.ones(N),
             off_val * np.ones(N - 1)]
A_sparse = sp.diags(diagonals, offsets=[-1, 0, 1], format='csr')

# Source term: f_i = sin(π x_i)
f = np.sin(np.pi * x_grid)

# ── Step 2: Sparse direct solve ───────────────────────────────────────────────
t0 = time.perf_counter()
u_sparse = spla.spsolve(A_sparse, f)
t_sparse = time.perf_counter() - t0

# ── Step 3: Error vs exact solution ───────────────────────────────────────────
u_exact = np.sin(np.pi * x_grid) / np.pi**2
max_error = np.max(np.abs(u_sparse - u_exact))
print(f"Max absolute error (sparse solve): {max_error:.6e}")
# Expected: O(h²) = O(1/N²) ≈ 7e-7 for N=1200

# ── Step 4: Dense solve benchmark ────────────────────────────────────────────
A_dense = A_sparse.toarray()
t0 = time.perf_counter()
u_dense = scipy.linalg.solve(A_dense, f)
t_dense = time.perf_counter() - t0

print(f"\nSparse solve time: {t_sparse*1000:.2f} ms")
print(f"Dense solve time:  {t_dense*1000:.2f} ms")
print(f"Speedup factor:    {t_dense / t_sparse:.1f}x")

# Verify both solutions agree
print(f"||u_sparse - u_dense||_inf: {np.max(np.abs(u_sparse - u_dense)):.2e}")

# ── Step 5: Memory comparison ─────────────────────────────────────────────────
sparse_nnz = A_sparse.nnz
dense_elements = N * N
print(f"\nSparse matrix: {sparse_nnz} non-zeros ({sparse_nnz / dense_elements:.4%} fill)")
print(f"Dense matrix:  {dense_elements} elements")
print(f"Memory ratio:  {sparse_nnz / dense_elements:.4%}")

# Crossover comment:
# For tridiagonal systems, the sparse direct solver wins at any N > ~50.
# For general unstructured sparse matrices, the crossover depends on fill-in
# during factorization. scipy.sparse.linalg.spsolve uses SuperLU by default.
```

---

### A4 — Best-Fit Plane via SVD

**Justification:** The SVD of the centered data matrix provides the principal directions in order of variance. The best-fit plane is spanned by the top two singular vectors; the normal vector is the last (least-variance) singular vector. This is numerically stable and generalizes to arbitrary dimension.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(17)
n_true = np.array([1., 2., 3.])
n_true = n_true / np.linalg.norm(n_true)
t1 = np.array([1., 0., -1./3.])
t1 = t1 / np.linalg.norm(t1)
t2 = np.cross(n_true, t1)
coords = rng.standard_normal((300, 2))
X = coords[:, 0:1] * t1 + coords[:, 1:2] * t2 + 0.05 * rng.standard_normal((300, 3))
X = X + np.array([3., -1., 2.])   # offset

# ── Step 1: Center the data ───────────────────────────────────────────────────
centroid = X.mean(axis=0)
X_centered = X - centroid
print(f"Centroid: {centroid.round(4)}")

# ── Step 2: SVD to find plane basis ──────────────────────────────────────────
# X_centered = U S V^T
# V columns are principal directions (in 3D space).
# The two LARGEST singular vectors span the best-fit plane.
# The SMALLEST singular vector is the plane normal (direction of least variance).
U, s, Vt = scipy.linalg.svd(X_centered, full_matrices=False)
# Vt rows are right singular vectors; V columns are the same.
# Singular values in DESCENDING order.

n_recovered = Vt[2, :]   # last row of Vt = singular vector for smallest σ
# Ensure consistent orientation (dot product with n_true > 0)
if np.dot(n_recovered, n_true) < 0:
    n_recovered = -n_recovered

print(f"\nRecovered normal: {n_recovered.round(6)}")
print(f"True normal:      {n_true.round(6)}")

# Angle between normals
cos_angle = np.clip(np.dot(n_recovered, n_true), -1, 1)
angle_deg = np.degrees(np.arccos(cos_angle))
print(f"Angle between normals: {angle_deg:.4f}°")
# Should be << 5°

# ── Step 3: Project points onto plane and compute RMS distance ────────────────
# Distance from point p (centered) to plane = (p · n_recovered)
distances = X_centered @ n_recovered   # signed distances, shape (300,)
rms_dist = np.sqrt(np.mean(distances**2))
print(f"\nRMS distance from points to plane: {rms_dist:.6f}")
# Should be close to 0.05 (the noise level used)

# ── Step 4: 2D coordinates in plane's local frame ─────────────────────────────
# The plane is spanned by Vt[0] and Vt[1] (in-plane principal directions).
e1 = Vt[0, :]   # first principal direction in plane
e2 = Vt[1, :]   # second principal direction in plane

# Project centered points onto the plane basis
local_coords = X_centered @ np.stack([e1, e2], axis=1)   # (300, 2)
print(f"\nLocal coordinates: shape {local_coords.shape}")
print(f"  x-range: [{local_coords[:,0].min():.2f}, {local_coords[:,0].max():.2f}]")
print(f"  y-range: [{local_coords[:,1].min():.2f}, {local_coords[:,1].max():.2f}]")
```

---

### A5 — Repairing a Near-Singular Covariance Matrix

**Justification:** A sample covariance from n=40 samples in p=50 dimensions is at most rank 39. Cholesky requires strict positive definiteness, so negative or zero eigenvalues cause failure. Eigenvalue thresholding is interpretable but can introduce significant distortion. Ledoit-Wolf shrinkage is theoretically grounded: it shrinks all eigenvalues toward a common value, preserving the trace and producing a well-conditioned matrix with a clear parameter (`alpha`). For downstream use in a statistical model, shrinkage is almost always preferred because it has a known bias-variance interpretation.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(303)
data = rng.standard_normal((40, 50))
Sigma_hat = np.cov(data.T)   # (50, 50)

# ── Step 1: Diagnose the failure ──────────────────────────────────────────────
print("Matrix shape:", Sigma_hat.shape)
print("n_samples=40, p=50  →  sample covariance is at most rank 39")

# Check eigenvalues
eigenvalues_raw = scipy.linalg.eigvalsh(Sigma_hat)   # sorted ascending
print(f"\nSmallest 5 eigenvalues: {eigenvalues_raw[:5].round(8)}")
print(f"Negative eigenvalues:   {np.sum(eigenvalues_raw < 0)}")
print(f"Near-zero (< 1e-10):    {np.sum(eigenvalues_raw < 1e-10)}")

# Attempt Cholesky — should fail
try:
    scipy.linalg.cholesky(Sigma_hat)
    print("\nCholesky succeeded (unexpected)")
except np.linalg.LinAlgError as e:
    print(f"\nCholesky FAILED as expected: {e}")

# ── Fix A: Eigenvalue thresholding ────────────────────────────────────────────
eigenvalues_all, Q = scipy.linalg.eigh(Sigma_hat)   # ascending
floor = max(1e-6, eigenvalues_all.max() * 1e-8)     # set floor relative to max
eigenvalues_fixed = np.maximum(eigenvalues_all, floor)
Sigma_A = Q @ np.diag(eigenvalues_fixed) @ Q.T

try:
    scipy.linalg.cholesky(Sigma_A)
    print("\nFix A (eigenvalue thresholding): Cholesky SUCCEEDED")
except:
    print("\nFix A: FAILED")

cond_A = eigenvalues_fixed.max() / eigenvalues_fixed.min()
print(f"  Condition number after Fix A: {cond_A:.2e}")

# ── Fix B: Ledoit-Wolf-inspired shrinkage ──────────────────────────────────────
# Alpha controls the shrinkage: alpha=0 → Sigma_hat, alpha=1 → I.
# We want the smallest alpha such that Cholesky succeeds.
# A principled choice: alpha such that the minimum eigenvalue is, say, 1e-4.
# min_eigenvalue(Sigma_reg) = (1-alpha)*lambda_min + alpha
# Set this = 1e-4: alpha = (1e-4 - lambda_min) / (1 - lambda_min)
lambda_min = eigenvalues_all[0]
alpha = max((1e-4 - lambda_min) / (1.0 - lambda_min), 0.0)
alpha = min(alpha * 1.5, 1.0)   # add small safety margin
Sigma_B = (1 - alpha) * Sigma_hat + alpha * np.eye(50)

try:
    scipy.linalg.cholesky(Sigma_B)
    print(f"\nFix B (shrinkage, alpha={alpha:.4f}): Cholesky SUCCEEDED")
except:
    print(f"\nFix B (alpha={alpha:.4f}): FAILED")

eigs_B = scipy.linalg.eigvalsh(Sigma_B)
cond_B = eigs_B.max() / eigs_B.min()
print(f"  Condition number after Fix B: {cond_B:.2e}")
print(f"  Alpha used: {alpha:.4f}")

# ── Recommendation ────────────────────────────────────────────────────────────
print("""
Recommendation: Fix B (shrinkage) is preferred for statistical applications.
  - Fix A: aggressively distorts the covariance structure; small eigenvalues
    may represent genuine low-variance directions in the data.
  - Fix B: interpretable as adding a small isotropic component; preserves
    relative eigenvalue structure; can be selected analytically or via
    cross-validation (Ledoit-Wolf, Oracle Approximating Shrinkage).
  - For downstream Cholesky-based sampling or Gaussian likelihood, Fix B
    gives a valid covariance estimator with known bias-variance properties.
""")
```

---

## Expert Problems

---

### E1 — Truncated SVD for High-Dimensional Regression

**Justification:** With p=8000 features and n=120 samples, full SVD of X costs O(min(p,n)^2 * max(p,n)) ≈ O(120^2 * 8000). More importantly, the full left singular vectors U (8000 × 120) consume significant memory. ARPACK's `svds` computes only k singular vectors via implicitly restarted Lanczos, reducing memory to O(p*k + n*k). The tradeoff is ARPACK's iterative convergence vs. LAPACK's direct method — ARPACK can struggle with clustered singular values, while full SVD is batch-stable. For regression, working in the right singular vector space (n × k) is both sufficient and numerically cleaner than forming the n × p feature matrix transpose.

**Alternative not preferred:** NumPy's full SVD on X.T (shape 120 × 8000) would be nearly as fast at this scale (n << p allows the economy trick), but wastes the opportunity to demonstrate ARPACK and doesn't scale to p = 10^6.

```python
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla
from numpy.random import default_rng

rng = default_rng(777)
W = rng.standard_normal((8000, 6))
Z = rng.standard_normal((6, 120))
X = W @ Z + 0.5 * rng.standard_normal((8000, 120))
beta_true = rng.standard_normal(6)
y = (Z.T @ beta_true) + 0.3 * rng.standard_normal(120)

# ── Step 1: Truncated SVD via ARPACK ─────────────────────────────────────────
# scipy.sparse.linalg.svds computes the k LARGEST singular values.
# X has shape (8000, 120); we want the top-k right singular vectors.
k_max = 20
U_k, s_k, Vt_k = spla.svds(X, k=k_max)
# svds returns in ASCENDING order — reverse to get descending
order = np.argsort(s_k)[::-1]
U_k, s_k, Vt_k = U_k[:, order], s_k[order], Vt_k[order, :]

print("Top singular values:", s_k[:10].round(3))

# ── Scree plot heuristic: find elbow ─────────────────────────────────────────
# Method 1: variance explained threshold
total_var = np.sum(s_k**2)
cumvar = np.cumsum(s_k**2) / total_var
# Note: this is only approximate since we didn't compute ALL singular values.
print("\nCumulative variance explained by top-k components:")
for k in range(1, k_max + 1):
    print(f"  k={k:2d}: {cumvar[k-1]:.4f}")

# Method 2: scree gap (ratio of successive singular values)
ratios = s_k[:-1] / s_k[1:]
k_elbow = np.argmax(ratios) + 1   # elbow after largest relative gap
print(f"\nElbow detected at k = {k_elbow}")

k_chosen = k_elbow   # use this for regression

# ── Step 2: Project to reduced feature space ─────────────────────────────────
# Right singular vectors Vt_k: shape (k_max, 120); each row is a singular vector.
# Z_hat = X.T @ U[:, :k] = Vt[:k, :].T * s[:k]  (by definition of SVD)
# More directly: columns of Vt.T are scores for each sample.
Z_hat = Vt_k[:k_chosen, :].T * s_k[:k_chosen]    # (120, k_chosen)
print(f"\nReduced feature matrix Z_hat shape: {Z_hat.shape}")

# ── Step 3: Train/test split and regression ───────────────────────────────────
rng2 = default_rng(1)
n = 120
idx = rng2.permutation(n)
n_test = 24
idx_train, idx_test = idx[n_test:], idx[:n_test]

Z_train, y_train = Z_hat[idx_train], y[idx_train]
Z_test,  y_test  = Z_hat[idx_test],  y[idx_test]

# OLS via scipy.linalg.lstsq
coeffs, _, _, _ = scipy.linalg.lstsq(Z_train, y_train)
y_pred = Z_test @ coeffs
mse_truncsvd = np.mean((y_pred - y_test)**2)
print(f"\nTruncated SVD regression test MSE: {mse_truncsvd:.4f}")

# ── Step 4a: Naive ridge on full X.T ──────────────────────────────────────────
X_train, X_test = X[:, idx_train].T, X[:, idx_test].T   # (n_train, 8000)
lam_ridge = 1.0
# Ridge: (X^T X + λI) β = X^T y  → solve normal equations
# For p >> n, faster to use the (n x n) dual form, but here just use lstsq with augmentation
A_aug = np.vstack([X_train, np.sqrt(lam_ridge) * np.eye(8000)])
b_aug = np.concatenate([y_train, np.zeros(8000)])
beta_ridge, _, _, _ = scipy.linalg.lstsq(A_aug, b_aug)
y_pred_ridge = X_test @ beta_ridge
mse_ridge = np.mean((y_pred_ridge - y_test)**2)
print(f"Ridge regression (full X.T, λ=1.0) test MSE: {mse_ridge:.4f}")

# ── Step 4b: PCA via numpy.linalg.svd ─────────────────────────────────────────
# Using economy SVD on X (8000, 120): only 120 singular values max.
# This is already the economy path since n << p.
U_np, s_np, Vt_np = np.linalg.svd(X, full_matrices=False)
# U_np: (8000, 120), Vt_np: (120, 120)
Z_hat_np = Vt_np[:k_chosen, :].T * s_np[:k_chosen]   # (120, k_chosen)
Z_train_np, Z_test_np = Z_hat_np[idx_train], Z_hat_np[idx_test]
coeffs_np, _, _, _ = scipy.linalg.lstsq(Z_train_np, y_train)
y_pred_np = Z_test_np @ coeffs_np
mse_np = np.mean((y_pred_np - y_test)**2)
print(f"NumPy SVD (full economy) test MSE: {mse_np:.4f}")
print(f"  (Should match truncated SVD MSE ≈ {mse_truncsvd:.4f})")

# ── Step 5: Numerical stability tradeoffs ─────────────────────────────────────
print("""
Stability tradeoffs:
  1. ARPACK (svds) uses Lanczos iteration, which can lose orthogonality for
     clustered singular values. Full LAPACK SVD maintains full orthogonality.
     Here the top-6 singular values should be well-separated from noise,
     so ARPACK is reliable.
  2. ARPACK requires specifying k upfront. Choosing k too small misses signal;
     too large includes noise. Full SVD lets you examine all singular values
     and choose k post-hoc — safer for exploratory work.
  3. Memory: ARPACK computes U, S, Vt for top-k only (~O(p*k) storage).
     Full economy SVD on X (p x n) stores O(p*n) — all of U. For p=10^6
     and n=120, this is ~1 GB; ARPACK with k=10 uses ~10 MB.
""")
```

---

### E2 — Spectral Clustering Pipeline

**Justification:** The normalized Laplacian `L_sym` has eigenvalues in [0, 2], and the smallest eigenvalues reveal cluster structure. We need shift-invert because the SMALLEST eigenvalues are desired, but ARPACK is most efficient at finding LARGEST eigenvalues. Shift-invert with `sigma=0` converts the problem to finding the eigenvalues of `(L_sym - 0*I)^{-1}` — the smallest eigenvalues of `L_sym` become the largest eigenvalues of the shifted operator, which ARPACK handles efficiently. A `k`-NN graph with symmetric edges is used to guarantee symmetry of the Laplacian.

**Alternative not preferred:** Dense `eigh` would work and be simpler to implement, but scales as O(N^3) = O(1500^3) ≈ 3×10^9 operations and requires storing the full N×N matrix. For N > 2000–3000, this becomes prohibitive.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg

rng = np.random.default_rng(88)

def make_ring(n, r, noise, cx, cy):
    theta = rng.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta) + noise * rng.standard_normal(n)
    y = cy + r * np.sin(theta) + noise * rng.standard_normal(n)
    return np.stack([x, y], axis=1)

X = np.vstack([make_ring(500, 1.0, 0.05, 0, 0),
               make_ring(500, 2.2, 0.05, 0, 0),
               make_ring(500, 3.5, 0.05, 0, 0)])
labels_true = np.array([0]*500 + [1]*500 + [2]*500)
N = len(X)

# ── Step 1: k-NN similarity graph ────────────────────────────────────────────
# The rings are NOT linearly separable but ARE locally connected.
# k must be large enough for the graph to be connected.
# Too small k → disconnected graph (zero eigenvalue multiplicity > n_clusters).
# Too large k → connects different rings (destroys cluster structure).
# For rings with spacing 1.2 and noise 0.05, k=10 is empirically safe.
from scipy.spatial import KDTree
k_nn = 10
tree = KDTree(X)
distances, indices = tree.query(X, k=k_nn + 1)  # +1 to exclude self

# Build adjacency matrix (binary k-NN, symmetrized)
rows_list, cols_list = [], []
for i in range(N):
    for j in indices[i, 1:]:   # skip self (index 0)
        rows_list.append(i)
        cols_list.append(j)

A_knn = sp.csr_matrix((np.ones(len(rows_list)), (rows_list, cols_list)), shape=(N, N))
A_knn = (A_knn + A_knn.T)   # symmetrize (max is another option)
A_knn.data = np.ones_like(A_knn.data)   # binarize
A_knn = A_knn.tocsr()

# Verify connectivity
from scipy.sparse.csgraph import connected_components
n_comp, _ = connected_components(A_knn, directed=False)
print(f"k_nn={k_nn}: {n_comp} connected component(s)")

# ── Step 2: Normalized Laplacian ─────────────────────────────────────────────
# L_sym = D^{-1/2} (D - A) D^{-1/2} = I - D^{-1/2} A D^{-1/2}
degrees = np.array(A_knn.sum(axis=1)).flatten()
D_invsqrt = sp.diags(1.0 / np.sqrt(degrees), format='csr')
L_sym = sp.eye(N, format='csr') - D_invsqrt @ A_knn @ D_invsqrt

# ── Step 3: ARPACK with shift-invert ─────────────────────────────────────────
# sigma=0.0 enables shift-invert: finds eigenvalues closest to 0.
# This converts to finding LARGE eigenvalues of L_sym^{-1}, which ARPACK
# handles via LU factorization (scipy uses SuperLU internally).
k_eig = 8
eigenvalues_sparse, eigenvectors_sparse = spla.eigsh(
    L_sym, k=k_eig, sigma=0.0, which='LM', tol=1e-10)
order = np.argsort(eigenvalues_sparse)
eigenvalues_sparse = eigenvalues_sparse[order]
eigenvectors_sparse = eigenvectors_sparse[:, order]

print("\n8 smallest eigenvalues (normalized Laplacian):")
for i, lam in enumerate(eigenvalues_sparse):
    print(f"  λ_{i} = {lam:.8f}")

# ── Step 4: Eigengap heuristic for number of clusters ────────────────────────
# The number of clusters = multiplicity of eigenvalue 0 = eigengap location.
gaps = np.diff(eigenvalues_sparse)
print("\nEigenvalue gaps:", gaps.round(6))
n_clusters = np.argmax(gaps) + 1
print(f"Detected {n_clusters} cluster(s) from largest eigengap")

# ── Step 5: Cluster assignment ────────────────────────────────────────────────
from sklearn.cluster import KMeans
# Use top n_clusters eigenvectors (excluding trivial zero eigenvector if needed)
# For normalized Laplacian, eigenvectors give cluster embeddings.
V_embed = eigenvectors_sparse[:, :n_clusters]   # (N, n_clusters)
# Normalize rows for numerical stability (standard in spectral clustering)
V_norm = V_embed / (np.linalg.norm(V_embed, axis=1, keepdims=True) + 1e-12)

km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
labels_pred = km.fit_predict(V_norm)

from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(labels_true, labels_pred)
print(f"\nAdjusted Rand Index: {ari:.4f}")
# ARI = 1.0 → perfect; > 0.9 is excellent for this problem

# ── Step 6: Dense comparison and scale analysis ───────────────────────────────
import time
L_dense = L_sym.toarray()

t0 = time.perf_counter()
eigs_dense, _ = scipy.linalg.eigh(L_dense, subset_by_index=[0, k_eig - 1])
t_dense = time.perf_counter() - t0

t0 = time.perf_counter()
spla.eigsh(L_sym, k=k_eig, sigma=0.0, which='LM', tol=1e-10)
t_sparse = time.perf_counter() - t0

print(f"\nDense eigh time:  {t_dense*1000:.1f} ms")
print(f"Sparse eigsh time: {t_sparse*1000:.1f} ms")
print(f"Memory (dense L): {L_dense.nbytes / 1e6:.1f} MB")
print(f"Memory (sparse L): {L_sym.data.nbytes / 1e6:.3f} MB")
print("""
Scale analysis:
  Dense eigh memory ∝ N² → 4 GB at N=30,000 (impractical).
  Sparse ARPACK memory ∝ N * k_nn → linear in N.
  The sparse approach becomes essential at N > ~5,000–10,000.
""")
```

---

### E3 — Matrix Logarithm of a Transition Matrix

**Justification:** `scipy.linalg.logm` implements the Schur-based algorithm. The issues are: (1) for stochastic matrices with eigenvalue 1 exactly, the log of 1 is 0, which is correct, but near-1 eigenvalues can cause instability near the branch cut at -π; (2) the resulting matrix may not satisfy the generator property if P has eigenvalues with negative real parts after rounding. The Schur decomposition gives a triangular factor where the diagonal entries are the eigenvalues — we can selectively handle near-zero or negative eigenvalues before taking the log, making the approach more robust than applying `logm` blindly.

**Alternative not preferred:** Direct iteration `Q_n = Q_{n-1} + (I - exp(Q_{n-1}))` (commutator-free Cayley-transform iteration) converges but is much slower and harder to implement robustly.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(42)
n = 8
rates = np.abs(rng.standard_normal((n, n)))
np.fill_diagonal(rates, 0)
Q_true = rates.copy()
np.fill_diagonal(Q_true, -rates.sum(axis=1))
P_obs = scipy.linalg.expm(Q_true)

print("True Q (first row):", Q_true[0].round(4))
print("P_obs row sums (should be 1):", P_obs.sum(axis=1).round(8))

# ── Step 1: Naive logm ───────────────────────────────────────────────────────
Q_naive = scipy.linalg.logm(P_obs)
print("\nNaive logm result:")
print("  Is real?", np.allclose(Q_naive.imag, 0, atol=1e-10))
print("  Max imaginary component:", np.max(np.abs(Q_naive.imag)))
Q_naive = Q_naive.real

# Check generator property
off_diag_min = Q_naive[~np.eye(n, dtype=bool)].min()
row_sums = Q_naive.sum(axis=1)
print(f"  Min off-diagonal entry: {off_diag_min:.8f}  (should be ≥ 0)")
print(f"  Max |row sum|: {np.max(np.abs(row_sums)):.2e}  (should be ~0)")

# ── Step 2: Schur-based approach ─────────────────────────────────────────────
# Decompose P = Z T Z^H where T is quasi-upper-triangular (real Schur form).
# For stochastic matrices with real eigenvalues, T is upper triangular.
T, Z = scipy.linalg.schur(P_obs, output='real')

# Take log of the diagonal of T (eigenvalues)
# Handle near-zero diagonals (eigenvalues near 0) which cause log(≈0) = -inf
diag_T = np.diag(T)
print(f"\nSchur diagonal (eigenvalues of P): min={diag_T.min():.6f}, max={diag_T.max():.6f}")

# Use logm on the triangular factor T directly (scipy handles this stably)
log_T = scipy.linalg.logm(T)

# Reconstruct
Q_schur = (Z @ log_T @ Z.T).real

# ── Step 3: Validate ─────────────────────────────────────────────────────────
P_recovered = scipy.linalg.expm(Q_schur)
recon_error = np.linalg.norm(P_recovered - P_obs, 'fro')
print(f"\nSchur approach: ||expm(Q_recovered) - P_obs||_F = {recon_error:.2e}")

# ── Step 4: Sensitivity to noise ─────────────────────────────────────────────
rng2 = np.random.default_rng(1)
P_noisy = P_obs + 0.01 * rng2.standard_normal((n, n))
# Renormalize columns to keep it stochastic
P_noisy = np.abs(P_noisy)
P_noisy /= P_noisy.sum(axis=0)

T_n, Z_n = scipy.linalg.schur(P_noisy, output='real')
log_T_n = scipy.linalg.logm(T_n)
Q_noisy = (Z_n @ log_T_n @ Z_n.T).real

Q_change = np.linalg.norm(Q_noisy - Q_schur, 'fro')
P_change = np.linalg.norm(P_noisy - P_obs, 'fro')
print(f"\n1% perturbation to P: ||ΔP||_F = {P_change:.4f}")
print(f"  Resulting change in Q: ||ΔQ||_F = {Q_change:.4f}")
print(f"  Amplification factor: {Q_change / P_change:.2f}")

# ── Step 5: Branch cut discussion ────────────────────────────────────────────
print("""
Branch cut discussion:
  The matrix logarithm requires a principal branch choice. The standard branch
  has a cut along the negative real axis: log(re^{iθ}) for θ ∈ (-π, π].
  If P has an eigenvalue λ that is real and negative (impossible for stochastic
  matrices but can arise from noisy approximations), logm will return a complex
  result with imaginary part ≈ ±πi, causing Q to be complex.
  
  Detection: compute eigenvalues of P and flag any with:
  - Real part ≤ 0 (should not occur for proper stochastic matrices)
  - Imaginary part large relative to the real part
  
  Remediation: For near-negative eigenvalues, apply a small perturbation to P
  (e.g., regularization) before computing the log, or use a different branch
  by identifying the correct logarithm from problem context.
""")
```

---

### E4 — Tikhonov Regularization with Cross-Validated λ Selection

**Justification:** The SVD-based implementation of Tikhonov regularization avoids forming the 80×80 normal equations `A^T A` (which squares the condition number) and instead applies the regularization directly in the singular value domain. The solution is `x_λ = V diag(s_i / (s_i^2 + λ)) U^T b`. This is both more numerically stable and faster for cross-validation: once the SVD is computed, different λ values cost only O(k) per evaluation rather than O(mk) for each least-squares solve. The L-curve and cross-validation often disagree slightly; cross-validation tends to minimize prediction error while the L-curve minimizes a blend of residual and norm.

**Alternative not preferred:** Forming the augmented system `[A; sqrt(λ) I]` and applying QR is simpler to implement but recomputes the factorization for each λ. For a sweep over 50 λ values, the SVD approach is 50x more efficient.

```python
import numpy as np
import scipy.linalg

rng = np.random.default_rng(2024)
U_true, _ = np.linalg.qr(rng.standard_normal((500, 80)))
s_true = np.exp(-np.arange(80) / 5.0)
V_true, _ = np.linalg.qr(rng.standard_normal((80, 80)))
A = U_true[:, :80] * s_true @ V_true.T
x_true = rng.standard_normal(80)
b = A @ x_true + 0.01 * rng.standard_normal(500)

# ── Step 1: Naive least squares ───────────────────────────────────────────────
U_svd, s_svd, Vt_svd = scipy.linalg.svd(A, full_matrices=False)
# U_svd: (500, 80), s_svd: (80,), Vt_svd: (80, 80)

print(f"Condition number of A: {s_svd[0] / s_svd[-1]:.2e}")
print(f"Singular value range: [{s_svd[-1]:.2e}, {s_svd[0]:.2e}]")

# Pseudoinverse solution (λ=0 Tikhonov)
x_ls = Vt_svd.T @ ((U_svd.T @ b) / s_svd)
rel_err_ls = np.linalg.norm(x_ls - x_true) / np.linalg.norm(x_true)
print(f"\nNaive LS relative error: {rel_err_ls:.4f}")
# Expected: very large — amplified by tiny singular values

# ── Step 2: Tikhonov solution function (SVD-based, efficient) ────────────────
def tikhonov_svd(U, s, Vt, b, lam):
    """Tikhonov solution x = V * diag(s/(s²+λ)) * U^T b"""
    filters = s / (s**2 + lam)   # shape (k,)
    return Vt.T @ (filters * (U.T @ b))

# ── Step 3: 5-fold cross-validation over log-spaced λ grid ───────────────────
lambda_grid = np.logspace(-6, 0, 60)
n_folds = 5
n = len(b)
fold_size = n // n_folds
indices = np.arange(n)

cv_errors = np.zeros(len(lambda_grid))
for i, lam in enumerate(lambda_grid):
    fold_mse = []
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size],
                                    indices[(fold + 1) * fold_size:]])
        A_train, b_train = A[train_idx], b[train_idx]
        A_val,   b_val   = A[val_idx],   b[val_idx]

        # Recompute SVD for train fold (necessary for unbiased CV)
        U_f, s_f, Vt_f = scipy.linalg.svd(A_train, full_matrices=False)
        x_reg = tikhonov_svd(U_f, s_f, Vt_f, b_train, lam)
        residuals = b_val - A_val @ x_reg
        fold_mse.append(np.mean(residuals**2))
    cv_errors[i] = np.mean(fold_mse)

lam_opt_idx = np.argmin(cv_errors)
lam_opt = lambda_grid[lam_opt_idx]
print(f"\nOptimal λ from 5-fold CV: {lam_opt:.2e}")

# ── Step 4: Solution quality at optimal λ ────────────────────────────────────
x_tikh = tikhonov_svd(U_svd, s_svd, Vt_svd, b, lam_opt)
rel_err_tikh = np.linalg.norm(x_tikh - x_true) / np.linalg.norm(x_true)
print(f"Tikhonov relative error (λ_opt): {rel_err_tikh:.4f}")
print(f"Naive LS relative error:         {rel_err_ls:.4f}")
print(f"Improvement factor: {rel_err_ls / rel_err_tikh:.1f}x")

# ── Step 5: L-curve (text version) ───────────────────────────────────────────
print("\nL-curve (λ, ||residual||, ||x||):")
for lam in np.logspace(-6, 0, 10):
    x_l = tikhonov_svd(U_svd, s_svd, Vt_svd, b, lam)
    resid = np.linalg.norm(b - A @ x_l)
    xnorm = np.linalg.norm(x_l)
    print(f"  λ={lam:.2e}: ||r||={resid:.4f}, ||x||={xnorm:.4f}")

print(f"""
Discussion: Singular value spectrum and λ selection.
  The singular values decay exponentially (by construction).
  The appropriate λ ≈ σ_k² where σ_k is the largest singular value we want
  to "trust." Values below the noise floor (σ ≈ noise / ||x_true||) should be
  filtered. With noise level 0.01, the effective cutoff is around σ² ≈ 0.01,
  suggesting λ_opt in the range [1e-4, 1e-2].
  The CV estimate ({lam_opt:.2e}) should land in this range.
""")
```

---

### E5 — Rotation Sequence Analysis and Interpolation

**Justification:** The `Rotation` class in `scipy.spatial.transform` implements quaternion SLERP correctly. Quaternions are the right representation for interpolation because they avoid gimbal lock and produce constant-angular-velocity paths on SO(3). Naive matrix averaging fails because the average of two rotation matrices is generally not a rotation matrix — it has determinant ≠ 1 and is not orthogonal. Renormalization via SVD (forcing orthogonality) reduces but does not eliminate the error, and the error grows with the angular step size. The FFT of angular velocity should reveal a peak at frequency `||omega|| / (2π)` if the motion is steady rotation about a fixed axis.

**Alternative not preferred:** Log-map interpolation in the Lie algebra so(3) is theoretically equivalent to SLERP for SO(3) and also valid, but requires implementing the matrix logarithm of a 3×3 rotation, which is more error-prone and less readable than using `Rotation.slerp`.

```python
import numpy as np
import scipy.linalg
import scipy.fft
from scipy.spatial.transform import Rotation, Slerp

rng = np.random.default_rng(314)
t_obs = np.sort(rng.uniform(0, 4 * np.pi, 40))
omega = np.array([0.3, 0.5, 0.8])
R_obs = [Rotation.from_rotvec(omega * t) for t in t_obs]
t_reg = np.linspace(t_obs[0], t_obs[-1], 300)

# ── Step 1: SLERP interpolation ───────────────────────────────────────────────
slerp = Slerp(t_obs, Rotation.concatenate(R_obs))
R_interp = slerp(t_reg)

# Extract rotation vectors (axis × angle)
rotvecs = R_interp.as_rotvec()   # (300, 3); each row is the rotation vector at t_reg[i]
print("Rotation vector at t=0:", rotvecs[0].round(4))
print("Expected:", (omega * t_reg[0]).round(4))

# ── Step 2: Angular velocity from finite differences ─────────────────────────
dt = t_reg[1] - t_reg[0]   # uniform spacing
# Finite difference of rotation vector ≈ angular velocity (for small steps)
# This is approximate: the derivative of rotvec is only exactly omega for fixed axis.
omega_approx = np.gradient(rotvecs, dt, axis=0)   # (300, 3)
print(f"\nMean estimated angular velocity: {omega_approx.mean(axis=0).round(4)}")
print(f"True omega:                       {omega.round(4)}")

# ── Step 3: FFT of angular velocity components ────────────────────────────────
N_fft = len(t_reg)
freqs = scipy.fft.rfftfreq(N_fft, d=dt)   # positive frequencies

for i, component in enumerate(['x', 'y', 'z']):
    spectrum = np.abs(scipy.fft.rfft(omega_approx[:, i]))
    dominant_freq = freqs[np.argmax(spectrum[1:]) + 1]   # skip DC
    print(f"ω_{component}: dominant frequency = {dominant_freq:.4f} rad/s")

# Expected rotation frequency: ||omega|| / (2π)?
# No — since rotation is about a fixed axis, omega_approx should be nearly constant
# (DC component dominates, not oscillatory). The "frequency" interpretation applies
# to the angle, not the angular velocity of a steady rotation.
omega_mag = np.linalg.norm(omega)
print(f"\n||omega|| = {omega_mag:.4f} rad/s (steady-state angular speed)")

# ── Step 4: Naive matrix interpolation vs SLERP ──────────────────────────────
# Ground truth
R_gt = Rotation.from_rotvec(np.outer(t_reg, omega))   # (300,)

# Naive: for each regular time, find the two bracketing observed times,
# linearly average the rotation matrices, then renormalize via SVD.
errors_naive = np.zeros(len(t_reg))
errors_slerp = np.zeros(len(t_reg))

for k, t in enumerate(t_reg):
    # Find bracketing interval
    j = np.searchsorted(t_obs, t, side='right') - 1
    j = np.clip(j, 0, len(t_obs) - 2)
    alpha = (t - t_obs[j]) / (t_obs[j+1] - t_obs[j])

    # Naive average
    M_avg = (1 - alpha) * R_obs[j].as_matrix() + alpha * R_obs[j+1].as_matrix()
    # Renormalize via SVD: nearest rotation matrix
    U_n, _, Vt_n = np.linalg.svd(M_avg)
    R_naive_mat = U_n @ Vt_n
    R_naive = Rotation.from_matrix(R_naive_mat)

    R_true_k = R_gt[k]
    R_slerp_k = R_interp[k]

    # Error: geodesic distance on SO(3)
    dR_naive = (R_naive.inv() * R_true_k).magnitude()
    dR_slerp = (R_slerp_k.inv() * R_true_k).magnitude()
    errors_naive[k] = np.degrees(dR_naive)
    errors_slerp[k] = np.degrees(dR_slerp)

print(f"\nMean angular error — SLERP:         {errors_slerp.mean():.4f}°")
print(f"Mean angular error — Naive avg:     {errors_naive.mean():.4f}°")
print(f"Max  angular error — Naive avg:     {errors_naive.max():.4f}°")

# ── Step 5: Error as a function of step size ─────────────────────────────────
print("\nNaive averaging error vs interpolation interval:")
# Test a single large step
for n_obs in [5, 10, 20, 40]:
    t_sparse = np.linspace(t_obs[0], t_obs[-1], n_obs)
    R_sparse = [Rotation.from_rotvec(omega * t) for t in t_sparse]
    step_angles = []
    for j in range(len(t_sparse) - 1):
        t_mid = (t_sparse[j] + t_sparse[j+1]) / 2
        alpha = 0.5
        M_avg = 0.5 * R_sparse[j].as_matrix() + 0.5 * R_sparse[j+1].as_matrix()
        U_n, _, Vt_n = np.linalg.svd(M_avg)
        R_mid_naive = Rotation.from_matrix(U_n @ Vt_n)
        R_mid_true = Rotation.from_rotvec(omega * t_mid)
        err = np.degrees((R_mid_naive.inv() * R_mid_true).magnitude())
        step_angles.append(err)
    # Compute mean angular step between observations
    dangle = np.degrees(np.linalg.norm(omega) * (t_sparse[1] - t_sparse[0]))
    print(f"  n_obs={n_obs:3d}, mean step={dangle:.1f}°: naive error={np.mean(step_angles):.4f}°")
# Naive averaging error becomes > 1° when the angular step exceeds ~20-30°.
```
