# SciPy Mastery Curriculum
## Principle 1 — Linear Algebra & Matrix Structure
### Problem Set

**Modules:** `scipy.linalg` (primary), `scipy.sparse` (supporting), `scipy.sparse.linalg` with ARPACK (supporting)

---

## Beginner Problems

*One module, one function, prescribed setup. Function names are provided. The primary learning target is the API.*

---

### B1 — Solving a Linear System

The following 4×4 linear system arises from a network flow problem:

```python
import numpy as np

A = np.array([[ 4., -1.,  0.,  1.],
              [-1.,  4., -1.,  0.],
              [ 0., -1.,  4., -1.],
              [ 1.,  0., -1.,  3.]], dtype=float)

b = np.array([7., 3., 8., 5.], dtype=float)
```

Use `scipy.linalg.solve` to find the vector `x` such that `Ax = b`. After computing `x`, verify your result by computing the residual `r = b - A @ x` and reporting its infinity norm. A correct solution should give `||r||_∞ < 1e-12`.

---

### B2 — LU Factorization for Multiple Right-Hand Sides

Use the same matrix `A` from **B1**. You now need to solve the system for three different right-hand side vectors:

```python
b1 = np.array([1., 0., 0., 0.], dtype=float)
b2 = np.array([0., 1., 0., 0.], dtype=float)
b3 = np.array([2., -1., 3., 1.], dtype=float)
```

Use `scipy.linalg.lu_factor` to factor `A` **once**, then call `scipy.linalg.lu_solve` three times to recover `x1`, `x2`, and `x3`. Print each solution. Explain in a comment why this approach is more efficient than calling `scipy.linalg.solve` three separate times.

---

### B3 — Singular Value Decomposition

Compute the full SVD of the following 5×3 matrix using `scipy.linalg.svd`:

```python
M = np.array([[ 1.,  2.,  3.],
              [ 4.,  5.,  6.],
              [ 7.,  8.,  9.],
              [10., 11., 12.],
              [ 2.,  4.,  1.]], dtype=float)
```

Your solution must:
1. Report all three singular values.
2. Verify reconstruction: compute `U @ np.diag(s) @ Vt` and confirm it matches `M` to within `1e-10` in the Frobenius norm.
3. Identify whether `M` is rank-deficient based on the singular values.

Use the `full_matrices=False` (economy/thin) option and explain in a comment what changes about the shapes of `U`, `s`, and `Vt`.

---

### B4 — Eigendecomposition of a Symmetric Matrix

The following 4×4 matrix is symmetric positive definite:

```python
A = np.array([[5., 2., 1., 0.],
              [2., 6., 2., 1.],
              [1., 2., 7., 2.],
              [0., 1., 2., 5.]], dtype=float)
```

Use `scipy.linalg.eigh` (not `eig`) to compute the eigenvalues and eigenvectors. Your solution must:
1. Report the eigenvalues in ascending order (note what `eigh` gives you by default).
2. Verify orthonormality: confirm `Q.T @ Q ≈ I` to within `1e-12`.
3. Verify reconstruction: confirm `Q @ np.diag(lam) @ Q.T ≈ A` to within `1e-10`.

In a comment, explain why `eigh` is preferred over `eig` for symmetric matrices.

---

### B5 — Matrix Exponential

The following 3×3 matrix is the **generator matrix** of a continuous-time Markov chain. Each row sums to zero, which is the defining property of a generator:

```python
import numpy as np

Q = np.array([[-3.,  2.,  1.],
              [ 1., -4.,  3.],
              [ 2.,  1., -3.]], dtype=float)
```

Use `scipy.linalg.expm` to compute the transition matrix `P(t) = expm(t * Q)` at `t = 0.5` and `t = 2.0`.

Verify that `P(t)` is a valid stochastic matrix at both time points by confirming:
1. All entries are non-negative.
2. Each row sums to 1.

Report the probability of being in state 2 at time `t = 2.0`, given that the chain starts in state 0.

---

## Intermediate Problems

*Two or more modules in tension. Function names are not provided. Understanding what is being computed — not just how — is required.*

---

### I1 — Conditioning and the Trustworthiness of a Solution

A structural engineer has assembled a 6×6 stiffness matrix `K` for a small frame structure. Due to a data entry error, the matrix is slightly asymmetric, but you must use it as-is:

```python
K = np.array([[10., -2.,  0.,  1.,  0.,  0.],
              [-2.,  8., -3.,  0.,  1.,  0.],
              [ 0., -3.,  9., -1.,  0.,  2.],
              [ 1.01, 0., -1.,  7., -2.,  0.],
              [ 0.,  1.,  0., -2.,  6., -1.],
              [ 0.,  0.,  2.,  0., -1.,  5.]], dtype=float)

f = np.array([1., 0., -1., 0., 2., 0.], dtype=float)
```

`K` has `K[3,0] = 1.01` but `K[0,3] = 1.0` — a 1% asymmetry.

Solve `Ku = f` for the displacement vector `u`. Then answer the following questions with code and brief comments:

1. What is the condition number of `K`?
2. Based on the condition number alone, how many decimal digits of `u` should you trust?
3. If the entries of `f` have a relative error of `1e-4` (from measurement uncertainty), what is the worst-case relative error in `u`?

You should use at least two functions from `scipy.linalg` in your solution.

---

### I2 — Low-Rank Approximation and Variance Explained

You are working with a 100×80 data matrix representing sensor measurements (generate it with the seed below):

```python
rng = np.random.default_rng(42)
signal = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 80))
noise  = 0.3 * rng.standard_normal((100, 80))
M = signal + noise
```

The matrix was constructed with low-rank signal plus noise, but you don't know its rank in practice.

Without being told which functions to use:
1. Compute the rank-k approximations of `M` for `k = 1, 2, 5, 10, 20`.
2. For each k, compute the relative Frobenius error: `||M - M_k||_F / ||M||_F`.
3. Determine the smallest k such that the approximation captures at least **95% of the total variance** (i.e., 95% of the sum of squared singular values).
4. Report the "effective rank" you would recommend for this dataset.

---

### I3 — Stationary Distribution via Eigendecomposition

A 6×6 column-stochastic transition matrix describes web surfing behavior between six pages:

```python
P = np.array([[0.10, 0.20, 0.30, 0.00, 0.10, 0.20],
              [0.20, 0.10, 0.10, 0.30, 0.20, 0.10],
              [0.30, 0.10, 0.10, 0.20, 0.10, 0.10],
              [0.10, 0.30, 0.10, 0.10, 0.20, 0.30],
              [0.10, 0.20, 0.20, 0.20, 0.20, 0.10],
              [0.20, 0.10, 0.20, 0.20, 0.20, 0.20]], dtype=float)
```

The stationary distribution π satisfies `P @ π = π` (i.e., π is the eigenvector of `P` corresponding to eigenvalue 1).

Find π using eigenvalue decomposition — **do not use iterative power methods**. Your answer must:
1. Extract the correct eigenvector (eigenvalue closest to 1.0).
2. Normalize it so that its entries sum to 1 and all entries are non-negative.
3. Verify: compute `P @ π - π` and report its norm.
4. State which page gets the most traffic under the stationary distribution.

---

### I4 — Least Squares and Rank Diagnosis

An overdetermined system with 12 equations and 3 unknowns arises from a calibration experiment:

```python
rng = np.random.default_rng(99)
A = rng.standard_normal((12, 3))
x_true = np.array([2.0, -1.5, 0.8])
b = A @ x_true + 0.05 * rng.standard_normal(12)
```

The system `Ax = b` has no exact solution.

Without using `numpy.linalg.lstsq` or `scipy.linalg.lstsq` directly:
1. Use the SVD (from `scipy.linalg`) to compute the pseudoinverse of `A` and recover the least-squares solution `x_ls`.
2. Compute and report the residual norm `||b - A @ x_ls||`.
3. Assess whether the columns of `A` are linearly independent using the singular values.
4. Compare `x_ls` to `x_true` and report the relative error.

---

### I5 — Cholesky Decomposition for Sampling

A multivariate normal distribution with zero mean has the following covariance matrix:

```python
rng = np.random.default_rng(7)
F = rng.standard_normal((5, 8))
Sigma = (F @ F.T) / 8 + 0.5 * np.eye(5)
```

You need to generate 2000 samples from `N(0, Σ)` efficiently. To do this, find a matrix `L` such that `L @ L.T = Σ`, then generate samples as `x = L @ z` where `z ~ N(0, I)`.

1. Compute `L` using the appropriate decomposition from `scipy.linalg`. Explain in a comment why this decomposition is preferred over the full eigendecomposition for this task.
2. Generate 2000 samples.
3. Estimate the sample covariance matrix from your samples.
4. Report the Frobenius norm of `||Σ_sample - Σ||_F`. A well-implemented sampler should give a value below 0.15.

---

## Advanced Problems

*A realistic scenario. Module and function selection is not given. Edge cases and critical interpretation are required. A defensible approach counts as a correct answer.*

---

### A1 — Spectral Unmixing of Chemical Mixtures

A chemist has collected absorbance spectra from a mixture of three chemical components. The measurement matrix `X` has shape (50, 200): 50 samples at 200 wavelength bands. The data was generated as follows (but you should treat this as unknown):

```python
rng = np.random.default_rng(123)
S_true = np.abs(rng.standard_normal((3, 200)))   # pure-component spectra
C_true = np.abs(rng.standard_normal((50, 3)))    # concentrations
X = C_true @ S_true + 0.01 * rng.standard_normal((50, 200))
```

The spectra of components A and B are known:

```python
rng2 = np.random.default_rng(123)
S_known = np.abs(rng2.standard_normal((2, 200)))  # rows: spectra of A and B
```

**Your task:**
1. Diagnose the rank structure of `X`. How many significant components does the data actually contain? Justify your answer quantitatively.
2. Recover the unknown spectrum of component C using the SVD and the known spectra of A and B.
3. Estimate the concentration of component C in each of the 50 samples.

You may not be able to recover C exactly due to noise — explain in comments what limits the recovery and what a practitioner should check.

---

### A2 — Sparse Laplacian and Graph Connectivity

A synthetic graph with `N = 600` nodes and randomly placed edges is provided below. You should construct it yourself:

```python
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(55)
N = 600
# Random sparse adjacency: each node connected to ~8 others on average
rows, cols = [], []
for i in range(N):
    neighbors = rng.choice(N, size=8, replace=False)
    for j in neighbors:
        if i != j:
            rows.extend([i, j])
            cols.extend([j, i])
A_adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
A_adj = (A_adj + A_adj.T)  # symmetrize
A_adj.data = np.ones_like(A_adj.data)  # unweighted
A_adj = A_adj.tocsr()
```

**Your task:**
1. Construct the unnormalized graph Laplacian `L = D - A` in an appropriate sparse format.
2. Use ARPACK (via `scipy.sparse.linalg`) to find the 8 smallest eigenvalues of `L` and their eigenvectors.
3. What does the multiplicity of the zero eigenvalue tell you about this graph?
4. Identify the Fiedler value (second smallest eigenvalue) and describe its practical significance.
5. Use the Fiedler vector to attempt a 2-partition of the graph. Report how balanced the partition is.

Justify your choice of sparse format and ARPACK parameters in comments.

---

### A3 — Tridiagonal Sparse System from Heat Equation Discretization

A 1D steady-state heat equation on the interval [0, 1] with Dirichlet boundary conditions is discretized on a uniform grid of `N = 1200` interior points. The resulting linear system has the form `A u = f` where `A` is a tridiagonal matrix:

- Diagonal entries: `2 / h²`
- Off-diagonal entries: `-1 / h²`
- `h = 1 / (N + 1)` (grid spacing)
- Source term: `f_i = sin(π x_i)` where `x_i = i * h`
- Exact solution: `u(x) = sin(πx) / π²`

**Your task:**
1. Assemble `A` in CSR sparse format (do **not** form a dense matrix).
2. Solve the system using a sparse direct solver.
3. Compare the numerical solution to the exact solution and report the maximum absolute error.
4. **Benchmark**: Measure wall-clock time for the sparse solve vs. converting `A` to dense and calling `scipy.linalg.solve`. Report the ratio.
5. Comment on when the crossover point between dense and sparse approaches occurs.

---

### A4 — Best-Fit Plane via SVD

A dataset of 300 3D points lies approximately on a plane embedded in 3D space, with added noise:

```python
rng = np.random.default_rng(17)
n_true = np.array([1., 2., 3.])
n_true = n_true / np.linalg.norm(n_true)     # true plane normal
t1 = np.array([1., 0., -1./3.])              # vector in the plane
t1 = t1 / np.linalg.norm(t1)
t2 = np.cross(n_true, t1)
coords = rng.standard_normal((300, 2))
X = coords[:, 0:1] * t1 + coords[:, 1:2] * t2 + 0.05 * rng.standard_normal((300, 3))
X = X + np.array([3., -1., 2.])              # offset from origin
```

**Your task:**
1. Subtract the centroid and find the best-fit plane normal using SVD.
2. Report the recovered normal vector and compare to `n_true`. The angle between them should be less than 5 degrees.
3. Project all 300 points onto the best-fit plane and report the RMS distance from the original points to their projections.
4. Report the 2D coordinates of the projected points in the plane's local frame (for potential downstream visualization).

---

### A5 — Repairing a Near-Singular Covariance Matrix

A covariance matrix has been estimated from a small dataset of 40 samples in 50 dimensions. The estimator used was the sample covariance, which is rank-deficient whenever `n < p`:

```python
rng = np.random.default_rng(303)
data = rng.standard_normal((40, 50))
Sigma_hat = np.cov(data.T)   # shape (50, 50), but rank at most 39
```

When a downstream algorithm attempts a Cholesky decomposition of `Sigma_hat`, it fails.

**Your task:**
1. Confirm the failure with code and diagnose why it fails (report the rank and the smallest eigenvalues).
2. Apply **two different fixes** and test each:
   - Fix A: Eigenvalue thresholding — replace all negative or near-zero eigenvalues with a small positive floor value.
   - Fix B: Ledoit-Wolf-inspired shrinkage — compute `Sigma_reg = (1 - alpha) * Sigma_hat + alpha * np.eye(50)` for a suitable `alpha`.
3. For both repaired matrices, verify that Cholesky succeeds and report the condition number.
4. Recommend which fix is more appropriate for this use case and why.

---

## Expert Problems

*Ambiguous problem statement. No uniquely correct solution. Performance and numerical robustness matter. Cross-principle contamination is deliberate. Justify your design choices.*

---

### E1 — Truncated SVD for High-Dimensional Regression

A gene expression dataset is stored as a matrix of shape `(p, n) = (8000, 120)`: 8000 genes, 120 patients. Because `p >> n`, the full SVD is wasteful. A clinical outcome vector `y` (length 120) is also available.

```python
rng = np.random.default_rng(777)
# Simulate: true signal lives in a 6-dimensional subspace
W = rng.standard_normal((8000, 6))
Z = rng.standard_normal((6, 120))
X = W @ Z + 0.5 * rng.standard_normal((8000, 120))
beta_true = rng.standard_normal(6)
y = (Z.T @ beta_true) + 0.3 * rng.standard_normal(120)
```

**Your task:**
1. Use ARPACK (via `scipy.sparse.linalg`) to compute the truncated SVD of `X` retaining only `k` components. Justify your choice of `k` using a scree plot heuristic — do not simply inspect `beta_true`.
2. Project `X` onto the top-k right singular vectors to get a reduced feature matrix `Z_hat` of shape `(n, k)`.
3. Solve the regression problem `y ~ Z_hat` using the least-squares approach from `scipy.linalg`.
4. Compare the predictive performance (held-out MSE, using a 20% test split) to: (a) naive ridge regression on the full `X.T`, and (b) PCA via `numpy.linalg.svd` followed by regression.
5. Discuss at least two numerical stability tradeoffs between the ARPACK approach and full SVD approach.

---

### E2 — Spectral Clustering Pipeline

You are building a spectral clustering pipeline for 1500 2D points arranged in three interlocking rings:

```python
rng = np.random.default_rng(88)
def make_ring(n, r, noise, cx, cy):
    theta = rng.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(theta) + noise * rng.standard_normal(n)
    y = cy + r * np.sin(theta) + noise * rng.standard_normal(n)
    return np.stack([x, y], axis=1)

X = np.vstack([make_ring(500, 1.0, 0.05, 0, 0),
               make_ring(500, 2.2, 0.05, 0, 0),
               make_ring(500, 3.5, 0.05, 0, 0)])
```

**Your task:**
1. Construct a k-NN similarity graph. Justify your choice of `k` (number of neighbors). The graph must be connected.
2. Form the **normalized** graph Laplacian: `L_sym = D^{-1/2} (D - A) D^{-1/2}`.
3. Use ARPACK with **shift-invert mode** to find the smallest non-trivial eigenvalues of `L_sym`. Explain why shift-invert is necessary here and what `sigma` value you choose.
4. Determine the number of clusters from the eigenvalue spectrum. Visualize the eigengap.
5. Assign cluster labels from the eigenvectors (use k-means or sign-based thresholding) and report the adjusted Rand index against the ground-truth labels.
6. Compare memory and wall-clock time to forming the full dense `L_sym` and calling `scipy.linalg.eigh`. At what scale does the sparse approach become essential?

---

### E3 — Matrix Logarithm of a Transition Matrix

A continuous-time Markov chain's transition matrix `P(t)` has been observed at a single time `t = 1`. You want to recover the **generator matrix** `Q` such that `expm(Q) = P`.

```python
rng = np.random.default_rng(42)
# Generate a valid generator matrix
n = 8
rates = np.abs(rng.standard_normal((n, n)))
np.fill_diagonal(rates, 0)
Q_true = rates.copy()
np.fill_diagonal(Q_true, -rates.sum(axis=1))
P_obs = scipy.linalg.expm(Q_true)
```

The problem is that `scipy.linalg.logm(P)` can give a complex or non-generator result when applied naively.

**Your task:**
1. Apply `scipy.linalg.logm` naively and report what happens. Is the result real? Does it satisfy the generator property (non-negative off-diagonals, rows sum to zero)?
2. Implement a more robust approach using the **Schur decomposition**: decompose `P = Z T Z^H`, compute the logarithm of the triangular factor analytically, reconstruct `Q = Z logm(T) Z^H`.
3. Validate: compute `||expm(Q_recovered) - P_obs||_F`.
4. Test the robustness of your pipeline by artificially perturbing `P_obs` by 1% noise and reporting how much `Q_recovered` changes. What does this imply about the inverse problem?
5. Discuss: what happens if `P` has an eigenvalue with negative real part or near the branch cut of the logarithm? How would you detect this and what would you do?

---

### E4 — Tikhonov Regularization with Cross-Validated λ Selection

A linear inverse problem arises from a signal reconstruction application. The forward operator `A` has shape `(500, 80)` and is severely ill-conditioned:

```python
rng = np.random.default_rng(2024)
U, _ = np.linalg.qr(rng.standard_normal((500, 80)))
s_true = np.exp(-np.arange(80) / 5.0)          # exponentially decaying singular values
V, _ = np.linalg.qr(rng.standard_normal((80, 80)))
A = U[:, :80] * s_true @ V.T
x_true = rng.standard_normal(80)
b = A @ x_true + 0.01 * rng.standard_normal(500)
```

**Your task:**
1. Attempt naive least squares (via SVD pseudoinverse) and report the relative error `||x_ls - x_true|| / ||x_true||`. Explain why the solution is poor despite low noise.
2. Implement Tikhonov regularization: solve `(A^T A + λI) x = A^T b`. Do this efficiently using the SVD of `A` — not by forming and solving the normal equations directly.
3. Use 5-fold cross-validation over `λ ∈ [1e-6, 1e0]` (log scale) to select the optimal λ. Report the cross-validation curve.
4. Report the relative error of the Tikhonov solution at the optimal λ. Compare to the naive least-squares error.
5. Plot the L-curve (solution norm vs. residual norm) as a function of λ. Identify the corner. Does the L-curve corner agree with the cross-validated λ?
6. Discuss: how does the singular value spectrum of `A` inform an appropriate range for λ?

---

### E5 — Rotation Sequence Analysis and Interpolation

A motion capture system records the orientation of a rigid body as a sequence of 3×3 rotation matrices at irregular time stamps. The data is simulated below:

```python
from scipy.spatial.transform import Rotation
import numpy as np

rng = np.random.default_rng(314)
# True: a smooth rotation about a slowly evolving axis
t_obs = np.sort(rng.uniform(0, 4 * np.pi, 40))  # irregular timestamps
omega = np.array([0.3, 0.5, 0.8])                # angular velocity vector
R_obs = [Rotation.from_rotvec(omega * t) for t in t_obs]
t_reg = np.linspace(t_obs[0], t_obs[-1], 300)    # regular target timestamps
```

**Your task:**
1. Interpolate the rotation sequence to regular timestamps using SLERP via the `Rotation` class. Plot the interpolated rotation vectors over time.
2. Extract the angular velocity at each regular timestep as the finite-difference approximation of the rotation vector derivative.
3. Apply an FFT to the three components of angular velocity. Identify the dominant frequency. Does it agree with `||omega||`?
4. **Numerical comparison:** Implement naive matrix interpolation by linearly averaging adjacent rotation matrices (i.e., `R_interp = (R_i + R_{i+1}) / 2`, then renormalize via SVD). Quantify the error introduced by naive averaging vs. SLERP by comparing to the known ground truth `Rotation.from_rotvec(omega * t_reg)`.
5. At what angular step size (in degrees) does the naive averaging error exceed 1 degree? Characterize this as a function of the interpolation interval.
