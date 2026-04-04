'''

### B3 — Singular Value Decomposition

Compute the full SVD of the following 5×3 matrix using `  `:

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

ans for Part 1: 
U=array([[-1.38909161e-01, -2.89690805e-01,  7.72517626e-01,
        -8.74960827e-02,  5.40688853e-01],
       [-3.39670269e-01, -1.89378116e-01,  3.85694228e-01,
         5.19667120e-01, -6.55702741e-01],
       [-5.40431378e-01, -8.90654269e-02, -1.12917078e-03,
        -7.76845992e-01, -3.10661077e-01],
       [-7.41192487e-01,  1.12472624e-02, -3.87952569e-01,
         3.44674955e-01,  4.25674965e-01],
       [-1.54583231e-01,  9.33892936e-01,  3.22409692e-01,
        -7.63278329e-17,  1.52655666e-16]])
s=array([25.76969686,  2.39640245,  1.08626852])
Vh=array([[-0.5045345 , -0.58483695, -0.63514618],
       [ 0.12919251,  0.67622032, -0.72528296],
       [-0.85367103,  0.44798641,  0.26561993]])


m1=array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.],
       [10., 11., 12.],
       [ 2.,  4.,  1.]])


2.  reconstruction close to orig:  True


3.  rank deficient: False


test for changes with full matrices set to True.
original sizing: U U.shape=(5, 5) s shape s.shape=(3,) Vh Vh.shape=(3, 3)
updated sizing: U U.shape=(5, 3) s shape s.shape=(3,) Vh Vh.shape=(3, 3)

output of U matches the smaller of the original matrices dimensions.  The other values, S and V, are limited by that smaller dimension in this example and don't change.

'''

import numpy as np
from scipy.linalg import svd

M = np.array([[ 1.,  2.,  3.],
              [ 4.,  5.,  6.],
              [ 7.,  8.,  9.],
              [10., 11., 12.],
              [ 2.,  4.,  1.]], dtype=float)

ans = svd(M)
U = ans[0]
s = ans[1]
Vh = ans[2]

print(f'ans for Part 1: \n{U=}\n{s=}\n{Vh=}')
m, n = M.shape

sigma = np.zeros((m,n))

for i in range(min(m,n)):
  sigma[i,i] = s[i]

m1 = np.dot(U, np.dot(sigma, Vh))

print(f'\n\n{m1=}')

print(f'\n\n2.  reconstruction close to orig:  {np.allclose(M, m1)}')

zeros = np.zeros(min(m,n))
rank_deficient = any(np.isclose(zeros, s))

print(f'\n\n3.  rank deficient: {rank_deficient}')

print(f'\n\ntest for changes with full matrices set to True.\noriginal sizing: U {U.shape=} s shape {s.shape=} Vh {Vh.shape=}')

ans = svd(M, full_matrices=False)
U = ans[0]
s = ans[1]
Vh = ans[2]

print(f'updated sizing: U {U.shape=} s shape {s.shape=} Vh {Vh.shape=}')

apple = 1