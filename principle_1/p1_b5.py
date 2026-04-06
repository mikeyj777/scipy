'''

### B5 — Matrix Exponential

The following 3×3 matrix is the **generator matrix** of a continuous-time Markov chain. Each row sums to zero, which is the defining 
property of a generator:

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


'''
import numpy as np
from scipy.linalg import expm

Q = np.array([[-3.,  2.,  1.],
              [ 1., -4.,  3.],
              [ 2.,  1., -3.]], dtype=float)

