'''

Sum of a list — Given a list of numbers, return their sum without using any built-in aggregation. 
Forces you to think about "handle the first element, recurse on the rest" — a pattern that shows up everywhere.

'''

import numpy as np

def summer(arr):
  if len(arr) == 1:
    return arr[0]
  return arr[0] + summer(arr[1:])

for n in range(1,100):
  arr = np.random.randint(1,100, n)
  print(f'{float(summer(arr) - sum(arr))=}')