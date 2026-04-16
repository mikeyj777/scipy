'''

Count occurrences — Given a value and a list, count how many times the value appears. 
Introduces the idea of conditionally contributing to the result depending on what you find at each step.

'''

import numpy as np
from random import randint

def count_occurrences(val, arr, tot = 0):
  if len(arr) == 0:
    return tot
  if val == arr[-1]:
    tot += 1
  return count_occurrences(val, arr[:-1] , tot)
  

working = 0
tot_count = 0

for n in range(10,900):
  arr = np.random.randint(1,100,n)
  val = randint(1, 100)
  unique, counts = np.unique(arr, return_counts=True)
  if val not in unique:
    continue
  unique_counts = dict(zip(unique, counts))
  val_count = unique_counts[val]
  val_count_recursive = count_occurrences(val, arr)
  diff = val_count_recursive - val_count
  if diff == 0:
    working += 1
  tot_count += 1
  
print(f'{working / tot_count=}')