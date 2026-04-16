'''

Factorial — Compute n!. The "hello world" of recursion. 
The recursive structure is almost identical to the mathematical definition,
so there's very little translation required.

'''
from itertools import count
import math

def fact(n):
  if n <= 1:
    return 1
  return n*fact(n-1)

counter = count(start=1)
while True:
  n = next(counter)
  print(f'{n=} {fact(n) - math.factorial(n) =}')

apple = 1