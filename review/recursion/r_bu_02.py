'''

Fibonacci sequence — Your first branching bottom-up problem. 
Two recursive calls, and the current frame can't produce its answer until both resolve. 
The base cases are the source of all values — every number in the sequence originates from the 0s and 1s at the bottom. 
Equal weighting, pure addition, so you can focus entirely on the branching structure.

'''

fib_dict = {}

def fibonacci(n):
  if n in fib_dict:
    return fib_dict[n]
  if n <= 0:
    return 0
  if n == 1:
    return 1
  if n-2 not in fib_dict:
    fib_dict[n-2] = fibonacci(n-2)
  if n-1 not in fib_dict:
    fib_dict[n-1] = fibonacci(n-1)
  return fib_dict[n-2] + fib_dict[n-1]

n = -1
while True:
  n += 1
  fib_dict[n] = fibonacci(n)
  if n % 100 == 0:
    print(f'{n=} fibonacci({n})= {fib_dict[n]}')
  if n % 1000 == 0:
    apple = 1