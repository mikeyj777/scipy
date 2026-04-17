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
    fib_dict[n] = 0
    return 0
  if n == 1:
    fib_dict[n] = 1
    return 1
  ans = fibonacci(n-2) + fibonacci(n-1)
  fib_dict[n] = ans
  return ans

# 1972 is max value that can be executed directly with the above format
# print(f'{fibonacci(1972)=}')

# the loop method fills out the dict and allows calculations limited by digit display settings
 
n = -1
while True:
  n += 1
  fibonacci(n)
  if n % 100 == 0:
    print(f'{n=} fibonacci({n})= {fib_dict[n]}')
  if n % 1000 == 0:
    apple = 1