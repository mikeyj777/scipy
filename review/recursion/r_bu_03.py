'''

Count paths in a grid — Moving only right or down, how many unique paths exist from the top-left 
to the bottom-right corner? Still equal weighting, but now the branching represents a real decision 
at each position rather than a mathematical identity. 

The base cases represent edges of the grid where only one direction is possible. 
Builds spatial intuition for what the call tree actually represents.

model output
[[70 35 15 5 1]
 [35 20 10 4 1]
 [15 10 6 3 1]
 [5 4 3 2 1]
 [1 1 1 1 1]]

'''

import numpy as np

grid_side = 5

results = np.full((grid_side, grid_side), None)

# start in bottom left corner of grid. that base case has zero paths to get anywhere, and is considered 1 path.
# go to the left and up.  add one each time you get to a new cell.  memoize the stored value
# continue back up to the origin.

def grid_walk(pos):
  x, y = pos
  if x == grid_side - 1 and y == grid_side -1:
    results[x, y] = 1
    return 1
  if results[x, y] is not None:
    return results[x, y]
  ans = 0
  if x < grid_side-1:
    ans += grid_walk([x+1, y])
  if y < grid_side-1:
    ans += grid_walk([x, y+1])
  results[x, y] = ans
  return ans

grid_walk([0, 0])

print(results)

apple = 1