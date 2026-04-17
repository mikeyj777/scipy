'''

Count paths in a grid — Moving only right or down, how many unique paths exist from the top-left 
to the bottom-right corner? Still equal weighting, but now the branching represents a real decision 
at each position rather than a mathematical identity. 

The base cases represent edges of the grid where only one direction is possible. 
Builds spatial intuition for what the call tree actually represents.

'''

import numpy as np

grid_side = 5

results = np.empty((grid_side, grid_side))

# start in bottom left corner of grid. that has zero paths to get anywhere.  maybe you could consider it the one path to stay where it is
# go to the left and up.  add one each time you get to a new cell.  memoize the stored value
# continue back up to the origin.

def grid_walk(pos):
  global walk_dict
  x, y = pos
  if not np.isnan(results[x, y]):
    return results[x, y]
  if x == grid_side - 1 and y == grid_side -1:
    return 0
  x_next = min(x + 1, grid_side-1)
  y_next = min(y + 1, grid_side-1)
  ans = 1 + grid_walk([x_next, y]) + grid_walk([x, y_next])
  results[x, y] = ans
  return ans

grid_walk([0, 0])

apple = 1