'''

Count paths in a grid — Moving only right or down, how many unique paths exist from the top-left 
to the bottom-right corner? Still equal weighting, but now the branching represents a real decision 
at each position rather than a mathematical identity. 

The base cases represent edges of the grid where only one direction is possible. 
Builds spatial intuition for what the call tree actually represents.

'''

import numpy as np

grid_side = 5

grid = np.ones((grid_side, grid_side), dtype=np.int64)

out_grid = np.empty((grid_side, grid_side))

walk_dict = {}

# start in bottom left corner of grid. that has zero paths to get anywhere.  maybe you could consider it the one path to stay where it is
# go to the left and up.  add one each time you get to a new cell.  memoize the stored value
# continue back up to the origin.

def grid_walk(pos):
  global walk_dict
  x, y = pos
  if f'{x} {y}' in walk_dict:
    return walk_dict[f'{x} {y}']
  if x >= grid_side - 1:
    return 1
  if y >= grid_side - 1:
    return 1
  ans = grid_walk([x + 1, y]) + grid_walk([x, y + 1])
  walk_dict[f'{x} {y}'] = ans
  out_grid[x,y] = ans
  return grid_walk([x + 1, y]) + grid_walk([x, y + 1])

grid_walk([0, 0])

apple = 1