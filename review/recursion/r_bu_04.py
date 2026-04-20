'''

Minimum cost path in a grid — Same grid structure as above, 
but now each cell carries a cost and you want the cheapest path. 
The combination step changes from pure addition to a comparison between 
two weighted results. The current frame now has to do real work with what
comes back before returning it. This is the first problem where the combination 
step requires thought at every level, not just the base.

Blank grid with costs assigned to each cell. The previous grid's values were path counts, 
which have no meaningful relationship to cost. You'll want to populate a fresh grid with 
some cost values — random integers work fine — and then find the path from top-left to 
bottom-right that minimizes the sum of costs along the way.

model output:

grid with costs:
[[2 6 7 7 7]
 [9 7 2 5 6]
 [6 1 6 1 1]
 [3 4 8 9 6]
 [2 2 2 3 4]]

cost of path of least cost:  31

'''

import numpy as np

grid_side = 5

results = np.full((grid_side, grid_side), None)

grid = np.random.randint(1,10,(grid_side, grid_side))

print('grid with costs:')
print(grid)
print('\n')

def grid_walk(pos):
  x, y = pos
  if x == grid_side - 1 and y == grid_side -1:
    results[x, y] = int(grid[x, y])
    return results[x, y]
  if results[x, y] is not None:
    return results[x, y]
  ans = None
  x_step = None
  y_step = None
  if x < grid_side-1:
    x_step = grid_walk([x+1, y])
  if y < grid_side-1:
    y_step = grid_walk([x, y+1])
  if x_step is not None:
    if y_step is None:
      ans = grid[x, y] + x_step
    else:
      ans = min(x_step, y_step) + grid[x, y]
  if ans is None and y_step is not None:
    ans = y_step + grid[x, y]
  if ans is None:
    raise Exception('you did something wrong.')
  results[x, y] = int(ans)
  return results[x, y]

grid_walk([0, 0])

print(f'cost of path of least cost:  {results[0, 0]}')

apple = 1