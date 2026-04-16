'''

Reverse a string — Return a string in reverse order. Similar structure to sum of a list, 
but now the combination step (how you reassemble on the way back up) requires a little more thought.

'''

def string_reverse(instr):
  if len(instr) <= 1:
    return instr
  return instr[-1] + string_reverse(instr=instr[:-1])

instr = 'abcdefghijklmnopqrstuvwxyz'
print(f'{instr} reversed is {string_reverse(instr)}')