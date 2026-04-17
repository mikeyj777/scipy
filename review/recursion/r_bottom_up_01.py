'''

Maximum value in a list — Still linear like count_occurrences, but the combination step is 
a comparison rather than addition. The base case returns an actual meaningful value,
and the current frame has to decide between what it holds and what came back from below. 
Gets you comfortable with combination logic that isn't just "add 1."

'''

def find_max_val(arr):
  if len(arr) <= 1:
    return arr[0]
  return max(find_max_val(arr[:-1]), arr[-1])

arr = [1,2,3,2,1]

print(find_max_val(arr))