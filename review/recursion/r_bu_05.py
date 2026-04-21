'''

Gambler's ruin — A gambler starts with some amount of money, 
wins one dollar with probability p and loses one dollar with probability 
1-p on each bet, and stops when they either go broke or reach a target amount. 
What's the probability they reach the target? The base cases return 0 or 1 as the source of meaning, 
each frame weights two recursive results by the transition probabilities before combining them, 
and the structure is nearly identical to your weather problem — just on a number line instead of a state matrix.

'''
bankroll = 10
target = 20
p = 0.3
bankroll_prob_dict = {}
iters = 0

def betting_until_payday(bankroll=bankroll, target=target, p=p, win = True):
  global iters, bankroll_prob_dict
  iters += 1
  if bankroll not in bankroll_prob_dict:
    bankroll_prob_dict[bankroll] = {}
  if win in bankroll_prob_dict[bankroll]:
    return bankroll_prob_dict[bankroll][win]
  if bankroll <= 0:
    bankroll_prob_dict[bankroll][win] = 0
    return 0  
  if bankroll >= target:
    bankroll_prob_dict[bankroll][win] = 1
    return 1
  if iters == 475:
    apple = 1
  if True in bankroll_prob_dict[bankroll]:
    return bankroll_prob_dict[bankroll][True]
  else:
    bankroll_prob_dict[bankroll][True] = p * betting_until_payday(bankroll=bankroll + 1, win=True)
  
  if False in bankroll_prob_dict[bankroll]:
    return bankroll_prob_dict[bankroll][False]
  else:
    bankroll_prob_dict[bankroll][False] = (1-p) * betting_until_payday(bankroll=bankroll - 1, win=False)

  return bankroll_prob_dict[bankroll][win]

betting_until_payday()

for k, v in bankroll_prob_dict.items():
  v_descr = 'no'
  if True in v:
    v_descr = f'a {v[True]}'
  print(f'with a bankroll of {k} you have {v_descr} chance of making your target')
  

apple = 1