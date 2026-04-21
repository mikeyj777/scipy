'''

Gambler's ruin — A gambler starts with some amount of money, 
wins one dollar with probability p and loses one dollar with probability 
1-p on each bet, and stops when they either go broke or reach a target amount. 
What's the probability they reach the target? The base cases return 0 or 1 as the source of meaning, 
each frame weights two recursive results by the transition probabilities before combining them, 
and the structure is nearly identical to your weather problem — just on a number line instead of a state matrix.

'''
bankroll = 1
target = 3
p = 0.3
bankroll_prob_dict = {}
iters = 0

def betting_until_payday(bankroll=bankroll, target=target, p=p, win = True):
  global iters, bankroll_prob_dict
  iters += 1
  if bankroll <= 0:
    return 0
  if bankroll >= target:
    return 1
  if bankroll in bankroll_prob_dict:
    if win in bankroll_prob_dict[bankroll]:
      return bankroll_prob_dict[bankroll][win]
  if iters == 475:
    apple = 1
  if bankroll in bankroll_prob_dict:
    if True in bankroll_prob_dict[bankroll]:
      return bankroll_prob_dict[bankroll][True]
  if bankroll not in bankroll_prob_dict:
    bankroll_prob_dict[bankroll] = {}
  bankroll_prob_dict[bankroll][True] = p * betting_until_payday(bankroll=bankroll + 1, win=True)
  
  if bankroll in bankroll_prob_dict:
    if False in bankroll_prob_dict[bankroll]:
      return bankroll_prob_dict[bankroll][False]
  bankroll_prob_dict[bankroll][False] = (1-p) * betting_until_payday(bankroll=bankroll - 1, win=False)

betting_until_payday()