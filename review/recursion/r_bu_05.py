'''

Gambler's ruin — A gambler starts with some amount of money, 
wins one dollar with probability p and loses one dollar with probability 
1-p on each bet, and stops when they either go broke or reach a target amount. 
What's the probability they reach the target? The base cases return 0 or 1 as the source of meaning, 
each frame weights two recursive results by the transition probabilities before combining them, 
and the structure is nearly identical to your weather problem — just on a number line instead of a state matrix.

'''
bankroll = 10000
target = 20000
p = 0.3
keep_track = []

def betting_until_payday(bankroll, target, p):
  global keep_track
  keep_track.append(bankroll/target)
  if len(keep_track) % 100 == 0:
    print(sum(keep_track)/len(keep_track))
  if bankroll <= 0:
    return 0
  if bankroll >= target:
    return 1
  win = p * betting_until_payday(bankroll, target, p)
  lose = (1-p) * betting_until_payday(bankroll, target, p)
  bankroll += win - lose
  return bankroll / target

epochs = 100000
ans = 0

for i in range(epochs):
  ans += betting_until_payday(bankroll=bankroll, target=target, p=p)

print(f'probability of making target: {ans / epochs}')