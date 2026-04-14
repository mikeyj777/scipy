'''

The Weather Model
A city has two weather states: Sunny and Rainy. You observe over many years:

If today is Sunny, there's a 80% chance tomorrow is Sunny and 20% chance it's Rainy
If today is Rainy, there's a 60% chance tomorrow is Sunny and 40% chance it's Rainy

The transition matrix is:
P = np.array([[0.8, 0.2],
              [0.6, 0.4]])
Questions to work through:

Verify P is a valid stochastic matrix
If today is Sunny, what's the probability distribution after 1 day? After 5 days? After 20 days?
Find the stationary distribution — the long run percentage of sunny vs rainy days
Verify the stationary distribution satisfies π @ P = π

'''
import numpy as np
from helpers import Chainlink

P = np.array([[0.8, 0.2], [0.6, 0.4]])

chain = []

def create_chain(is_sunny, parent_chainlink:Chainlink, target_day):
  global chain
  day = parent_chainlink.day
  if day == target_day:
    return
  day += 1
  parent_prob = parent_chainlink.my_probability_of_existing
  parent_was_sunny = parent_chainlink.is_sunny
  prob_row = P[(int(parent_was_sunny) + 1) % 2]
  prob = prob_row[(int(is_sunny) + 1) % 2]
  link = Chainlink(is_sunny=is_sunny, my_probability_of_existing=prob, P=P, parent_id=parent_chainlink.id)
  chain.append(link)
  create_chain


# If today is Sunny, what's the probability distribution after 1 day? After 5 days? After 20 days?
num_days = [1, 5, 20]

init_row = 0
results = {}
is_sunny = True
prob_of_existing = 1
for days in num_days:
  parent_link = Chainlink(is_sunny=is_sunny, my_probability_of_existing=prob_of_existing, P = P)
  parent_id = link0.me
  for day in days:
    for is_sunny in [True, False]:
      link = Chainlink(is_sunny=is_sunny)

    