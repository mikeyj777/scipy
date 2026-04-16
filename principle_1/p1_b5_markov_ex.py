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
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from principle_1.helpers import Chainlink

P = np.array([[0.8, 0.2], [0.6, 0.4]])



# If today is Sunny, what's the probability distribution after 1 day? After 5 days? After 20 days?
num_days = [1, 5, 20]
starting_weather_is_sunny = True
for days in num_days:
  link = Chainlink(is_sunny=starting_weather_is_sunny, target_days=days, P=P)
  link_dict = link.get_link_dict()
  links_after_x_days = link_dict[days]
  for link_future in links_after_x_days:
    if link_future.is_sunny:
      # sum probs for each day, sunny vs not
      # print(f'day {days} | prob is sunny: {link_future.my_probability_of_existing}')