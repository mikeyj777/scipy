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

script results:

in 1 days the probability of it being sunny is 0.800.  the probability of it being cloudy is 0.200
in 5 days the probability of it being sunny is 0.750.  the probability of it being cloudy is 0.250
in 20 days the probability of it being sunny is 0.750.  the probability of it being cloudy is 0.250

'''
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

P = np.array([[0.8, 0.2], [0.6, 0.4]])

def weather_prob(is_sunny, target_day, current_day = 0, my_probability = 1):
  if current_day == target_day:
    if is_sunny:
      return my_probability
    return 0
  idx = (int(is_sunny) + 1) % 2
  P_row = P[idx]
  p_tmrw_sunny = P_row[0]
  p_tmrw_cloudy = P_row[1]
  return weather_prob(is_sunny=True, target_day=target_day, current_day=current_day + 1, my_probability=p_tmrw_sunny*my_probability) + \
    weather_prob(is_sunny=False, target_day=target_day, current_day=current_day+1, my_probability=p_tmrw_cloudy*my_probability)


# If today is Sunny, what's the probability distribution after 1 day? After 5 days? After 20 days?
num_days = [1, 5, 20]
# num_days = [30, 40, 50]
starting_weather_is_sunny = True
sunny_prob_dict = {}
# days = 0
for days in num_days:
  # days += 1
  sunny_prob_dict[days] = weather_prob(is_sunny=starting_weather_is_sunny, target_day=days)
  k = days
  v= sunny_prob_dict[days]
  print(f'in {k} days the probability of it being sunny is {v:.03f}.  the probability of it being cloudy is {1-v:.03f}')

apple = 1

# from principle_1.helpers import Chainlink
# for days in num_days:
#   link = Chainlink(is_sunny=starting_weather_is_sunny, target_days=days, P=P)
#   link_dict = link.get_link_dict()
#   links_after_x_days = link_dict[days]
#   for link_future in links_after_x_days:
#     if link_future.is_sunny:
      # sum probs for each day, sunny vs not
      # print(f'day {days} | prob is sunny: {link_future.my_probability_of_existing}')