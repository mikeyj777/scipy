import numpy as np
from scipy.linalg import expm

class Chainlink:
  def __init__(self, is_sunny, my_probability_of_existing = 1, P = None, Q = None, parent_id= None, day):
    self.P = P
    self.Q = Q
    self.my_probability_of_existing = my_probability_of_existing
    self.parent_id= parent_id
    self.children = []
    self.is_sunny = is_sunny
    self.prob_row_num = (int(is_sunny) + 1) % 2
    self.probability_tmrw_is_sunny = None
    self.id = id(self)
    self.day = day
    
  def get_P_at_t(self, t):
    return expm(t * self.Q)
  
  def set_P(self, P):
    self.P = P
  
  def set_probabilities(self, P = None):
    if P is None:
      P = self.P
    p_row = P[self.is_sunny]
    self.probability_tmrw_is_sunny = self.my_probability_of_existing * p_row[0]
  
  
    