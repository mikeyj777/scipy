import numpy as np
from scipy.linalg import expm

class Chainlink:
  def __init__(self, is_sunny, my_probability_of_existing, P = None, Q = None, parent = None):
    self.P = P
    self.Q = Q
    self.p = my_probability_of_existing
    self.parent = parent
    self.children = []
    self.is_sunny = int(is_sunny)
    self.probability_tmrw_is_sunny = None
    self.me = id(self)
    
  def get_P_at_t(self, t):
    return expm(t * self.Q)
  
  def set_P(self, P):
    self.P = P
  
  def set_probabilities(self, P = None):
    if P is None:
      P = self.P
    p_row = P[self.is_sunny]
    self.probability_tmrw_is_sunny = self.p * p_row[0]
    
    