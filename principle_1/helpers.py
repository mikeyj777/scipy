import numpy as np
from scipy.linalg import expm

class Chainlink:
<<<<<<< HEAD
  def __init__(self, is_sunny, my_probability_of_existing = 1, P = None, Q = None, parent_id= None, day):
=======
  def __init__(self, is_sunny, my_probability_of_existing, current_day, P = None, Q = None, parent = None):
>>>>>>> fb9efd80d7aafd5975bfad8d462ffc5ed06966e3
    self.P = P
    self.Q = Q
    self.my_probability_of_existing = my_probability_of_existing
    self.parent_id= parent_id
    self.children = []
    self.is_sunny = is_sunny
    self.prob_row_num = (int(is_sunny) + 1) % 2
    self.probability_tmrw_is_sunny = None
<<<<<<< HEAD
    self.id = id(self)
    self.day = day
=======
    self.me = id(self)
    self.current_day = current_day
>>>>>>> fb9efd80d7aafd5975bfad8d462ffc5ed06966e3
    
  def get_P_at_t(self, t):
    return expm(t * self.Q)
  
  def set_P(self, P):
    self.P = P
  
  def set_probabilities(self, P = None):
    if P is None:
      P = self.P
    p_row = P[self.is_sunny]
<<<<<<< HEAD
    self.probability_tmrw_is_sunny = self.my_probability_of_existing * p_row[0]
  
  
    
=======
    self.probability_tmrw_is_sunny = self.p * p_row[0]
>>>>>>> fb9efd80d7aafd5975bfad8d462ffc5ed06966e3
