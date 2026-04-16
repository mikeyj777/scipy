import numpy as np
from scipy.linalg import expm

link_dict = {}

class Chainlink:
  def __init__(self, is_sunny, target_days, my_probability_of_existing = 1, current_day = 0, P = None, Q = None, parent_id= None):
    global link_dict
    self.P = P
    self.Q = Q
    self.my_probability_of_existing = float(my_probability_of_existing)
    self.is_sunny = is_sunny
    self.target_days = target_days
    self.current_day = current_day
    self.parent_id= parent_id
    prob_row_idx = (int(is_sunny) + 1) % 2
    prob_row = self.P[prob_row_idx]
    self.prob_tmrw_is_sunny = prob_row[0]
    self.id = id(self)
    self.children = []
    if current_day not in link_dict:
      link_dict[current_day] = []
    link_dict[current_day].append(self)
    if current_day < target_days:
      self.make_children()
      
    
  def get_P_at_t(self, t):
    return expm(t * self.Q)
  
  def get_link_dict(self):
    return link_dict
  
  def set_P(self, P):
    self.P = P
    
  def make_children(self):
    #tomorrow is sunny
    self.children.append(Chainlink(is_sunny=True, target_days=self.target_days, my_probability_of_existing=self.my_probability_of_existing * self.prob_tmrw_is_sunny, current_day=self.current_day + 1, P = self.P, Q=self.Q, parent_id=self.id))
    self.children.append(Chainlink(is_sunny=False, target_days=self.target_days, my_probability_of_existing=self.my_probability_of_existing * (1-self.prob_tmrw_is_sunny), current_day=self.current_day + 1, P = self.P, Q=self.Q, parent_id=self.id))
    