# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:46:20 2020

@author: moin_
"""

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment

experiment_name = 'deap test 1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
env.state_to_log()