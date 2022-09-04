import os
import glob
import numpy as np
from os.path import exists
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH

counter = 0

for file in glob.glob(os.path.join('data/', '*.npz')):
  episode_data = np.load(file)
  
  observations = episode_data["observations"]
  actions = episode_data["actions"]
  rewards = episode_data["rewards"]
  terminals = episode_data["terminals"]
  
  #create Markov-Decision-Process Dataset from collected data or append existing MDP Dataset
  if not exists(DATAPATH):
      dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))   
  else:
      dataset = MDPDataset.load(DATAPATH)
      dataset.append(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))
        
  counter += 1
  print(counter)
  #save dataset in file path for episode
  dataset.dump(DATAPATH)