import os
import glob
import numpy as np
from os.path import exists
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH

if __name__=="__main__":

    cwd = os.getcwd()
    os.chdir(cwd + '/offline_rl')

    episode_start = np.load('data/one_cliff_level/random_generated_oneCliffLevel_reward-63_1662464945.npz')

    observations_start = episode_start['observations']
    actions_start = episode_start["actions"]
    rewards_start = episode_start["rewards"]
    terminals_start = episode_start["terminals"]

    #initialize Markov-Decision-Process Dataset from collected data
    if not exists(DATAPATH):
        dataset = MDPDataset(np.array(observations_start), np.array(actions_start), np.array(rewards_start), np.array(terminals_start))   
    else:
        dataset = MDPDataset.load(DATAPATH)
        dataset.append(np.array(observations_start), np.array(actions_start), np.array(rewards_start), np.array(terminals_start))
    dataset.dump(DATAPATH)

    counter = 0   
    for file in glob.glob(os.path.join('data/one_cliff_level/', '*.npz')):
        episode_data = np.load(file)
  
        observations = episode_data["observations"]
        actions = episode_data["actions"]
        rewards = episode_data["rewards"]
        terminals = episode_data["terminals"]
  
        #create Markov-Decision-Process Dataset from collected data or append existing MDP Dataset
        if not exists(DATAPATH):
            dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))   
        else:
            #dataset = MDPDataset.load(DATAPATH)
            dataset.append(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))
        
        counter += 1
        print(counter)

    #save dataset in file path for episode
    dataset.dump(DATAPATH)