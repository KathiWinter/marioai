#Demo Training with d3rlpy.dqn to test dataset
import os
import glob
import numpy as np
import d3rlpy
import gym
from gym_marioai import levels
from cdqn import CDQN
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import td_error_scorer

dataset = MDPDataset.load(DATAPATH)

cdqn = CDQN(n_steps=100, gamma=0.99, batch_size=264, target_update_interval=10)
log_dir="d3rlpy_logs"

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2) 

td_error = td_error_scorer(cdqn, test_episodes)

#train with the given dataset
#eval_episodes: list of episodes to train
#n_epochs: number of epochs to train (one epoch contains a complete pass through the training dataset)
#save_interval: interval to save parameters (save model after x epochs)
#shuffle: flag to shuffle transitions on each epoch (different data combinations prevent overfitting)
cdqn.fit(train_episodes, eval_episodes=test_episodes, n_epochs=1, logdir=log_dir, save_interval=1, shuffle=True, scorers={'td_error': td_error_scorer})

#use this instead of cdqn.fit when cdqn.fit() has already been run
#cdqn.build_with_dataset(dataset)

#fetch latest dataset
latest_logs = max(glob.glob(os.path.join(log_dir, '*/')), key=os.path.getmtime)

#fetch latest model
latest_model = max(glob.iglob(latest_logs + '/*.pt'), key=os.path.getctime)
print(latest_model)
#to get specific model (not the latest), change this file path
cdqn.load_model(latest_model)
cdqn.save_policy(latest_logs +'/policy.pt')

