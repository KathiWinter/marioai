#Demo Training with d3rlpy.dqn to test dataset
import os
import glob
import numpy as np
import d3rlpy
import gym
from gym_marioai import levels
import copy
from cdqn import CDQN
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

dataset = MDPDataset.load(DATAPATH)

cdqn = CDQN(gamma=0.99, batch_size=128)
log_dir="d3rlpy_logs"

train_episodes, test_episodes = train_test_split(dataset, test_size=0.1) 

cdqn.build_with_dataset(dataset)

all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=False,
               level_path=levels.hard_level,
               compact_observation=False, #this must stay false for proper saving in dataset
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)

evaluate_scorer = evaluate_on_environment(env)


#train with the given dataset
#eval_episodes: list of episodes to train
#n_epochs: number of epochs to train (one epoch contains a complete pass through the training dataset)
#save_interval: interval to save parameters (save model after x epochs)
#shuffle: flag to shuffle transitions on each epoch (different data combinations prevent overfitting)
#cdqn.fit(train_episodes, eval_episodes=test_episodes, n_epochs=15, logdir=log_dir, save_interval=1, shuffle=True)
fitter = cdqn.fitter(train_episodes, eval_episodes=test_episodes, n_epochs=20, shuffle=True, scorers={'environment': evaluate_scorer})

metr = []
for epoch, metrics in fitter:
  metr.append(metrics.get('environment'))
  if metrics.get('environment') > 160:
    print("model found with environment metrics: " + str(metrics.get('environment')))
    break

with open('metrics.txt', 'w') as f:
  for value in metr:
    f.write(str(value))
    f.write('\n')

#fetch latest dataset
latest_logs = max(glob.glob(os.path.join(log_dir, '*/')), key=os.path.getmtime)

#fetch latest model
latest_model = max(glob.iglob(latest_logs + '/*.pt'), key=os.path.getctime)
print(latest_model)
#to get specific model (not the latest), change this file path
cdqn.load_model(latest_model)
cdqn.save_policy(latest_logs +'/policy.pt')

