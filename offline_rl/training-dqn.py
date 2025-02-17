#Demo Training with d3rlpy.dqn to test dataset
import os
import glob
import gym
from gym_marioai import levels
import matplotlib.pyplot as plt 
from d3rlpy.algos import DQN
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.preprocessing import ClipRewardScaler, MinMaxRewardScaler
from d3rlpy.preprocessing import StandardRewardScaler
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
from torch.optim import Adam


plt.xlabel('epoch') 
plt.ylabel('reward') 

#encoder_factory = DefaultEncoderFactory(dropout_rate=0.5)
#clip_reward_scaler = ClipRewardScaler(-1.0, 1.0)
#standard_reward_scaler = StandardRewardScaler(mean=0.0, std=1.0)
dataset = MDPDataset.load(DATAPATH)
#min_max_reward_scaler = MinMaxRewardScaler(minimum=0, maximum=100)
#optim_factory = OptimizerFactory(Adam, eps=0.001)

dqn = DQN(gamma=0.8, batch_size=128)

log_dir="d3rlpy_logs"

train_episodes, test_episodes = train_test_split(dataset, test_size=0.1) 

all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=False, seed=0,
               compact_observation=False, #this must stay false for proper saving in dataset
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)

evaluate_scorer = evaluate_on_environment(env)


#train with the given dataset
#eval_episodes: list of episodes to train
#n_epochs: number of epochs to train (one epoch contains a complete pass through the training dataset)
#save_interval: interval to save parameters (save model after x epochs)
#shuffle: flag to shuffle transitions on each epoch (different data combinations prevent overfitting)
#dqn.fit(train_episodes, eval_episodes=test_episodes, n_epochs=15, logdir=log_dir, save_interval=1, shuffle=True)
fitter = dqn.fitter(train_episodes, eval_episodes=test_episodes, n_epochs=50, shuffle=True, scorers={'environment': evaluate_scorer, 'td_error': td_error_scorer})


metr = []
for epoch, metrics in fitter:
  metr.append(metrics.get('environment'))
  if metrics.get('environment') > 160:
    break


with open('metrics.txt', 'w') as f:
  for value in metr:
    f.write(str(value))
    f.write('\n')
  
plt.plot(metr)
plt.title("training curve")
plt.show()
  
  
#fetch latest dataset
latest_logs = max(glob.glob(os.path.join(log_dir, '*/')), key=os.path.getmtime)

#fetch latest model
latest_model = max(glob.iglob(latest_logs + '/*.pt'), key=os.path.getctime)
print(latest_model)
#to get specific model (not the latest), change this file path
dqn.load_model(latest_model)
dqn.save_policy(latest_logs +'/policy.pt')

