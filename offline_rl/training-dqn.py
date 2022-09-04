#Demo Training with d3rlpy.dqn to test dataset
import os
import glob
# use the d3rlpy DQN algorithm instead of our CDQN algorithm
from d3rlpy.algos import DQN
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH


dataset = MDPDataset.load(DATAPATH)

dqn = DQN()
log_dir="d3rlpy_logs"

#train with the given dataset
#eval_episodes: list of episodes to train
#n_epochs: number of epochs to train (one epoch contains a complete pass through the training dataset)
#save_interval: interval to save parameters (save model after x epochs)
#shuffle: flag to shuffle transitions on each epoch (different data combinations prevent overfitting)
dqn.fit(dataset, eval_episodes=dataset.episodes, n_epochs=30, logdir=log_dir, save_interval=10, shuffle=True)

#use this instead of dqn.fit when dqn.fit() has already been run
#dqn.build_with_dataset(dataset)

#fetch latest dataset
latest_logs = max(glob.glob(os.path.join(log_dir, '*/')), key=os.path.getmtime)

#fetch latest model
latest_model = max(glob.iglob(latest_logs + '/*.pt'), key=os.path.getctime)
print(latest_model)
#to get specific model (not the latest), change this file path
dqn.load_model(latest_model)
dqn.save_policy(latest_logs +'/policy.pt')

