#Demo Training with d3rlpy.SAC to test dataset

from d3rlpy.algos import DQN
from d3rlpy.dataset import MDPDataset
import torch


dataset = MDPDataset.load('data/dataset.h5')

sac = DQN()

sac.fit(dataset, eval_episodes=dataset,n_epochs=100)

#use this instead of sac.fit when sac.fit() has already been run
#sac.build_with_dataset(dataset)

sac.load_model('d3rlpy_logs\DQN_20220828125412\model_504.pt')
sac.save_policy('d3rlpy_logs\DQN_20220828125412\policy.pt')

actions = torch.jit.load('d3rlpy_logs\DQN_20220828125412\policy.pt')
