import os
import keyboard

import gym
from gym_marioai import levels
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH
from d3rlpy.algos import DQN


dataset = MDPDataset.load(DATAPATH)

### Evaluation of the d3rlpy DQN algorithm ### 
dqn = DQN()


#use this instead of dqn.fit when dqn.fit() has already been run
dqn.build_with_dataset(dataset)

dqn.load_model('../evaluations\hard_level\DQN\model_43898.pt')
all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=True,
               level_path=levels.hard_level,
               compact_observation=False, #this must stay false for proper saving in dataset
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)


while True:
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            observation, reward, done, info = env.step(dqn.predict([observation])[0])
            total_reward += reward
        print(f'finished episode, total_reward: {total_reward}')