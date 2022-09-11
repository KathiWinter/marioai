import gym
from gym_marioai import levels
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH
from cdqn import CDQN

dataset = MDPDataset.load(DATAPATH)


### Evaluation of our implemented Convergent DQN algorithm based on the d3rlpy DQN ###
cdqn = CDQN()


#use this instead of dqn.fit when dqn.fit() has already been run
cdqn.build_with_dataset(dataset)

#choose your model here
cdqn.load_model('../evaluations\hard_level\CDQN\model_15980.pt')

all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=True, # turn this off for fast training without video
               level_path=levels.hard_level,
               compact_observation=False, #this must stay false for proper saving in dataset
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)


while True:
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
  
            action = cdqn.predict([observation])[0]
            observation, reward, done, info = env.step(action)
 
       
            total_reward += reward
        print(f'finished episode, total_reward: {total_reward}')