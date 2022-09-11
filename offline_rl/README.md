# how to run the project

To run the project, please install the dependencis with the following comand:
`pip install keyboard`
`pip install gym`
`pip install d3rlpy` or `conda install d3rlpy`



# how to play

### 1) run the server

run `java -jar ./marioai-server-0.1-jar-with-dependencies.jar` in gym_marioai/server folder

### 2) run the game

please set the user flag (`python play-for-training.py -u <int>` or `python play-for-training.py --user <int>`) when collecting data for training.
0 = test user we will ignore
all other: user-ids

`python play-for-training.py` runs the default seed 0
`python play-for-training.py -l coinLevel` or `python play-for-training.py --level coinLevel` runs specified level 'coinLevel'
`python play-for-training.py -s 188` or `python play-for-training.py --seed 188` runs specified seed 188
`python play-for-training.py -s random` or `python play-for-training.py --seed random` runs new random seed for each episode. Print output reveals seed number which can be specified in the next python call to repeat the level.

`python play-for-training.py --level coinLevel --seed 188` When a level is specified, python runs specified level and ignores seed number.

# how to train and evaluate the agent

### 1) get the data

if a dataset (.h5 file) already exists from the data the agent should train on, make sure the constant `DATAPATH` correctly points to the file.

if a dataset does not exist already, run `python preprocess_data.py` to create a new dataset .h5 file. Before running the script, change the constant `DATAPATH` to a new individual name. We agreed to name the files with their creation date, e.g. "dataset20220904.h5" for a dataset created on September 4th, 2022.
The command will add all _.npz-files in the "data" folder to the .h5 dataset. If you would like to use different _.npz-files, change the folder path in line 10.

You can also download ready-made datasets under the following link and add them to your project: https://drive.google.com/drive/folders/1mb82j67q6DG_lYNQlL3NSR3zHqozhgUj

### 2) train the agent

to train a CDQN-agent, call the script `training-cdqn.py`. For a DQN-agent, use the `training-dqn.py` script accordingly.
make sure that the `DATAPATH` points to the right dataset.
the script will generate logs and models in the `log_dir` directory. From the latest model, it will automatically generate the policy and save it to `policy.pt`.

### 3) evaluate the trained agent

to evaluate a trained CDQN-agent, call the script `evaluation-cdqn.py`. or a DQN-agent, use the `evaluation-dqn.py` script accordingly (for the stochastic version of the CDQN-agent, use `evaluation-cdqn-stoch.py`). Before running the script, make sure that the `DATAPATH` in `constants.py` points to the correct dataset and that the script loads the correct model. To watch Mario run through the level, make sure that the parameter `render` in the `gym.make()` function is set to `True`. If you want to run many levels without needing to watch them, set `render` to `False` instead. The cumulated reward of an episode is printed out in the terminal window. 

Our evaluations (including training data, videos and plots) can be found under this link: https://drive.google.com/drive/folders/168bFIhtkT2X-TNT5c0IPdzyjAhhXRXy6?usp=sharing

![](https://media.giphy.com/media/k1eyXufeMRyrow6PNJ/giphy.gif)
