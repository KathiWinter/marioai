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

if a dataset does not exist already, run `python preprocess_data.py` to create a new dataset .h5 file. Before running the script, change the constant `DATAPATH` to a new individual name. We aggreed to name the files with their creation date, e.g. "dataset20220904.h5" for a dataset created on September 4th, 2022.
The command will add all _.npz-files in the "data" folder to the .h5 dataset. If you would like to use different _.npz-files, change the folder path in line 10.

### 2) train the agent

to train a CDQN-agent, call the script `training-cdqn.py`. For a DQN-agent, use the `training-dqn.py` script accordingly.
make sure that the `DATAPATH` points to the right dataset.
the script will generate logs and models. From the latest model, it will automatically generate the policy and save it to `policy.pt`.

### 3) evaluate the trained agent

to evaluate a trained CDQN-agent, call the script `evaluation-cdqn.py`. or a DQN-agent, use the `evaluation-dqn.py` script accordingly. Before running the script, make sure that the constant `POLICY` in `constants.py` points to the right policy.pt file and the `DATAPATH` points to the correct dataset.
