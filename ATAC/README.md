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
