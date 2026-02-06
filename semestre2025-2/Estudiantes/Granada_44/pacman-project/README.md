# Pac-Man: Supervised vs. Reinforcement Learning Project

In this repository, I compare two different ways to beat Pac-Man: Supervised Learning and Reinforcement Learning

### How to Run It

(inside /pacman)
To run the Q learning Agent: 

python pacman.py -p TrainAgentFixed -a "alpha=0.03,gamma=0.9,epsilon=0.2" -l smallClassic -n 10

To run the Supervised Learning:

python pacman.py -p SupervisedAgent -l smallClassic -n 10

Environment taken from [UC Berkeley CS188 Pac-Man setup](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj1/#welcome-to-pacman).

Built by me <3