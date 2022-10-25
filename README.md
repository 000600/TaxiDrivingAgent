# Taxi Driving Agent

## The Agents
There are two agents presented in this repository, one as a sample case (the random agent found in the **cliff_random_agent.py** file) and one as a basic exercise in reinforcement learning (the Q-Agent found in the **cliff_qagent.py** file).

1. The first model, found in the **taxi_random_agent.py** file, selects a random action from the action space (found with the **env.action_space.sample()** function provided by Gym) for the agent to take given an environment. As a result, the agent takes a large number of steps to ultimately reach the desired destination without falling off the cliff.

2. The second model, found in the **taxi_qagent.py** file, selects a random action based on a Q-Table that is updated every epoch during the training processs according to the Q-Table updating equation. The Q-Table has dimensions of 500 x 6 for 500 possible game states (25 taxi positions, 5 passenger positions, and 4) and 6 actions. The actions — north, east, west, south, pick up, drop off — are encoded as 0, 1, 2, 3, 4, 5 (respectively) in the model. I found that — after training for enough epochs — the agent was able to reach the desired destination from the starting position in the fewest number of steps while never taking an unnecessary turn or maneuvor.

## Hyperparameters
The model found in the **cliff_qagent.py** file is a basic Q-Learning algorithm, and thus has the following hyperparameters:
- A learning rate (alpha) value of 0.1
- A gamma value of 0.6
- An epsilon value of 0.4
- An epsilon decay rate of 0.999
- A minimum epsilon of 0.1
- 300 epochs

Feel free to build or improve upon the model or its hyperparameters!

## The Environment
The environment is provided by OpenAI's Gym package (linked below) under the package's *Toy Text Environments*. It provides a reward of -1 for each time step (to force the agent to be efficient), with a reward of -10 if the agent walks off of the cliff. The environment stops running when the agent has dropped off the passenger at the correct destination.

## Libraries
The models here were constructed with the help of the OpenAI Gym library.
- Gym's Website: https://www.gymlibrary.dev/
- Gym Installation Instructions: https://pypi.org/project/gym/0.7.4/
