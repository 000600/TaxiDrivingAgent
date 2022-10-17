# Imports
import gym
import random
import time

# Set up environment
env = gym.make("Taxi-v3").env
obs = env.reset()

# Encode state
state = env.encode(3, 1, 2, 0)
env.s = state

# Set base variables
counter = 0 # Counts the number of steps it takes to solve from the intial state
penalties = 0 # Number of incurred penalties
done = False # Loop condition

# Loop
while done == False:
    # Take an action
    action = env.action_space.sample() # Select random action
    state, reward, done, info = env.step(action) # take that action

    # Update rewards and penalties
    if reward == -10:
        penalties -= 1

    # Update counter
    counter += 1

    # View results
    env.render()
    print("Time Step:", counter)
    print("State:", state)
    print("Action:", action)
    print("Reward:", reward)

# View steps and penalties
print("Number of Steps Taken:", counter)
print("Number of Penalties:", penalties)