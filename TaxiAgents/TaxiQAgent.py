# Imports
import gym
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# Set up environment
env = gym.make("Taxi-v3").env
obs = env.reset()

# Initialize q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n]) # All possible states (500) by all possible actions (6)

# Initialize hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.6 # Value placed on future Rewards
epsilon = 0.1 # Exploration and exploitation tradeoff
epsilon_minimum = 0.01
epochs = 25000
decay = 0.999 # Epsilon decay value

# Initialize metric lists
train_steps = []
train_penalties = []

# Training process
for epoch in range(1, epochs):
    state = env.reset()

    # Set base variables
    counter = 0 # Counts the number of steps it takes to solve from the intial state
    penalties = 0 # Number of incurred penalties
    done = False # Loop condition

    print(f"\nEpoch {epoch}")
    print("----------------------------------------------------------------------------------")

    while done == False:
        # Choose action 
        if random.uniform (0, 1) < epsilon:
            action = env.action_space.sample() # If the random number is less than the epsilon, choose a random action
        else:
            action = np.argmax(q_table[state]) # If the random number is more than the epsilon, choose a learned action

        next_state, reward, done, info = env.step(action)

        # Collect values for updates
        old_val = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Update q_table
        new_val = (1 - alpha) * old_val + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_val

        # Update rewards and penalties
        if reward == -10:
            penalties -= 1
        
        # Update state and counter
        state = next_state
        counter += 1

        while epsilon > epsilon_minimum:
            epsilon *= decay

        # View iteration values
        print("Time Step:", counter)
        print("State:", state)
        print("Action:", action)
        print(f"Reward: {reward} \n")
    
    # View epoch metrics
    print("Total Time Steps:", counter)
    print("Total Penalties:", penalties)

    # Add values to metric lists
    train_steps.append(counter)
    train_penalties.append(penalties)

# Evaluate model
episodes = 100
test_steps = 0
test_penalties = 0

# Test process
for episode in range(episodes):
    state = env.reset()

    # Set base variables
    counter = 0 # Counts the number of steps it takes to solve from the intial state
    penalties = 0 # Number of incurred penalties
    done = False # Loop condition

    while done == False:
        # Get action
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action) # Take action

        # Update rewards and penalties
        if reward == -10:
            penalties -= 1
        
        # Update counter
        counter += 1

        env.render()

    # View episode
    print(f"\n Test Episode: {episode} | Status: Over")

    # Update performance metrics
    test_steps += counter
    test_penalties += penalties

# View evaluation
print(f"\n\nModel's Average Time Steps Over {episodes} Episodes: {test_steps / episodes}")
print(f"Model's Average Penalty Over {episodes} Episodes: {test_penalties / episodes}")

# Demonstrate model
done = False
state = env.reset()

while done == False:
    # Render model
    env.render()

    # Get action
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action) # Take action
    time.sleep(0.1)

plt.plot(train_steps, label = "Training Steps")
plt.plot(train_penalties, label = "Training Penalties")
plt.xlabel("Epochs")
plt.ylabel("Number of Steps or Penalties")
plt.title("Training Steps and Penalties Across Epochs")
plt.legend()
plt.show()