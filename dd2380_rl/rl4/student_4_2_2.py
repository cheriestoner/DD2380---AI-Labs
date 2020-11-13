#!/usr/bin/env python3
# Taking the shortest path to catch the King Fish.
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
# rewards = [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]
rewards = [40, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -5]
# Q learning learning rate
alpha = 0.6

# Q learning discount rate
gamma = 0.2

# Epsilon initial
epsilon_initial = 1

# Epsilon final
epsilon_final = 0

# Annealing timesteps
annealing_timesteps = 1000

# threshold
threshold = 1e-6
