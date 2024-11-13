import numpy as np
import random

# Parameters for Q-learning
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.3     # Exploration rate
num_episodes = 500 # Number of training episodes

# Example puzzle setup
puzzle = {
    "correct_groups": [
        ["HAIL", "RAIN", "SLEET", "SNOW"],    # WET WEATHER
        ["BUCKS", "HEAT", "JAZZ", "NETS"],     # NBA TEAMS
        ["OPTION", "RETURN", "SHIFT", "TAB"],  # KEYBOARD KEYS
        ["KAYAK", "LEVEL", "MOM", "RACECAR"]   # PALINDROMES
    ],
    "all_words": [
        "HAIL", "RAIN", "SLEET", "SNOW", "BUCKS", "HEAT", "JAZZ", "NETS", 
        "OPTION", "RETURN", "SHIFT", "TAB", "KAYAK", "LEVEL", "MOM", "RACECAR"
    ]
}

# Initialize Q-table as a dictionary
Q_table = {}

# Function to check if a grouping is correct
def is_correct_grouping(action, correct_groups):
    return any(set(action) == set(group) for group in correct_groups)

# Generate all possible groupings of four words as actions
def generate_possible_actions(words):
    from itertools import combinations
    return list(combinations(words, 4))

# Reward function: +1 for a correct grouping, -1 for incorrect
def reward_function(action, correct_groups):
    return 1 if is_correct_grouping(action, correct_groups) else -1

# Update Q-value function
def update_q_table(state, action, reward, next_state):
    state_action = (state, action)
    old_value = Q_table.get(state_action, 0)
    next_max = max(Q_table.get((next_state, a), 0) for a in generate_possible_actions(state['remaining_words']))
    Q_table[state_action] = old_value + alpha * (reward + gamma * next_max - old_value)

# Q-learning algorithm
def q_learning(puzzle, num_episodes):
    correct_groups = puzzle["correct_groups"]
    all_words = puzzle["all_words"]

    for episode in range(num_episodes):
        state = {
            "correct_groups": [],
            "incorrect_groups": [],
            "remaining_words": all_words.copy()
        }
        done = False

        while not done:
            # Epsilon-greedy action selection
            possible_actions = generate_possible_actions(state["remaining_words"])
            if random.uniform(0, 1) < epsilon:
                action = random.choice(possible_actions)  # Explore
            else:
                # Exploit: choose the action with the max Q-value that’s not in incorrect_groups
                action = max((a for a in possible_actions if set(a) not in state["incorrect_groups"]),
                             key=lambda x: Q_table.get((state, x), 0), default=random.choice(possible_actions))

            # Determine reward and next state
            reward = reward_function(action, correct_groups)
            if reward == 1:
                # Update state if action was correct
                state["correct_groups"].append(set(action))
                state["remaining_words"] = [w for w in state["remaining_words"] if w not in action]
            else:
                # Add to incorrect groups to avoid repeating
                state["incorrect_groups"].append(set(action))
            
            # Check if the puzzle is solved
            done = len(state["correct_groups"]) == 4
            
            # Update Q-table
            next_state = state.copy()
            update_q_table(str(state), action, reward, str(next_state))

# Train the agent
q_learning(puzzle, num_episodes=500)

# Evaluate the agent on the puzzle
def evaluate_agent(puzzle):
    state = {
        "correct_groups": [],
        "incorrect_groups": [],
        "remaining_words": puzzle["all_words"].copy()
    }
    guesses = 0
    done = False
    while not done:
        possible_actions = generate_possible_actions(state["remaining_words"])
        action = max((a for a in possible_actions if set(a) not in state["incorrect_groups"]),
                     key=lambda x: Q_table.get((str(state), x), 0), default=random.choice(possible_actions))
        
        # Update the state and reward based on action
        if is_correct_grouping(action, puzzle["correct_groups"]):
            state["correct_groups"].append(set(action))
            state["remaining_words"] = [w for w in state["remaining_words"] if w not in action]
        else:
            state["incorrect_groups"].append(set(action))
        
        guesses += 1
        done = len(state["correct_groups"]) == 4
    
    print(f"Solved the puzzle in {guesses} guesses")

# Evaluate the trained agent
evaluate_agent(puzzle)
