import json
import random
from itertools import combinations

# Baseline Q-learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.3  # Exploration rate
num_episodes = 500  # Number of training episodes

# Initialize Q-table as a dictionary
Q_table = {}

# Function to check if a grouping is correct
def is_correct_grouping(action, correct_groups):
    return any(set(action) == set(group["members"]) for group in correct_groups)

# Generate all possible groupings of four words as actions
def generate_possible_actions(words):
    return list(combinations(words, 4))

# Reward function: +1 for a correct grouping, -1 for incorrect
def reward_function(action, correct_groups):
    return 1 if is_correct_grouping(action, correct_groups) else -1

# Update Q-value function
def update_q_table(state, action, reward, next_state):
    state_action = (str(state), action)
    old_value = Q_table.get(state_action, 0)
    
    # Handle empty possible actions
    possible_actions = generate_possible_actions(state['remaining_words'])
    if possible_actions:
        next_max = max(Q_table.get((str(next_state), a), 0) for a in possible_actions)
    else:
        next_max = 0
    
    Q_table[state_action] = old_value + alpha * (reward + gamma * next_max - old_value)

# Q-learning algorithm for multiple puzzles
def q_learning(puzzles, num_episodes):
    for episode in range(num_episodes):
        for puzzle in puzzles:
            correct_groups = puzzle["answers"]
            all_words = [word for group in correct_groups for word in group["members"]]
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
                    # Exploit: choose the action with the max Q-value
                    action = max(
                        (a for a in possible_actions if set(a) not in state["incorrect_groups"]),
                        key=lambda x: Q_table.get((str(state), x), 0),
                        default=random.choice(possible_actions)
                    )

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
                update_q_table(state, action, reward, next_state)

# Evaluate the baseline method on a puzzle
def evaluate_baseline(puzzle):
    print(puzzle)
    correct_groups = puzzle["answers"]
    all_words = puzzle["puzzle_words"]
    state = {
        "correct_groups": [],
        "incorrect_groups": [],
        "remaining_words": all_words.copy()
    }
    guesses = 0
    done = False

    while not done:
        possible_actions = generate_possible_actions(state["remaining_words"])
        action = max(
            (a for a in possible_actions if set(a) not in state["incorrect_groups"]),
            key=lambda x: Q_table.get((str(state), x), 0),
            default=random.choice(possible_actions)
        )

        is_correct = is_correct_grouping(action, correct_groups)
        if is_correct:
            state["correct_groups"].append(set(action))
            state["remaining_words"] = [w for w in state["remaining_words"] if w not in action]
        else:
            state["incorrect_groups"].append(set(action))

        guesses += 1
        done = len(state["correct_groups"]) == 4

    return guesses