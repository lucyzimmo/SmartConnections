"""Experiments: 
Dynamic epsilon Decay: Encourages exploration early and exploitation later.
Nuanced Rewards: Introduced a slight penalty for incorrect guesses (-0.1) to provide better guidance.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from itertools import combinations
from gensim.models import Word2Vec

# Hyperparameters
alpha = 0.001
gamma = 0.9
embedding_dim = 300
epsilon_start = 0.3
epsilon_decay = 0.995
epsilon_min = 0.1

# Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

# Train Word2Vec
def train_word2vec_experiment(puzzles):
    words = [word for puzzle in puzzles for group in puzzle["answers"] for word in group["members"]]
    model = Word2Vec([words], vector_size=embedding_dim, window=5, min_count=1, workers=4)
    return model

# Embed words using Word2Vec
def embed_words(words, model):
    if not words:
        return torch.zeros(embedding_dim, dtype=torch.float32)
    valid_vectors = [model.wv[word] for word in words if word in model.wv]
    if not valid_vectors:
        return torch.zeros(embedding_dim, dtype=torch.float32)
    return torch.tensor(sum(valid_vectors) / len(valid_vectors), dtype=torch.float32)

# Train Q-network
def train_q_network(puzzles, word2vec_model, num_episodes, q_network, optimizer, loss_fn, save_path="q_network.pth"):
    epsilon = epsilon_start
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}/{num_episodes}")
        for puzzle_idx, puzzle in enumerate(puzzles):
            correct_groups = puzzle["answers"]
            all_words = [word for group in correct_groups for word in group["members"]]
            state = {"correct_groups": [], "incorrect_groups": [], "remaining_words": all_words.copy()}

            done = False
            while not done:
                state_embedding = embed_words(state["remaining_words"], word2vec_model)

                # Precompute embeddings for possible actions
                possible_actions = list(combinations(state["remaining_words"], 4))
                action_embeddings = {action: embed_words(action, word2vec_model) for action in possible_actions}

                # Epsilon-greedy action selection
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(possible_actions)
                else:
                    action = max(possible_actions, key=lambda a: q_network(torch.cat((state_embedding, action_embeddings[a]))).item())

                # Reward and next state
                reward = 1 if any(set(action) == set(group["members"]) for group in correct_groups) else -0.1
                next_state = state.copy()

                if reward == 1:
                    next_state["correct_groups"].append(set(action))
                    next_state["remaining_words"] = [word for word in next_state["remaining_words"] if word not in action]
                else:
                    if set(action) not in next_state["incorrect_groups"]:
                        next_state["incorrect_groups"].append(set(action))

                # Check if solved
                done = len(next_state["correct_groups"]) == 4

                # Compute target Q-value
                if next_state["remaining_words"]:
                    next_state_embedding = embed_words(next_state["remaining_words"], word2vec_model)
                    next_q_value = max(
                        q_network(torch.cat((next_state_embedding, action_embeddings[a]))).item()
                        for a in combinations(next_state["remaining_words"], 4)
                    )
                else:
                    next_q_value = 0  # Terminal state

                target_q_value = reward + gamma * next_q_value

                # Update Q-network
                optimizer.zero_grad()
                predicted_q_value = q_network(torch.cat((state_embedding, action_embeddings[action])))
                loss = loss_fn(predicted_q_value, torch.tensor([target_q_value], dtype=torch.float32))
                loss.backward()
                optimizer.step()

                # Update state
                state = next_state

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1} completed. Epsilon: {epsilon:.4f}")

        # Save model checkpoint
        torch.save(q_network.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# Evaluate Q-network
def evaluate_q_network_experiment(puzzle, q_network, word2vec_model):
    correct_groups = puzzle["solution"]
    all_words = puzzle["puzzle_words"]
    state = {
        "correct_groups": [],
        "incorrect_groups": [],
        "remaining_words": all_words.copy(),
        "guess_history": set()
    }

    print("\nStarting Evaluation:")
    guesses = 0
    done = False

    while not done:
        guesses += 1

        # Embed current state
        state_embedding = embed_words(state["remaining_words"], word2vec_model)

        # Generate possible actions excluding guessed ones
        possible_actions = [
            a for a in combinations(state["remaining_words"], 4) if tuple(a) not in state["guess_history"]
        ]

        # Stop if no more actions are possible
        if not possible_actions:
            print("No more unguessed actions available. Stopping evaluation.")
            break

        # Select the action with the highest Q-value
        action_scores = {
            a: q_network(torch.cat((state_embedding, embed_words(a, word2vec_model)))).item()
            for a in possible_actions
        }
        action = max(action_scores, key=action_scores.get)
        state["guess_history"].add(tuple(action))  # Mark the action as guessed

        print(f"Guess #{guesses}: {action} (Q-value: {action_scores[action]:.4f})")

        # Check if the guess is correct
        if any(set(action) == set(group["members"]) for group in correct_groups):
            print(f"Correct guess: {action}")
            state["correct_groups"].append(set(action))
            state["remaining_words"] = [w for w in state["remaining_words"] if w not in action]
        else:
            print(f"Incorrect guess: {action}")
            state["incorrect_groups"].append(set(action))

        # Check if the puzzle is solved
        done = len(state["correct_groups"]) == len(correct_groups)


    print(f"\nPuzzle solved in {guesses} guesses!")
    print(f"Correct groups: {state['correct_groups']}")
    print(f"Incorrect groups: {state['incorrect_groups']}")
    return guesses
