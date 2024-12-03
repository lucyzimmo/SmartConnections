import torch
import torch.nn as nn
import torch.optim as optim
import random
from itertools import combinations
from gensim.models import Word2Vec

# Deep Q-learning Parameters
alpha = 0.001
gamma = 0.9
embedding_dim = 300

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
def train_word2vec(puzzles):
    words = [word for puzzle in puzzles for group in puzzle["answers"] for word in group["members"]]
    model = Word2Vec([words], vector_size=embedding_dim, window=5, min_count=1, workers=4)
    return model

# Embed words
def embed_words(words, model):
    if not words:
        return torch.zeros(embedding_dim, dtype=torch.float32)  # Return a zero-vector if no words
    valid_vectors = [model.wv[word] for word in words if word in model.wv]
    if not valid_vectors:
        return torch.zeros(embedding_dim, dtype=torch.float32)  # Return zero-vector if no valid words
    return torch.tensor(sum(valid_vectors) / len(valid_vectors), dtype=torch.float32)


# Safe loading for PyTorch
def load_q_network(filepath, input_dim):
    q_network = QNetwork(input_dim)
    try:
        q_network.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    except RuntimeError as e:
        print(f"Failed to load Q-network. Check the file or consider setting weights_only=True.\n{e}")
        raise
    return q_network

def train_q_network(puzzles, word2vec_model, num_episodes, q_network, optimizer, loss_fn):
    import time  # For timing episodes
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        print(f"\nStarting episode {episode + 1}/{num_episodes}...")

        for puzzle_idx, puzzle in enumerate(puzzles):
            print(f"  Training on puzzle {puzzle_idx + 1}/{len(puzzles)}...")
            
            correct_groups = puzzle["answers"]
            all_words = [word for group in correct_groups for word in group["members"]]
            state = {
                "correct_groups": [],
                "incorrect_groups": [],
                "remaining_words": all_words.copy()
            }

            done = False
            step_count = 0
            while not done:
                step_count += 1
                # Embed the current state
                state_embedding = embed_words(state["remaining_words"], word2vec_model)

                # Generate possible actions
                possible_actions = list(combinations(state["remaining_words"], 4))
                if step_count % 50 == 0:
                    print(f"    Step {step_count}: {len(possible_actions)} possible actions to evaluate...")

                # Epsilon-greedy action selection
                if random.uniform(0, 1) < 0.3:  # Exploration
                    action = random.choice(possible_actions)
                else:  # Exploitation
                    action = max(
                        possible_actions,
                        key=lambda a: q_network(
                            torch.cat((state_embedding, embed_words(a, word2vec_model)))
                        ).item()
                    )

                # Compute reward and next state
                reward = 1 if any(set(action) == set(group["members"]) for group in correct_groups) else -1
                next_state = state.copy()

                if reward == 1:
                    next_state["correct_groups"].append(set(action))
                    next_state["remaining_words"] = [
                        word for word in next_state["remaining_words"] if word not in action
                    ]
                else:
                    # Only add the incorrect group if it's not already present
                    if set(action) not in next_state["incorrect_groups"]:
                        next_state["incorrect_groups"].append(set(action))

                # Check if puzzle is solved
                done = len(next_state["correct_groups"]) == 4

                # Compute target Q-value
                if next_state["remaining_words"]:
                    next_state_embedding = embed_words(next_state["remaining_words"], word2vec_model)
                    next_q_value = max(
                        q_network(
                            torch.cat((next_state_embedding, embed_words(a, word2vec_model)))
                        ).item()
                        for a in combinations(next_state["remaining_words"], 4)
                    )
                else:
                    next_q_value = 0  # Terminal state

                target_q_value = reward + 0.9 * next_q_value  # Gamma = 0.9

                # Update Q-network
                optimizer.zero_grad()
                predicted_q_value = q_network(
                    torch.cat((state_embedding, embed_words(action, word2vec_model)))
                )
                loss = loss_fn(predicted_q_value, torch.tensor([target_q_value], dtype=torch.float32))
                loss.backward()
                optimizer.step()

                # Update state
                state = next_state

            print(f"  Puzzle {puzzle_idx + 1}/{len(puzzles)} completed in {step_count} steps.")

        episode_duration = time.time() - episode_start
        print(f"Episode {episode + 1} completed in {episode_duration:.2f} seconds.")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.2f} seconds.")

# Generate all possible groupings of four words as actions
def generate_possible_actions(words):
    return list(combinations(words, 4))

# Evaluate using deep Q-learning
def evaluate_deepq(puzzle, q_network, word2vec_model):
    correct_groups = puzzle["solution"]
    all_words = puzzle["puzzle_words"]
    state = {
        "correct_groups": [],
        "incorrect_groups": [],
        "remaining_words": all_words.copy()
    }

    print("\nStarting evaluation:")
    guesses = 0
    done = False

    while not done:
        state_embedding = embed_words(state["remaining_words"], word2vec_model).unsqueeze(0)  # Ensure 2D tensor
        possible_actions = [
            action for action in generate_possible_actions(state["remaining_words"])
            if set(action) not in state["correct_groups"] and set(action) not in state["incorrect_groups"]
        ]

        action = max(
            possible_actions,
            key=lambda a: q_network(
                torch.cat((
                    state_embedding, 
                    embed_words(a, word2vec_model).unsqueeze(0)  # Ensure 2D tensor
                ), dim=1)
            ).item()
        )

        is_correct = any(set(action) == set(group["members"]) for group in correct_groups)
        if is_correct:
            state["correct_groups"].append(set(action))
            state["remaining_words"] = [w for w in state["remaining_words"] if w not in action]
        else:  # Incorrect guess
            if set(action) not in state["incorrect_groups"]:
                state["incorrect_groups"].append(set(action))


        guesses += 1
        done = len(state["correct_groups"]) == 4
        
        print(f"Current correct groups: {state['correct_groups']}")
        print(f"Remaining words: {state['remaining_words']}")
        print(f"Incorrect groups so far: {state['incorrect_groups']}")

    print(f"\nPuzzle solved in {guesses} guesses!\nFinal correct groups: {state['correct_groups']}")

    print(f"Puzzle solved in {guesses} guesses.")
    return guesses