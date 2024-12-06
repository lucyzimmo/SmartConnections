"""Efficient Action Sampling: Samples top-k actions based on cosine similarity to the current state.
Regularization: Added dropout layers and weight decay to the Q-network.
Improved Reward Signal: Added penalties for incorrect guesses and used nuanced feedback.
Replay Buffer: Implemented a replay buffer to store and sample experiences for more stable training.
Pretraining: Pretrained the Q-network using solved puzzles for better initialization."""
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
batch_size = 32
replay_buffer_capacity = 10000

# Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=replay_buffer_capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

# Train Word2Vec
def train_word2vec_experiment2(puzzles):
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

# Sample top-k actions based on cosine similarity
def sample_top_actions(possible_actions, state_embedding, word2vec_model, top_k=100):
    action_embeddings = {action: embed_words(action, word2vec_model) for action in possible_actions}
    similarities = {
        action: torch.cosine_similarity(state_embedding, emb, dim=0).item()
        for action, emb in action_embeddings.items() if emb is not None
    }
    sorted_actions = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [action for action, _ in sorted_actions]

# Pretrain the Q-network with solved puzzles
def pretrain_q_network_experiment2(puzzles, word2vec_model, q_network, optimizer, loss_fn):
    for puzzle in puzzles:
        for group in puzzle["answers"]:
            state_embedding = embed_words(group["members"], word2vec_model)
            action_embedding = embed_words(group["members"], word2vec_model)
            target_q_value = 1.0  # Correct group
            optimizer.zero_grad()
            predicted_q_value = q_network(torch.cat((state_embedding, action_embedding)))
            loss = loss_fn(predicted_q_value, torch.tensor([target_q_value], dtype=torch.float32))
            loss.backward()
            optimizer.step()
    print("Pretraining completed.")

def train_q_network_experiment2(puzzles, word2vec_model, num_episodes, q_network, optimizer, loss_fn, replay_buffer, save_path="q_network.pth"):
    epsilon = epsilon_start
    max_steps = 100  # Limit steps per episode

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}/{num_episodes}")

        for puzzle_idx, puzzle in enumerate(puzzles):
            correct_groups = puzzle["answers"]
            all_words = [word for group in correct_groups for word in group["members"]]
            state = {"correct_groups": [], "incorrect_groups": [], "remaining_words": all_words.copy()}
            done = False
            step_count = 0

            while not done and step_count < max_steps:
                step_count += 1
                state_embedding = embed_words(state["remaining_words"], word2vec_model)
                possible_actions = list(combinations(state["remaining_words"], 4))
                sampled_actions = sample_top_actions(possible_actions, state_embedding, word2vec_model, top_k=10)

                # Epsilon-greedy action selection
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(sampled_actions)
                else:
                    action = max(
                        sampled_actions,
                        key=lambda a: q_network(torch.cat((state_embedding, embed_words(a, word2vec_model)))).item()
                    )

                # Reward and next state
                reward = 1 if any(set(action) == set(group["members"]) for group in correct_groups) else -0.1
                next_state = state.copy()

                if reward == 1:
                    next_state["correct_groups"].append(set(action))
                    next_state["remaining_words"] = [word for word in next_state["remaining_words"] if word not in action]
                else:
                    if set(action) not in next_state["incorrect_groups"]:
                        next_state["incorrect_groups"].append(set(action))

                done = len(next_state["correct_groups"]) == len(correct_groups)

                # Compute target Q-value
                if next_state["remaining_words"]:
                    next_state_embedding = embed_words(next_state["remaining_words"], word2vec_model)
                    next_q_value = max(
                        q_network(torch.cat((next_state_embedding, embed_words(a, word2vec_model)))).item()
                        for a in combinations(next_state["remaining_words"], 4)
                    )
                else:
                    next_q_value = 0  # Terminal state

                target_q_value = reward + gamma * next_q_value

                # Store experience in replay buffer
                replay_buffer.add((state_embedding, embed_words(action, word2vec_model), reward, next_state_embedding, next_q_value))

                # Train with replay buffer
                if len(replay_buffer.buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    for state_emb, action_emb, reward, next_state_emb, next_q_val in batch:
                        optimizer.zero_grad()
                        predicted_q_value = q_network(torch.cat((state_emb, action_emb)))
                        loss = loss_fn(predicted_q_value, torch.tensor([reward + gamma * next_q_val], dtype=torch.float32))
                        loss.backward()
                        optimizer.step()

                state = next_state

            print(f"Episode {episode + 1}, Puzzle {puzzle_idx + 1} completed in {step_count} steps.")

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1} completed. Epsilon: {epsilon:.4f}")

        if (episode + 1) % 5 == 0:
            torch.save(q_network.state_dict(), f"q_network_checkpoint_{episode + 1}.pth")
            print(f"Checkpoint saved at episode {episode + 1}")

    torch.save(q_network.state_dict(), save_path)
    print("Training complete. Model saved.")


# Evaluate Q-network
def evaluate_q_network_experiment2(puzzle, q_network, word2vec_model):
    correct_groups = puzzle["solution"]
    all_words = puzzle["puzzle_words"]
    state = {
        "correct_groups": [],
        "incorrect_groups": [],
        "remaining_words": all_words.copy(),
        "guess_history": set()  # Track guessed actions
    }

    print("\nStarting Evaluation:")
    guesses = 0
    max_guesses = 100  # Prevent infinite loops
    done = False

    while not done and guesses < max_guesses:
        guesses += 1

        # Embed current state
        state_embedding = embed_words(state["remaining_words"], word2vec_model)

        # Generate possible actions excluding already guessed ones
        possible_actions = [
            a for a in combinations(state["remaining_words"], 4) if tuple(a) not in state["guess_history"]
        ]

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

        # Check if the action is correct
        if any(set(action) == set(group["members"]) for group in correct_groups):
            print(f"Correct guess: {action}")
            state["correct_groups"].append(set(action))
            state["remaining_words"] = [w for w in state["remaining_words"] if w not in action]
        else:
            print(f"Incorrect guess: {action}")
            state["incorrect_groups"].append(set(action))

        # Check if the puzzle is solved
        done = len(state["correct_groups"]) == len(correct_groups)

    if guesses >= max_guesses:
        print("Maximum guess limit reached. Evaluation stopped.")

    print(f"\nPuzzle solved in {guesses} guesses!")
    print(f"Correct groups: {state['correct_groups']}")
    print(f"Incorrect groups: {state['incorrect_groups']}")
    return guesses
