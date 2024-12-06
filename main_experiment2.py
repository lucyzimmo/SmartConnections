import json
import random
import os
from baseline import evaluate_baseline, q_learning
from generateRandom import generate_random_puzzle_from_past_words
from deepq_experiment2 import train_word2vec, QNetwork, evaluate_q_network, train_q_network, ReplayBuffer
import torch
import torch.optim as optim
import torch.nn as nn
import copy

# Constants
embedding_dim = 300  # Make sure this matches Word2Vec vector size

# Function to load and format a random puzzle
def pick_and_format_puzzle_from_small_test_json():
    try:
        # Load the test puzzles
        with open('data/small_test.json', 'r') as f:
            puzzles = json.load(f)
    except FileNotFoundError:
        print("Error: 'data/small_test.json' not found.")
        return None

    # Pick a random puzzle
    selected_puzzle = random.choice(puzzles)
    
    # Extract the solution in the required format
    puzzle_words = []
    puzzle_solution = []
    for group in selected_puzzle["answers"]:
        category = group["group"]
        words = group["members"]
        puzzle_words.extend(words)
        puzzle_solution.append({"group": category, "members": words})
    
    # Shuffle the puzzle words
    random.shuffle(puzzle_words)
    
    # Return the structured puzzle
    return {
        "puzzle_words": puzzle_words,
        "solution": puzzle_solution
    }

# Main function
def main():
    # Load puzzles
    try:
        with open('data/medium.json', 'r') as f:
            puzzles = json.load(f)
    except FileNotFoundError:
        print("Error: 'data/medium.json' not found.")
        return

    # Train Word2Vec
    print("Training Word2Vec...")
    word2vec_model = train_word2vec(puzzles)

    # Initialize Q-network
    input_dim = embedding_dim * 2  # State embedding + action embedding
    q_network = QNetwork(input_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(capacity=10000)

    # Check if Q-network weights exist
    if not os.path.exists("q_network.pth"):
        print("Q-network weights not found. Training the Q-network...")
        train_q_network(
            puzzles=puzzles,
            word2vec_model=word2vec_model,
            num_episodes=10,
            q_network=q_network,
            optimizer=optimizer,
            loss_fn=loss_fn,
            replay_buffer=replay_buffer,
        )
        torch.save(q_network.state_dict(), "q_network.pth")
        print("Q-network training complete and weights saved to 'q_network.pth'.")
    else:
        print("Loading pre-trained Q-network weights...")
        try:
            q_network.load_state_dict(torch.load("q_network.pth"))
            q_network.eval()
        except Exception as e:
            print(f"Error loading Q-network weights: {e}")
            return

    # Evaluate the Q-network
    print("Evaluating the Q-network...")
    puzzle = pick_and_format_puzzle_from_small_test_json()
    if puzzle:
        evaluate_q_network(puzzle, q_network, word2vec_model)
    else:
        print("Error: Puzzle data is invalid.")

if __name__ == "__main__":
    main()
