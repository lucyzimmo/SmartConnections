import json
import random
import os
from baseline import evaluate_baseline, q_learning
from generateRandom import generate_random_puzzle_from_past_words
from deepq_experiment import train_word2vec_experiment, QNetwork, evaluate_q_network_experiment, train_q_network_experiment
import torch
import torch.optim as optim
import torch.nn as nn
import copy

# Function to load and format a random puzzle from small_test.json
def pick_and_format_puzzle_from_small_test_json():
    # Load the test puzzles
    with open('data/small_test.json', 'r') as f:
        puzzles = json.load(f)
    
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
    with open('data/small.json', 'r') as f:
        puzzles = json.load(f)

    # Train baseline Q-learning
    #print("Training baseline Q-learning...")
    #q_learning(puzzles, 500)

    # Train Word2Vec
    print("Training Word2Vec...")
    word2vec_model = train_word2vec(puzzles)

    # Initialize Q-network
    input_dim = 600  # 300 (state embedding) + 300 (action embedding)
    q_network = QNetwork(input_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Check if Q-network weights exist
    if not os.path.exists("q_network_small.pth"):
        print("Q-network weights not found. Training the Q-network...")
        train_q_network(puzzles, word2vec_model, num_episodes=10, q_network=q_network, optimizer=optimizer, loss_fn=loss_fn)
        torch.save(q_network.state_dict(), "q_network_small.pth")
        print("Q-network training complete and weights saved to q_network.pth")
    else:
        print("Loading pre-trained Q-network weights...")
        q_network.load_state_dict(torch.load("q_network_experiment_medium.pth"))
        q_network.eval()

    # Pick and format a random puzzle from test.json
    puzzle = pick_and_format_puzzle_from_small_test_json()

    # Evaluate both methods
    print("\nEvaluating Baseline Method...")
    baseline_puzzle = copy.deepcopy(puzzle)  # Create a deep copy
    baseline_guesses = evaluate_baseline(baseline_puzzle)
    print("\nEvaluating Deep Q-Learning Method...")
    deepq_puzzle = copy.deepcopy(puzzle)  # Create a deep copy
    deepq_guesses = evaluate_q_network(deepq_puzzle, q_network, word2vec_model)
    # Compare results
    print("\nComparison:")
    print(baseline_guesses)
    print(deepq_guesses)
    if baseline_guesses < deepq_guesses:
        print("Baseline performed better.")
    elif deepq_guesses < baseline_guesses:
        print("Deep Q-Learning performed better.")
    else:
        print("Both methods performed equally well.")

if __name__ == "__main__":
    main()
