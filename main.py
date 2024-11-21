import json
import random
import os
from baseline import evaluate_baseline, q_learning
from deepq import train_word2vec, QNetwork, evaluate_deepq, train_q_network
import torch
import torch.optim as optim
import torch.nn as nn

# Generate a random puzzle for evaluation
def generate_random_puzzle():
    with open('data/small.json', 'r') as f:
        puzzles = json.load(f)
    return random.choice(puzzles)

# Main function
def main():
    # Load puzzles
    with open('data/small.json', 'r') as f:
        puzzles = json.load(f)

    # Train baseline Q-learning
    print("Training baseline Q-learning...")
    q_learning(puzzles, 500)

    # Train Word2Vec
    print("Training Word2Vec...")
    word2vec_model = train_word2vec(puzzles)

    # Initialize Q-network
    input_dim = 600  # 300 (state embedding) + 300 (action embedding)
    q_network = QNetwork(input_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Check if Q-network weights exist
    if not os.path.exists("q_network.pth"):
        print("Q-network weights not found. Training the Q-network...")
        train_q_network(puzzles, word2vec_model, num_episodes=10, q_network=q_network, optimizer=optimizer, loss_fn=loss_fn)
        torch.save(q_network.state_dict(), "q_network.pth")
        print("Q-network training complete and weights saved to q_network.pth")
    else:
        print("Loading pre-trained Q-network weights...")
        q_network.load_state_dict(torch.load("q_network.pth"))
        q_network.eval()

    # Generate a random puzzle
    puzzle = generate_random_puzzle()

    # Evaluate both methods
    print("\nEvaluating Baseline Method...")
    baseline_guesses = evaluate_baseline(puzzle)
    print("\nEvaluating Deep Q-Learning Method...")
    deepq_guesses = evaluate_deepq(puzzle, q_network, word2vec_model)

    # Compare results
    print("\nComparison:")
    print(f"Baseline guesses: {baseline_guesses}")
    print(f"Deep Q-Learning guesses: {deepq_guesses}")
    if baseline_guesses < deepq_guesses:
        print("Baseline performed better.")
    elif deepq_guesses < baseline_guesses:
        print("Deep Q-Learning performed better.")
    else:
        print("Both methods performed equally well.")

if __name__ == "__main__":
    main()