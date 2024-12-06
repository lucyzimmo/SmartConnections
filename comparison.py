import json
import random
import os
from baseline import evaluate_baseline, q_learning
from generateRandom import generate_random_puzzle_from_past_words
from deepq import train_word2vec, QNetwork, evaluate_q_network
from deepq_experiment import train_word2vec_experiment, evaluate_q_network_experiment
from deepq_experiment2 import train_word2vec_experiment2, evaluate_q_network_experiment2, ReplayBuffer
import torch
import torch.optim as optim
import torch.nn as nn
import copy

# Function to load and format a random puzzle from small_test.json
def pick_and_format_puzzle_from_small_test_json():
    with open('data/small_test.json', 'r') as f:
        puzzles = json.load(f)
    selected_puzzle = random.choice(puzzles)
    puzzle_words = []
    puzzle_solution = []
    for group in selected_puzzle["answers"]:
        category = group["group"]
        words = group["members"]
        puzzle_words.extend(words)
        puzzle_solution.append({"group": category, "members": words})
    random.shuffle(puzzle_words)
    return {
        "puzzle_words": puzzle_words,
        "solution": puzzle_solution
    }

# Main function
def main():
    # Load puzzles
    with open('data/small.json', 'r') as f:
        puzzles = json.load(f)

    # Train Word2Vec
    print("Training Word2Vec...")
    word2vec_model = train_word2vec(puzzles)

    # Initialize Q-networks
    input_dim = 600  # 300 (state embedding) + 300 (action embedding)
    q_network = QNetwork(input_dim)
    q_network_experiment = QNetwork(input_dim)
    q_network_experiment2 = QNetwork(input_dim)

    # Load pre-trained weights
    print("Loading pre-trained Q-network weights...")
    q_network.load_state_dict(torch.load("q_network_small.pth"))
    q_network.eval()

    print("Loading pre-trained Q-network weights for Experiment 1...")
    q_network_experiment.load_state_dict(torch.load("q_network_experiment.pth"))
    q_network_experiment.eval()

    print("Loading pre-trained Q-network weights for Experiment 2...")
    q_network_experiment2.load_state_dict(torch.load("q_network_experiment2.pth"))
    q_network_experiment2.eval()

    # Run evaluations 10 times
    baseline_results = []
    deepq_results = []
    experiment1_results = []
    experiment2_results = []

    for i in range(10):
        print(f"\nRun {i + 1}/10")

        # Pick and format a random puzzle
        puzzle = pick_and_format_puzzle_from_small_test_json()

        # Evaluate Baseline Method
        baseline_puzzle = copy.deepcopy(puzzle)
        baseline_guesses = evaluate_baseline(baseline_puzzle)
        baseline_results.append(baseline_guesses)

        # Evaluate Standard Q-network
        deepq_puzzle = copy.deepcopy(puzzle)
        deepq_guesses = evaluate_q_network(deepq_puzzle, q_network, word2vec_model)
        deepq_results.append(deepq_guesses)

        # Evaluate Experiment 1 Q-network
        experiment1_puzzle = copy.deepcopy(puzzle)
        experiment1_guesses = evaluate_q_network_experiment(experiment1_puzzle, q_network_experiment, word2vec_model)
        experiment1_results.append(experiment1_guesses)

        # Evaluate Experiment 2 Q-network
        experiment2_puzzle = copy.deepcopy(puzzle)
        experiment2_guesses = evaluate_q_network_experiment2(experiment2_puzzle, q_network_experiment2, word2vec_model)
        experiment2_results.append(experiment2_guesses)

    # Summarize results
    def summarize_results(results):
        return {
            "average": sum(results) / len(results),
            "min": min(results),
            "max": max(results),
        }

    baseline_summary = summarize_results(baseline_results)
    deepq_summary = summarize_results(deepq_results)
    experiment1_summary = summarize_results(experiment1_results)
    experiment2_summary = summarize_results(experiment2_results)

    # Display results
    print("\nResults Summary (Average, Min, Max Guesses):")
    print(f"Baseline: {baseline_summary}")
    print(f"Standard Q-Network: {deepq_summary}")
    print(f"Experiment 1 Q-Network: {experiment1_summary}")
    print(f"Experiment 2 Q-Network: {experiment2_summary}")

if __name__ == "__main__":
    main()
