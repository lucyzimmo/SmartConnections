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
import numpy as np
import matplotlib.pyplot as plt

# Function to load and format a random puzzle from small_test.json
def pick_and_format_puzzle_from_small_test_json():
    with open('data/medium_test.json', 'r') as f:
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

# Summarize results
def summarize_results(results):
    return {
        "average": np.mean(results),
        "median": np.median(results),
        "min": min(results),
        "max": max(results),
    }

# Plot results
def plot_results(methods, averages, medians, mins, maxs):
    x = np.arange(len(methods))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, averages, 0.2, label="Average")
    plt.bar(x, medians, 0.2, label="Median")
    plt.bar(x + 0.2, mins, 0.2, label="Min")
    plt.bar(x + 0.4, maxs, 0.2, label="Max")
    plt.xticks(x + 0.1, methods)
    plt.xlabel("Method")
    plt.ylabel("Guesses")
    plt.title("Performance Comparison")
    plt.legend()
    plt.show()

# Main function
def main():
    # Load puzzles
    with open('data/medium.json', 'r') as f:
        puzzles = json.load(f)

    print("Training baseline Q-learning...")
    q_learning(puzzles, 500)

    # Train Word2Vec
    print("Training Word2Vec...")
    word2vec_model = train_word2vec(puzzles)

    # Initialize Q-networks
    input_dim = 600  # 300 (state embedding) + 300 (action embedding)
    q_network = QNetwork(input_dim)
    q_network_experiment = QNetwork(input_dim)

    # Load pre-trained weights
    print("Loading pre-trained Q-network weights...")
    q_network.load_state_dict(torch.load("q_network_medium.pth"))
    q_network.eval()

    print("Loading pre-trained Q-network weights for Experiment 1...")
    q_network_experiment.load_state_dict(torch.load("q_network_experiment_medium.pth"))
    q_network_experiment.eval()


    # Run evaluations 100 times
    baseline_results = []
    deepq_results = []
    experiment1_results = []

    for i in range(100):
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


    # Summarize results
    baseline_summary = summarize_results(baseline_results)
    deepq_summary = summarize_results(deepq_results)
    experiment1_summary = summarize_results(experiment1_results)

    # Display results
    print("\nResults Summary (Average, Median, Min, Max Guesses):")
    print(f"Baseline: {baseline_summary}")
    print(f"Standard Q-Network: {deepq_summary}")
    print(f"Experiment 1 Q-Network: {experiment1_summary}")

    # Plot results
    methods = ["Baseline", "Standard Q-Network", "Experiment 1 Q-Network"]
    averages = [baseline_summary["average"], deepq_summary["average"], experiment1_summary["average"]]
    medians = [baseline_summary["median"], deepq_summary["median"], experiment1_summary["median"]]
    mins = [baseline_summary["min"], deepq_summary["min"], experiment1_summary["min"]]
    maxs = [baseline_summary["max"], deepq_summary["max"], experiment1_summary["max"]]

    plot_results(methods, averages, medians, mins, maxs)

if __name__ == "__main__":
    main()
