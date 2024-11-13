import json
from itertools import combinations
import random
from collections import defaultdict

# Load the JSON data to generate a list of words and their categories from past puzzles
with open('data/connections.json', 'r') as f:
    puzzles = json.load(f)

# Build a dictionary of words categorized by their group names
word_categories = defaultdict(list)
for puzzle in puzzles:
    for group in puzzle["answers"]:
        category = group["group"]
        words = group["members"]
        word_categories[category].extend(words)

# Remove duplicates within each category to ensure uniqueness
for category in word_categories:
    word_categories[category] = list(set(word_categories[category]))

# Function to generate a random Connections puzzle
def generate_random_puzzle_from_past_words():
    # Randomly select 4 unique categories
    selected_categories = random.sample(list(word_categories.keys()), 4)
    
    # Pull 4 words from each selected category
    puzzle_words = []
    puzzle_solution = []
    for category in selected_categories:
        words = random.sample(word_categories[category], 4)
        puzzle_words.extend(words)
        puzzle_solution.append({"group": category, "members": words})
    
    # Shuffle the puzzle words to mix them up
    random.shuffle(puzzle_words)
    
    # Return the puzzle and solution in a structured format
    return {
        "puzzle_words": puzzle_words,
        "solution": puzzle_solution
    }