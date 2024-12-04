import torch
from deepq import QNetwork

# Load the Q-network
input_dim = 600  # Replace with the correct input dimension used during training
q_network = QNetwork(input_dim)
q_network.load_state_dict(torch.load("q_network.pth", map_location=torch.device("cpu")))

# Print the model architecture
print("Q-network architecture:")
print(q_network)

# Inspect weights
print("\nQ-network weights:")
for name, param in q_network.named_parameters():
    print(f"{name}: {param.data}")
