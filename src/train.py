import torch
from model import SimpleCNN

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Number of model parameters:", sum(p.numel() for p in model.parameters()))
