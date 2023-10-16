import torch

from data import DataGenerator
from model import Regressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data generator
generator = DataGenerator(20, 2)

# Create model
model = Regressor(20).to(device)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create loss function
criterion = torch.nn.MSELoss()

num_epochs = 1000

for epoch in range(num_epochs):
    
        # Zero out gradients
        optimizer.zero_grad()

        # Sample data
        samples, scores, mode_idx = generator.sample(1000)

        # Move data to device
        samples = samples.to(device)
        scores = scores.to(device)
    
        # Get model outputs
        outputs, _ = model(samples)

        # Compute loss
        loss = criterion(outputs, scores)

        # Compute gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print loss
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))



