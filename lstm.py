import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        
        # Use the last hidden state as the encoded representation
        encoded = hidden[-1]
        
        # Decoder
        decoded, _ = self.decoder(encoded.unsqueeze(1).repeat(1, x.size(1), 1))
        output = self.fc(decoded)
        
        return output, encoded

# Function to create synthetic 3D time series data
def create_synthetic_data(num_samples, seq_length, num_features):
    data = np.random.randn(num_samples, seq_length, num_features)
    return torch.FloatTensor(data)

# Set random seed for reproducibility
torch.manual_seed(42)

# Parameters
input_size = 3  # Number of features in your 3D time series
hidden_size = 64
num_layers = 2
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Create synthetic data
num_samples = 1000
seq_length = 50
X = create_synthetic_data(num_samples, seq_length, input_size)

# Create DataLoader
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = LSTMAutoencoder(input_size, hidden_size, num_layers)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        x = batch[0]
        
        # Forward pass
        output, _ = model(x)
        loss = criterion(output, x)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Plot loss curve
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Function to encode data
def encode_data(model, data):
    model.eval()
    with torch.no_grad():
        _, encoded = model(data)
    return encoded

# Encode the entire dataset
encoded_data = encode_data(model, X)

print("Shape of original data:", X.shape)
print("Shape of encoded data:", encoded_data.shape)

# Visualize original and encoded data (first two dimensions)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0, 0], X[:, 0, 1], alpha=0.5)
plt.title('Original Data (First Two Dimensions)')
plt.subplot(1, 2, 2)
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], alpha=0.5)
plt.title('Encoded Data (First Two Dimensions)')
plt.tight_layout()
plt.show()