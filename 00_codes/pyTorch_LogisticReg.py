# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# %%
# 2. Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer (input_dim -> 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation for binary classification



# %%
# 1. Generate random dataset (features and labels)
# Features are two-dimensional, labels are binary (0 or 1)
np.random.seed(42)
torch.manual_seed(42)

X_train = np.random.randn(100, 2)  # 100 samples, 2 features
y_train = (np.sum(X_train, axis=1) > 0).astype(int)  # Label is 1 if sum of features > 0, else 0
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

# %%

# %%

# %%

# %%

# %% [markdown]
# ### 3. Initialize the model, loss function, and optimizer

# %%
# 3. Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss 
# criterion = nn.BCEWithLogitsLoss() even better? 
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    y_pred = model(X_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Track the loss
    losses.append(loss.item())

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# %%

# 5. Plot the loss over epochs
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

# 6. Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_predicted = model(X_train)
    y_predicted = (y_predicted > 0.5).float()  # Convert probabilities to 0 or 1

# Calculate accuracy
accuracy = (y_predicted == y_train).sum().item() / len(y_train)
print(f'Accuracy: {accuracy * 100:.2f}%')

# %%
