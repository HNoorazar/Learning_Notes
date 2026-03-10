# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3.13.11 (Conda)
#     language: python
#     name: py313
# ---

# %%
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# %% [markdown]
# #### The ```Dataset``` class
#
#
# In PyTorch, a dataset is represented by a regular Python class that inherits from the Dataset class. You can think of it as a list of tuples, each tuple corresponding to one point (features, label).
#
# - ```__init__(self)```: It takes whatever arguments are needed to build a list of tuples; it may be the name of a CSV file that will be loaded and processed; it may be two tensors with one for features and another one for labels; or anything else, depending on the task at hand.
#
# ----------------------------
#
# There is no need to load the whole dataset in the constructor method (```__init__```). If your dataset is big (e.g., tens of thousands of image files), loading it at once would not be memory efficient. It is recommended to load them on demand whenever ```__get_item__``` is called.
#
# ----------------------------
#
#
# - ```__get_item__(self, index)```: It allows the dataset to be indexed so that it can work like a list (```dataset[i]```). It should return a tuple (features, label) corresponding to the requested data point. We can either return the corresponding slices of our pre-loaded dataset or, as mentioned above, load them on demand (like in this example).
#
# - ```__len__(self)```: It should simply return the size of the whole dataset. So, whenever it is sampled, its indexing is limited to the actual size.

# %%
true_b = 1
true_w = 2
N = 100

# Data generation process
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
val_idx = idx[int(N*.8):]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

print('x_train: {}'.format(x_train[:5]))


# %%

# %%
class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

# Wait, is this a CPU tensor now? Why? Where is .to(device)?
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

# %% [markdown]
# ### Tensor Dataset

# %%
train_data = TensorDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

# %% [markdown]
# ### We use it for mini-batch GD
#
#
# There is more to a ```DataLoader``` than meets the eye. For instance, it is also possible to use it together with a sampler to fetch mini-batches that compensate for imbalanced classes.

# %%
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

# %%
# will return a list -> 2 tensors (features, labels)
print(next(iter(train_loader)))

# %% [markdown]
# ### Use the dataset and data loader in mini-batch

# %%
n_epochs = 1000

losses = []
for epoch in range(n_epochs):
    # inner loop
    mini_batch_losses = []                              
    for x_batch, y_batch in train_loader:
        # the dataset "lives" in the CPU, so do our mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Performs one train step and returns the 
        # corresponding loss for this mini-batch
        mini_batch_loss = train_step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    # Computes average loss over all mini-batches (That's the epoch loss)
    loss = np.mean(mini_batch_losses)
    losses.append(loss)
