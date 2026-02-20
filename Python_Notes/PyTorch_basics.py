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
import sys
print(sys.executable)

# %% [markdown]
# At this time, Feb 18, 2026 Torch is not working with Python 3.13. Stop updating too quickly!

# %%
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot

# %% [markdown]
# A scalar (a single number) has zero dimensions, a vector has one dimension, a matrix has two dimensions, and a tensor has three or more dimensions. That's it!

# %%
scalar = torch.tensor(3.14159)
vector = torch.tensor([1, 2, 3])
matrix = torch.ones((2, 3), dtype=torch.float)
tensor = torch.randn((2, 3, 4), dtype=torch.float)

print(scalar)
L=50
print ("-"*L)
print(vector)
print ("-"*L)
print(matrix)
print ("-"*L)
print(tensor)
print ("-"*L)
print(tensor.size(), tensor.shape)
dummy_array = np.array([1, 2, 3])
dummy_tensor = torch.as_tensor(dummy_array)
# Modifies the numpy array
dummy_array[1] = 0
# Using numpy function to convert PyTorch tensor to Numpy array
print(dummy_tensor.numpy())
print(scalar.shape)

# %% [markdown]
# You can also reshape a tensor using its ```view()``` (preferred) or ```reshape()``` methods.
#
# **Beware**: The ```view()``` method only returns a tensor with the desired shape that shares the underlying data with the original tensor; it does not create a new, independent tensor!
#
# The ```reshape()``` method may or may not create a copy! The reasons behind this (apparently) weird behavior are beyond the scope of this lesson, but this behavior is the reason why ```view()``` is preferred.

# %%
matrix = torch.ones((2, 3), dtype=torch.float)
# We get a tensor with a different shape but it still is the SAME tensor
same_matrix = matrix.view(1, 6)
# If we change one of its elements...
same_matrix[0, 1] = 2.
# It changes both variables: matrix and same_matrix
print(matrix)
print ("-"*L)
print(same_matrix)

# %% [markdown]
# If you want to copy all data for real, that is, duplicate the data in memory, you may use either its ```new_tensor()``` or ```clone()``` methods.
#
# ----------
#
# The first ```matrix``` below acts as a template. It controls:
#
# - Data type (e.g., float32, int64)
# - Device (CPU or CUDA GPU)
# - Layout / other tensor settings

# %%
# We can use "new_tensor" method to REALLY copy it into a new one
different_matrix = matrix.new_tensor(matrix.view(1, 6))

# Now, if we change one of its elements...
different_matrix[0, 1] = 3.

# The original tensor (matrix) is left untouched!
# But we get a "warning" from PyTorch telling us 
# to use "clone()" instead!
print(matrix)
print ("-"*L)
print(different_matrix)

# %% [markdown]
# It seems that PyTorch prefers that we use ```clone()``` together with ```detach()``` instead of ```new_tensor()```. Both ways accomplish the same result, but the code below is deemed cleaner and more readable:

# %%
# Lets follow PyTorch's suggestion and use "clone" method
another_matrix = matrix.view(1, 6).clone().detach()

# Again, if we change one of its elements...
another_matrix[0, 1] = 4.

# The original tensor (matrix) is left untouched!
print(matrix)
print ("-"*L)
print(another_matrix)

# %% [markdown]
# ### Conversions between Numpy and PyTorch
#
# The ```as_tensor``` method preserves the type of the array, which can also be seen in the code below:

# %%
x_train = np.random.randn(10)

x_train_tensor = torch.as_tensor(x_train)
print(x_train.dtype, x_train_tensor.dtype)

# %%
x_train_tensor = torch.as_tensor(x_train)
float_tensor = x_train_tensor.float() # (going from float64 to float32) 
print(float_tensor.dtype)

# %% [markdown]
# **IMPORTANT**: Both ```as_tensor()``` and ```from_numpy()``` return a tensor that shares the underlying data with the original Numpy array. Similar to what happened with ```view()```, if you modify the original Numpy array, you are modifying the corresponding PyTorch tensor too and vice-versa.

# %%
dummy_array = np.array([1, 2, 3])
dummy_tensor = torch.as_tensor(dummy_array)

dummy_array[1] = 0 # Modifies the numpy array
print(dummy_tensor) # Tensor gets modified too...

# %% [markdown]
# What do we need ```as_tensor()``` for? Why can’t we just use ```torch.tensor()```?
#
# Well, you could. Just keep in mind that ```torch.tensor()``` always makes a copy of the data instead of sharing the underlying data with the Numpy array.

# %%
## Opposite Direction

dummy_array = np.array([1, 2, 3])
dummy_tensor = torch.as_tensor(dummy_array)
# Modifies the numpy array
dummy_array[1] = 0
# Using numpy function to convert PyTorch tensor to Numpy array
print(dummy_tensor.numpy())

# %% [markdown]
# ## GPU

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_cudas = torch.cuda.device_count()
for i in range(n_cudas):
    print(torch.cuda.get_device_name(i))

# %%
# if you had GPU, in the print output you would see device='cudaXYZ'
gpu_tensor = torch.as_tensor(x_train).to(device)
print(gpu_tensor[0])

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them 
# into PyTorch's Tensors and then we send them to the  chosen device
x_train_tensor = torch.as_tensor(x_train).float().to(device)

# %%
print(type(x_train), "****", type(x_train_tensor), "****", x_train_tensor.type())

# %%
# GPU tensor back to NumPy:
back_to_numpy = x_train_tensor.numpy() # For CPU
back_to_numpy = x_train_tensor.cpu().numpy() # For GPU

print(back_to_numpy.shape)

# %% [markdown]
# The following would fail; the ```requires_grad``` will be lost if we do it this way.
# Next cell is the correct way of doing it.
#
# ```python
# b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
# w = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
# ```

# %%

# %% [markdown]
# ### ```backward```, ```grad```, and ```zero_``` Methods
#
# Every time we use the gradients to update the parameters, we need to zero the gradients afterward. And that is what ```zero_()``` is good for. i.e. by default gradients accumulate! we need to set them to zero!

# %% [markdown]
# ```python
# yhat = b + w * x_train_tensor
# error = (yhat - y_train_tensor)
# loss = (error ** 2).mean()
#
# # Step 3 - Computes gradients for both "b" and "w" parameters
# # No more manual computation of gradients! 
# # b_grad = 2 * error.mean()
# # w_grad = 2 * (x_tensor * error).mean()
# loss.backward()
#
# print(b.grad, w.grad)
# ```

# %% [markdown]
# # Optimizer

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
lr = 0.1 

optimizer = optim.SGD([b, w], lr=lr)               # 1)
n_epochs = 1000

for epoch in range(n_epochs):
    yhat = b + w * x_train_tensor
    error = (yhat - y_train_tensor)
    loss = (error ** 2).mean()
    loss.backward()
    
    # Step 4 - updates parameters using gradients and 
    # the learning rate. No more manual update!
    # with torch.no_grad():
    #     b -= lr * b.grad
    #     w -= lr * w.grad
    optimizer.step()                               # 2)
    
    # No more telling Pytorch to let gradients go!
    # b.grad.zero_()
    # w.grad.zero_()
    optimizer.zero_grad()                          # 3)
    
print(b, w)

# %%
###
### Cleaned up
###
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
lr = 0.1 

optimizer = optim.SGD([b, w], lr=lr)    # 1)
n_epochs = 1000

for epoch in range(n_epochs):
    yhat = b + w * x_train_tensor
    error = (yhat - y_train_tensor)
    loss = (error ** 2).mean()
    loss.backward()
    
    optimizer.step()                    # 2)
    optimizer.zero_grad()               # 3)
    
print(b, w)

# %% [markdown]
# # Loss
#
# ```python
# import torch.nn as nn
#
# loss_fn = nn.MSELoss(reduction='mean')
# predictions = torch.tensor([0.5, 1.0])
# labels = torch.tensor([2.0, 1.3])
# print(loss_fn(predictions, labels))
# ```

# %%
lr = 0.1
torch.manual_seed(42)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

optimizer = optim.SGD([b, w], lr=lr)
loss_fn = nn.MSELoss(reduction='mean')             # 1) MSE loss function
n_epochs = 1000

for epoch in range(n_epochs):
    # Step 1 - computes model's predicted output - forward pass
    yhat = b + w * x_train_tensor    
    loss = loss_fn(yhat, y_train_tensor)           # 2)

    # Step 3 - gradients for "b" and "w"
    loss.backward()
    
    # Step 4 - updates parameters
    optimizer.step()
    optimizer.zero_grad()
    
print(b, w)

# %% [markdown]
# Since ```loss``` is computing gradients we need to use ```detach``` to convert it to numpy or ```tolist```:
#
# ```python
# print(loss.detach().cpu().numpy())
# print(loss.tolist())
# loss.item() # only if there is one scaler.
# ```

# %%
