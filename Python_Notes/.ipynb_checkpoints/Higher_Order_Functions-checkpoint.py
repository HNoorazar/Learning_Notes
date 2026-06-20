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
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# %%
def square(x):
    return x ** 2

def cube(x):
    return x ** 3
 
def fourth_power(x):
    return x ** 4


# %%
def generic_exponentiation(x, exponent):
    return x ** exponent

print(generic_exponentiation(2, 5))


# %% [markdown]
# Should build (and return) functions

# %%
def exponentiation_builder(exponent):
    def skeleton_exponentiation(x):
        return x ** exponent
    
    return skeleton_exponentiation

square_func = exponentiation_builder(2)
cube_func = exponentiation_builder(3)
print(square_func)
print(square_func(5))
print(cube_func(5))


# %%
def calculate_result(num):
    def equation(x):

        def equation2(y):
            return (x + num) / (y - num)

        return equation2

    return equation

res = calculate_result(2)
res2 = res(4)
print(res2(5))


# %%
def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def perform_train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        
        # Step 1 - computes model's predictions - forward pass
        yhat = model(x)
        # Step 2 - computes the loss
        loss = loss_fn(yhat, y)
        # Step 3 - computes gradients for "b" and "w" parameters
        loss.backward()
        # Step 4 - updates parameters using gradients and the learning rate
        optimizer.step()
        optimizer.zero_grad()
        
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the 
    # train loop
    return perform_train_step


def make_val_step(model, loss_fn):
    # Builds function that performs a step in the validation loop
    def perform_val_step(x, y):
        # Sets model to EVAL mode
        model.eval()     # 1)
        
        # Step 1 - Computes our model's predicted output
        yhat = model(x) # forward pass
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # There is no need to compute Steps 3 and 4, 
        # since we don't update parameters during evaluation
        return loss.item()
    
    return perform_val_step


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.1
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Defines a SGD optimizer to update the parameters 
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean') # Defines a MSE loss function

# Creates the train_step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer) # 1)

n_epochs = 1000
losses = []

for epoch in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)

# %%
