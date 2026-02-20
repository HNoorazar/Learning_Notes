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


# %% [markdown]
# In the ```__init__``` method, we define our two parameters of ```b``` and ```w```, using the ```Parameter()``` class, to tell PyTorch that these tensors, which are attributes of the ```ManualLinearRegression``` class, should be considered parameters of the model the class represents.
#
# Why should we care about that? By doing so, we can use our model’s ```parameters()``` method to retrieve an iterator over all model’s parameters, including parameters of nested models. Then we can use it to feed our optimizer (instead of building a list of parameters ourselves).

# %% [markdown]
# ```nn.Module``` is the base class for all neural networks.
# By inheriting from ```nn.Module```, your class automatically gets:
# - Parameter tracking
# - ```.parameters()``` method
# - ```.to(device)``` support
# - ```.train()``` / ```.eval()``` modes
# - Model saving/loading (```state_dict```)
# - Integration with optimizers
# - Automatic registration of layers
#
# ----------
#
# ```super().__init__()``` calls the parent class’s constructor (here ```nn.Module```) to properly initialize all the internal PyTorch machinery so your model works correctly.
#
#
# PyTorch sets up:
# - Internal dictionaries to store parameters
# - Hooks
# - Buffers
# - Submodules
# - Gradient tracking infrastructure

# %%
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "b" and "w" real parameters of the model, 
        # we need to wrap them with nn.Parameter
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x):
        # Computes the outputs / predictions
        return self.b + self.w * x
    
torch.manual_seed(42)
# Creates a "dummy" instance of our ManualLinearRegression model
dummy = ManualLinearRegression()
print(list(dummy.parameters()))

# %% [markdown]
# **The ```state_dict``` method**
#
# Moreover, we can get the current values of all parameters using our model’s ```state_dict()``` method.

# %%
ummy = ManualLinearRegression()
print(dummy.state_dict())

# %% [markdown]
# In the following cell, ```yhat = model(x_train_tensor)``` returns predictions because how Python is built and works.
#
# The ```model()``` is instance of our class, that class inherited stuff from ```nn.Module``` and when we do ```yhat = model(x_train_tensor)```, Python looks for ```forward()``` in the class. If there is no ```forward()``` it will raise an error.
#
# **Remember:** You should make predictions that call model(x).
#
# Do **not** call ```model.forward(x)```!
#
# Otherwise, your model’s hooks will not work (if you have them).

# %%
lr = 0.1
torch.manual_seed(42)

model = ManualLinearRegression().to(device)        # 1)

# optimizer = optim.SGD([b, w], lr=lr) <- we did this before.
# Now, since we wrapped them in Prameters, we just do:
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')

n_epochs = 1000
for epoch in range(n_epochs):
    # set the model to training mode
    model.train()                                # 2)

    # Step 1 - computes model's predicted output - forward pass
    # No more manual prediction!
    yhat = model(x_train_tensor)                 # 3)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    
    # Step 4 - updates parameters
    optimizer.step()
    optimizer.zero_grad()
    
# We can also inspect its parameters using its state_dict
print(model.state_dict())

# %% [markdown]
# It turns out, state dictionaries can also be used for checkpointing a model as we will see in the Rethinking the Training Loop chapter.
#
#
# # Nested v. Sequential Models
#
# PyTorch's built-in lienar regression model: 1 input, 1 output:

# %%
linear = nn.Linear(1, 1)
print(linear)
print(linear.state_dict())


# %% [markdown]
# We can replace our manually created bias and weight in the class we created above by the built-in function of PyTorch.
#
# The ```linear``` model in our class is what they call nested model!
#
# -----------
# **Old Class**
#
# ```python
# class ManualLinearRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
#         self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
#         
#     def forward(self, x):
#         return self.b + self.w * x
# ```
#
# -----------
# In the ```__init__``` method, we create an attribute that contains our nested ```Linear``` model.
#
# In the ```forward()``` method, we call the nested model itself to perform the forward pass (notice that we are not calling ```self.linear.forward(x))```.

# %%
class MyLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear model 
        # with a single input and a single output
        self.linear = nn.Linear(1, 1)
                
    def forward(self, x):
        # Now it only takes a call
        self.linear(x)

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dummy = MyLinearRegression().to(device)
print(list(dummy.parameters()))
print ("-"*90)
print(dummy.state_dict())

# %% [markdown]
# #### Sequential models
#
# For straightforward models that use a series of built-in PyTorch models (like ```Linear```) where the output of one is sequentially fed as an input to the next, we can use a ```Sequential``` model.
#
# In our case, we would build a ```Sequential``` model with a single argument; that is, the ```Linear``` model we used to train our linear regression. The model would look like this:

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
# Alternatively, you can use a Sequential model
model = nn.Sequential(nn.Linear(1, 1)).to(device)
print(model.state_dict())

# %% [markdown]
# ## Hold On. Zoom out:
#
# A nested model is ``nested'' because:
# - A module contains other modules
# - Those modules contain other modules
# - It forms a tree structure
#
# Sequential models are also technically nested! The difference is **control over the forward pass**.
#
# Nested:
#
# ```python
# def forward(self, x):
#     x = self.block1(x)
#     x = self.block2(x)
#     return self.output(x)
# ```

# %% [markdown]
#
# You can:
# - Add skip connections
# - Add branching
# - Use multiple inputs
# - Store intermediate outputs
# - Use conditionals
# - Reuse blocks
# - Loop dynamically
#
# ```python
# def forward(self, x):
#     x1 = self.block1(x)
#     x2 = self.block2(x1)
#     return self.output(x1 + x2)
# ```
#
# ```Sequential``` is basically output of one layer → input of next layer. No branching, no nothing:
#
# ```python
# def forward(self, x):
#     for module in self.modules:
#         x = module(x)
#     return x
# ```

# %%
### Sequential model with naming layers:

torch.manual_seed(42)
model = nn.Sequential()
model.add_module('layer1', nn.Linear(3, 5))
model.add_module('layer2', nn.Linear(5, 1))
print(model.to(device))

# %% [markdown]
# Magic commands of Jupyter:
#
# ```python
# %%writefile data_preparation/v0.py
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# x_train_tensor = torch.as_tensor(x_train).float().to(device)
# y_train_tensor = torch.as_tensor(y_train).float().to(device)
#
#
# %run -i data_preparation/v0.py
# ```
#
# ``` -i``` option to make all variables available from both the notebook and the file 

# %%
###
### Model Config:
###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1, 1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')
###
### Train:
###
n_epochs = 1000

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward() # Step 3 - computes gradients
    # Step 4 - updates parameters
    optimizer.step()
    optimizer.zero_grad()

# %%

# %%
