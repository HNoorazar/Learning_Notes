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

# %% [markdown]
# - Constructor
# - Optional and Essential Arguments
# - Public, protected, and private methods
#
# ----------------
# **Constructor** ```__init__(self)```
#
# The constructor defines the parts that make up the class. These parts are the attributes of the class. Typical, attributes include:
#
# - Arguments provided by the user.
# - Placeholders for other objects that are not available at the moment of creation (pretty much like delayed arguments).
# - Variables we may want to keep track of.
# - Functions that are dynamically built using some of the arguments and higher-order functions.

# %% [markdown]
#

# %%

# %%
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."

class Dog(Animal):  # Dog is the child class, Animal is the parent class
    def __init__(self, name, breed):
        super().__init__(name)  # Call the parent class's __init__ method
        self.breed = breed

    def speak(self):
        return f"{self.name} barks!"

# Create an instance of the child class
my_dog = Dog("Buddy", "Golden Retriever")

# Access attributes and methods from both parent and child classes
print(my_dog.name)
print(my_dog.breed)
print(my_dog.speak())


# %%

# %% [markdown]
# ### Placeholders
#
# Find some general and simple examples and put below

# %%

# %% [markdown]
# The following is borrowed from educative.io course *Deep Learning with PyTorch Step-by-Step: Part I - Fundamentals*.
#
# First, we can define whatever inside constructor such as ```self.device = 'cuda' if torch.cuda.is_available() else 'cpu'```
#
# Secondly, the ```self.model.to(self.device)``` inside constructor is using PyTorch model’s ```to()```. It is not using the method ```to()``` defined in that class. The method ```to()``` is there for the user to change the devide if he wishes to.
#
#
# **Placeholders or delayed arguments**:
#
# We expect the user to eventually provide some of these, as they are not necessarily required. In our class, there are another three elements that fall into this category: train and validation data loaders and a summary writer to interface with TensorBoard.
#
# The constructor with the appended code will look like the following:

# %%
class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # attributes of our class
        
        # We start by storing the arguments as attributes 
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # send the model to the specified device right away
        self.model.to(self.device)
        
        # These attributes are defined here, but since they are
        # not available at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        def to(self, device):
            # This method allows the user to specify a different device
            # It sets the corresponding attribute (to be used later in
            # the mini-batches) and sends the model to the device
            self.device = device
            self.model.to(self.device)

# %% [markdown]
# ### Methods
#
# - Public: can be called by user
# - Protected: internal use or use by its child: single underscore
# - Private: strictly internal use: double underscore

# %%
