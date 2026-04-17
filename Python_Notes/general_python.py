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

# %% [markdown]
# ```defaultdict``` is a subclass of the built-in ```dict``` class from the ```collections``` module. It automatically assigns a default value to keys that do not exist, which means you don't have to manually check for missing keys and avoid KeyError.
#
# This example shows how a defaultdict automatically creates missing keys with a default empty list.
#
# #### From **[Geeks for Geeks](https://www.geeksforgeeks.org/python/defaultdict-in-python/)**

# %%
from collections import defaultdict
words = ["eat", "tea", "tan", "ate", "nat", "bat"]

# %%
groups = defaultdict(list)
groups['fruits'].append('apple')
groups['vegetables'].append('carrot')
print(groups)

# %%
groups.keys()

# %%
print(groups['fruits'])
print(groups['juices'])

# %%
groups.keys()

# %% [markdown]
# **Explanation:** This code creates a defaultdict with a default value of an empty list. It adds elements to the 'fruits' and 'vegetables' keys. When trying to access the 'juices' key, no KeyError is raised, and an empty list is returned since it doesn't exist in the dictionary.
#
# ```defaultdict(default_factory)```
#
# - ```default_factory```: A callable (like int, list, set, str or a custom function) that provides the default value for missing keys.
# - If this argument is None, accessing a missing key raises a KeyError.
#
#
# **Why do we need defaultdict()**
#
# In a normal dictionary, accessing a missing key raises a KeyError. defaultdict solves this by:
#
# - Automatically creating missing keys with a default value.
# - Reducing repetitive if key not in dict checks.
# - Making tasks like counting, grouping, or collecting items easier.
# - Being especially useful for histograms, graph building, text grouping, and caching.

# %%

# %%
from collections import Counter

arr = [1, 2, 3, 1, 5, 5, 5]
freq = Counter(arr)

# %%
freq

# %%
