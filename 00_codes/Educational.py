# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %% [markdown]
# # \*args and \**kwargs
#
# https://book.pythontips.com/en/latest/args_and_kwargs.html
#
#
# 1. `*args` and `**kwargs`
#  I have come to see that most new python programmers have a hard time figuring out the `*args` and `**kwargs` magic variables. So what are they ? First of all, let me tell you that it is not necessary to write `*args` or `**kwargs`. Only the `*` (asterisk) is necessary. You could have also written `*var` and `**vars`. Writing `*args` and `**kwargs` is just a convention. So now let’s take a look at `*args` first.
#
# 1.1. Usage of `*args`
#
# `*args` and `**kwargs` are mostly used in function definitions. `*args` and `**kwargs` allow you to pass an unspecified number of arguments to a function, so when writing the function definition, you do not need to know how many arguments will be passed to your function. `*args` is used to send a **non-keyworded** variable length argument list to the function. Here’s an example to help you get a clear idea:

# %%
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')


# %%
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

greet_me(name="yasoob", lastName="Noor")

# %%

# %%
