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
#import mmf_setup;mmf_setup.nbinit()
# %load_ext autoreload

# %%
import shutup
shutup.please()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import sys
sys.path.append("/Users/HN/Documents/00_GitHub/Learning_Notes/00_codes/Figures/")
sys.path.append("/Users/HN/Documents/00_GitHub/mmfutils-fork/")

# %%

# %%

# %%
# # !pip3 install mmfutils

# %%
# %pylab inline --no-import-all
# %autoreload

import figure_style
from figure_style import Paper

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


class PaperFigures(Paper):
    style = "nature"
    #style = 'arXiv'
    #style = "none"

    SAMPLE_RATE = 44100
    DURATION = 10

    def __init__(self, **kw):
        super().__init__(**kw)
        _, nice_tone = generate_sine_wave(freq = 2, sample_rate=self.SAMPLE_RATE, duration = self.DURATION)
        _, noise_tone_100 = generate_sine_wave(freq = 20, sample_rate = self.SAMPLE_RATE, duration = self.DURATION)

        noise_tone_100 = noise_tone_100 * 0.4
        self.noisy_tone = nice_tone  + noise_tone_100
        self.nice_tone = nice_tone
        print (len(nice_tone))

    def fig_demo(self):
        myfig = self.figure(
            num=0,  # If you want to redraw in the same window
            width="textwidth",  # For two-column documents, vs. 'textwidth'
            height=0.3,   # Fraction of width
            constrained_layout=True,
            #margin_factors=dict(top=0, left=0, bot=0, right=0),
        )
        ax = plt.gca()
        # ax.plot(nice_tone[:10000], '-r', label = 'nice tone')
        ax.plot(self.noisy_tone, '-r', label='noisy signal')
        ax.plot(self.nice_tone, '-', label='true signal', c='k')

        ax.set_xlabel('t')
        ax.set_ylabel('magnitude')
        ax.legend(loc="best")
        ax.grid(True);
        return myfig


# %%
f = PaperFigures()
#f.fig_demo()
f.draw('fig_demo')

# %%
plt.rcParams['text.usetex'] = False
x = np.linspace(0, 10, 1000)
dx = np.diff(x).mean()
y = np.sin(10*x) + np.cos(28*x)
y += np.random.normal(size=len(x))
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(x, y)
plt.subplot(122)
plt.psd(y, Fs=1/dx*2*np.pi);  # Fs is WRONG.


# %%
# %timeit np.fft.rfft(y)
# %timeit np.fft.fft(y)

# %%
# np.fft.rfft?

# %%
k = 2*np.pi * np.fft.fftfreq(len(x), d=dx)[:len(x)//2]
#plt.plot(k, abs(np.fft.rfft(y)), '-+')
#plt.plot(np.fft.fftshift(k), np.fft.fftshift(abs(np.fft.rfft(y))), '-+')

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# %%
import shutil
shutil.rmtree(matplotlib.get_cachedir())

# %%
# import matplotlib
# import matplotlib.font_manager
# matplotlib.font_manager._rebuild()

# %%
warnings.filterwarnings("ignore", category=UserWarning, module="xkcd")

# %%
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    # ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    ax.annotate('local max', xy=(3, 1),  xycoords='data')

# %%
with plt.xkcd():
    x = np.linspace(-2,2,100);
    y = np.cos(x);
    plt.plot(x, y);

# %% [markdown]
# ## ROC Curve

# %%
with plt.xkcd():
    x = np.linspace(0,1,100)
    plt.plot(x, x, "--", c="red")
    plt.text(0.4, 0.31, 'random classifier', rotation=35)
    
    plt.plot(x, np.sqrt(x), 'k') # 
    plt.plot(x, x**(1/3), 'y')
    
    """
    xy     is the end of the arrow
    xytext is the start of text
    """
    plt.annotate('better', xy=(0.8, 1), xytext=(0.9, 0.822),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.scatter(0, 1)    
    plt.arrow(x=0.2, y=1, dx=-0.1, dy=0, head_width=0.05) # , width=0.01
    plt.text(0.2, 0.98, 'best classifier')
    
    plt.xlabel('False Positive rate') # , labelpad = 15);
    plt.ylabel('True Positive rate') # , labelpad = 15);
    plt.title('ROC curve')
    
    plot_dir = "/Users/hn/Documents/00_GitHub/Learning_Notes/00_figures/"
    file_name = plot_dir + "ROC_curve1.pdf"
    plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %% [markdown]
# ## Gini Index

# %%
import math

# # %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
# from pathlib import Path
# import matplotlib as mpl
# mpl.rcParams['figure.dpi']= 400
# mpl.rcdefaults()

# # plt.style.use('Figures/tufte.mplstyle')
# plt.subplots_adjust(right=1.5, top=1.1)
# pt_per_inch = 72.27
# marginwidth_inch = 144.0/pt_per_inch
# textwidth = 312.0/pt_per_inch
# textheight = 616.0/pt_per_inch


with plt.xkcd():
    x = np.linspace(0,1,100)
    #
    # Misclassification error
    #
    plt.plot(x, 1 - np.maximum(x, 1-x), "--", c="red")
    plt.text(0.07, 0.045, 'Misclassification error', rotation=52, c="r")
    #
    # Gini index
    #
    plt.plot(x, 2*x*(1-x), "-", c="y")
    plt.text(0.12, 0.18, 'Gini index', rotation=60, c="y")    
    #
    # cross-entropy
    #
    z = x[1:-1]
    w = 1-z
    logz = [math.log(num, 4) for num in z]
    logw = [math.log(num, 4) for num in w]
    
    entropy = -z*logz - (w*logw)
    plt.plot(z, entropy, "-", c="b")
    plt.text(0.07, 0.28, 'Cross-entropy', rotation=55, c="b")



#    plt.xlabel('False Positive rate') # , labelpad = 15);
#    plt.ylabel('True Positive rate') # , labelpad = 15);
    plt.title('impurity measures')
#    plt.ylim([0, 0.55])
    
    plot_dir = "/Users/hn/Documents/00_GitHub/Learning_Notes/00_figures/"
    file_name = plot_dir + "gini_crossEntropy.pdf"
    plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %%
