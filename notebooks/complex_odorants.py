# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Complex Odorant Concentration Waveform

# %%
import h5py 
import matplotlib.pyplot as plt
import numpy as np

# %%
f = h5py.File('../../al_rfc/data/data.h5')
fig,axes = plt.subplots(2,1, sharex=True)
axes[0].plot(f['white_noise/stimulus/x'][()], f['white_noise/stimulus/y'][()])
axes[1].plot(f['white_noise/psth/x'][()], f['white_noise/psth/y'][()])
f.close()

# %%
