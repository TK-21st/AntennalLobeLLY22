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
# # Contrast and Concentration Invariance
# This notebook illustrates the following:
# 1. Decomposition of OSN and PN responses into steady-state and transient components,
# 2. Computing Odorant Concentration Contrast
# 3. Measuring concentration invariance and contrast boosting by comparing steady-state and transient
#    OSN and PN responses to concentration waveform and concentration contrast.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import seaborn as sns
from tqdm.auto import tqdm
from utils import get_contrast, decompose_signal, yyaxis, corrcoef

# %%
data = np.load('../data/staircase.npz', allow_pickle=True)
t = data['t']
dt = t[1] - t[0]

# %% [markdown]
# ## Plot raw data

# %%
fig,axes = plt.subplots(
    4, 7, figsize=(20,8), 
    sharex=True, sharey='row', 
    gridspec_kw=dict(hspace=.5, wspace=.3)
)

for n, (osn_i, osn_o, pn_i, pn_o) in enumerate(zip(
    data['osn_input'],
    data['osn_output'],
    data['pn_input'],
    data['pn_output'],
)):
    axes[0,n].plot(t, osn_i, c='k')
    axes[0,n].set(title='Acetone Concentration')
    axes[1,n].plot(t, osn_o, c='k')
    axes[1,n].set(title='Or59b OSN Response')
    axes[2,n].plot(t, pn_i, c='k')
    axes[2,n].set(title='Acetone Concentration')
    axes[3,n].plot(t, pn_o, c='k')
    axes[3,n].set(title='DM4 PN Response')
_ = [ax.set(xlabel='Time [sec]') for ax in axes[-1]]
axes[0,0].set(ylabel='Concentration [ppm]')
axes[1,0].set(ylabel='PSTH [Hz]')
axes[2,0].set(ylabel='Concentration [ppm]')
axes[3,0].set(ylabel='PSTH [Hz]')

# %% [markdown]
# ## Decompose PN responses to get transient and steady-state components

# %%
fig,axes = plt.subplots(
    4, 7, figsize=(20,8), 
    sharex=True, sharey='row', 
    gridspec_kw=dict(hspace=.5, wspace=.5)
)

for n, (pn_i, pn_o) in enumerate(zip(
    data['pn_input'],
    data['pn_output'],
)):
    peak_time, _, steady_state, peak = decompose_signal(t, pn_i[None,:], pn_o[None,:])
    u_contrast = get_contrast(t, pn_i[None,:])
    axes[0,n].plot(t, pn_i.T, c='k')
    axes[0,n].set(title='Acetone Concentration')
    axes[1,n].plot(t, pn_o.T, c='k')
    axes[1,n].set(title='DM4 PN Response')
    axes[2,n].plot(t, steady_state.T, c='k')
    axes[2,n].set(title='Steady-State Response')
    axes[3,n].plot(t, peak.T, c='k')
    axes[3,n].set(ylim=[-100, 200])
    ax2 = yyaxis(axes[3,n], c='r')
    ax2.plot(t, u_contrast.T, c='r')
    if n == 0:
        ax2.set(ylabel='Contrast [/s]')
    ax2.set(ylim=[-10, 20])
    axes[3,n].set(title='Transient Response \nvs. Contrast')
_ = [ax.set(xlabel='Time [sec]') for ax in axes[-1]]
axes[0,0].set(ylabel='Concentration [ppm]')
axes[1,0].set(ylabel='PSTH [Hz]')
axes[2,0].set(ylabel='PSTH [Hz]')
axes[3,0].set(ylabel='PSTH [Hz]')

# %% [markdown]
# ## Correlation Steady-State vs. Concentration Waveform, Transient vs. Contrast

# %%
delays = np.arange(int(-.1//dt), int(.1//dt), 10)
corr = dict(xy = [], corr = [], neuron=[])

for n, (inp, out) in tqdm(
    enumerate(zip(data['osn_input'], data['osn_output'])), 
    total=len(data['pn_input'])
):
    _, _, ss, pk = decompose_signal(t, inp[None,:], out[None,:])
    ss = np.squeeze(ss)
    pk = np.squeeze(pk)
    inp_contrast = np.squeeze(get_contrast(t, inp[None,:]))
    output_conc_corr = np.max(corrcoef(
        np.vstack([np.roll(out, d) for d in delays]),
        inp[None,:]
    ))
    ss_conc_corr = np.max(corrcoef(
        np.vstack([np.roll(ss, d) for d in delays]),
        inp[None,:]
    ))
    pk_conc_corr = np.max(corrcoef(
        np.vstack([np.roll(pk, d) for d in delays]),
        inp_contrast[None,:]
    ))
    
    corr['xy'].append('response vs. concentration')
    corr['corr'].append(output_conc_corr)
    corr['neuron'].append('osn')
    
    corr['xy'].append('steady-state vs. concentration')
    corr['corr'].append(ss_conc_corr)
    corr['neuron'].append('osn')
    
    corr['xy'].append('transient vs. contrast')
    corr['corr'].append(pk_conc_corr)
    corr['neuron'].append('osn')

for n, (inp, out) in tqdm(
    enumerate(zip(data['pn_input'], data['pn_output'])), 
    total=len(data['pn_input'])
):
    _, _, ss, pk = decompose_signal(t, inp[None,:], out[None,:])
    ss = np.squeeze(ss)
    pk = np.squeeze(pk)
    inp_contrast = np.squeeze(get_contrast(t, inp[None,:]))
    output_conc_corr = np.max(corrcoef(
        np.vstack([np.roll(out, d) for d in delays]),
        inp[None,:]
    ))
    ss_conc_corr = np.max(corrcoef(
        np.vstack([np.roll(ss, d) for d in delays]),
        inp[None,:]
    ))
    pk_conc_corr = np.max(corrcoef(
        np.vstack([np.roll(pk, d) for d in delays]),
        inp_contrast[None,:]
    ))
    
    corr['xy'].append('response vs. concentration')
    corr['corr'].append(output_conc_corr)
    corr['neuron'].append('pn')
    
    corr['xy'].append('steady-state vs. concentration')
    corr['corr'].append(ss_conc_corr)
    corr['neuron'].append('pn')
    
    corr['xy'].append('transient vs. contrast')
    corr['corr'].append(pk_conc_corr)
    corr['neuron'].append('pn')

# %%
fig,ax = plt.subplots(1,1,figsize=(8,8))

vp = sns.boxplot(data=corr, x="corr", y="xy", hue="neuron", width=.5)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
sns.swarmplot(data=corr, x="corr", y="xy", hue="neuron", dodge=True, size=8, color='.25')
_ = ax.set(
    xlim=[0, 1], 
    xlabel='Maximum Correlation', 
    ylabel='Comparison', 
    title='Comparison of Concentration Invariance and Contrast-Boosting Properties of OSN and PN physiology responses'
)
