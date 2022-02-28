# AntennalLobeLLY22
Code for "The Functional Logic of Odor Information Processing in the Drosophila Antennal Lobe" by Lazar, Liu, Yeh, 2022.

Please find relevant JuPyteR notebooks in the `/notebooks` folder. The python scripts `.py` are synchronized with the notebooks with the same name `.ipynb` using [`jupytext`](https://github.com/mwouts/jupytext) for easy source control.


### Software Requirements
The notebooks are developed using `Python 3.8.8` and earlier versions are not guaranteed to work.

The following packages are required:
```
neural    # in-house package, see below
jupytext  # synchronization between jupyter notebooks and python scripts
seaborn
```

Note that `neural` is an in-house package that supports massively parallel evaluation of 
biological neural networks. It is written purely in Python and supports a natural 
API. This notebook, was developed using a specific pinned version of neural,
and can be installed from GitHub [here](https://github.com/chungheng/neural/tree/al).


## Citation
```
@article{LLY22,
  author = "Aurel A. Lazar and Tingkai Liu and C.-H. Yeh",
  title = "The Functional Logic of Odor Information Processing in the Drosophila Antennal Lobe",
  year = 2022,
  journal = "bioRxiv  ",
  month = "Feb"
}
```
