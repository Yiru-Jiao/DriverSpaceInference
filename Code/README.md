# Folder structure

`DriverSpaceInference.py` is the library including classes and functions for the experiments

Run `Preprocessing.py` and `Sampling` to preprocess and sample vehicles, respectively

Run `Exp_pNEUMA.py` to repeat the experiments in our article

# Operation system

We performed the experiments on the Linux system and the inference used 10 CPUs for parallelizing.

# Requirements
Library requirements: `numpy`, `pandas`, `scipy`, `pyproj (3.2.0)`

Note: Input data is saved as `.h5` file in our experiments. This requires the python package named `tables`.

