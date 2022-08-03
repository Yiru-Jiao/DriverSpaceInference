# Folder structure

`DriverSpaceApproximation.py` is the library including classes and functions for the experiments

Run `PreprocessingAndSampling.py` to preprocess and sample vehicles

Run `Exp_pNEUMA.py` to repeat the experiments in our article

Use `ResultEvaluation.ipynb` to accept and reject inference results

# Operation system

We performed the experiments on the Linux system and the inference used 20 CPUs for parallelizing.

# Requirements
Library requirements: `numpy`, `pandas`, `scipy`, `pyproj (3.2.0)`

Note: Input data is saved as `.h5` file in our experiments. This requires the python package named `tables`.

