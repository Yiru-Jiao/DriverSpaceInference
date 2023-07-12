# Folder structure

Run `Preprocessing.py` first to preprocess the rawdata

Use `IntersectionDetection.py` and `IntersectionData.ipynb` to identify and select intersections in the pNEUMA dataset

Run `Sampling_exp1-2.py`, `Sampling_exp3.ipynb`, and `Sampling_exp4.py` to sample vehicle pairs for different experiments

`DriverSpaceInference.py` is the library including classes and functions for the experiments

Run `Experiments.py` to repeat the experiments in our article

# Operation system

We performed the experiments on the Linux system and the inference used 15 CPUs for parallelizing.

# Requirements
Library requirements: `numpy`, `pandas`, `scipy`, `tqdm`, `joblib`, `pyproj (3.2.0)`, `shapely`, `matplotlib`, `scikit-learn`

Note: Input data is saved as `.h5` file in our experiments. This requires the python package named `tables` or `pytables`.

