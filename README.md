# Inferring vehicle spacing in urban traffic from trajectory data
This study is published in the journal "Transportation Research Part C: Emerging Technologies" with gold open access, available at <https://doi.org/10.1016/j.trc.2023.104289>.
## Highlights
- A method to infer average 2D vehicle spacing from trajectory data is proposed.
- A perspective on the relative movement between interacting vehicles is taken.
- Empirical relations between average 2D spacing and relative speeds are identified.
- The empirical relations are termed as interaction Fundamental Diagrams (iFDs).
- iFDs describe the variation in required space for vehicle interactions.

## In order to repeat the experiments:

### Dependencies
`tqdm`, `numpy`, `pandas`, `scipy`, `pyproj=3.2.0`, `joblib`, `matplotlib`, `shapely`, `scikit-learn`

### Data 
Download raw data from <https://open-traffic.epfl.ch/index.php/downloads/> and save them in the folder "RawDatasets".

### Step-by-step instructions

__Step 1.__ Run `./Code/Preprocessing.py` to preprocess the rawdata.

__Step 2.__ Use `./Code/IntersectionDetection.py` and `./Code/IntersectionData.ipynb` to identify and select intersections in the pNEUMA dataset.

__Step 3.__ Run `./Code/Sampling_exp1-2.py`, `./Code/Sampling_exp3.ipynb`, and `./Code/Sampling_exp4.py` to transform coordinates and sample vehicle pairs for different experiments.

__Step 4.__ Run `./Code/Experiments.py` to repeat our experiments in the article.

__*__ `./Code/DriverSpaceInference.py` is the library including classes and functions for the experiments

__*__ We run the experiments in Linux with a cluster of CPUs. To be run on other OSs may need adjustments regarding the number of cores for parallel processing.

## In order to apply the method to another dataset:

__Step 1.__ Save raw data in the folder `./RawDatasets/`.

__Step 2.__ Create code to align the format of the new dataset to the format of the data to be saved in the folder `./InputData/`.

__Step 3.__ Design your application according to the code in `./Code/Experiments.py`.

## Citation
````latex
@article{Jiao2023,
  doi = {10.1016/j.trc.2023.104289},
  year = {2023},
  volume = {155},
  pages = {104289},
  author = {Yiru Jiao and Simeon C. Calvert and Sander {van Cranenburgh} and Hans {van Lint}},
  title = {Inferring vehicle spacing in urban traffic from trajectory data},
  journal = {Transportation Research Part C: Emerging Technologies}
}
````
