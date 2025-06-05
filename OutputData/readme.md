# Resulting dataset README

This dataset deposited at <https://doi.org/10.4121/8cadc255-5fd8-46ab-893a-64b76ca7b7f9> contains the processed outputs generated from raw vehicle trajectory data (pNEUMA) after completing preprocessing, intersection detection, and sampling. It is used to perform driver space inference experiments as described in the associated publication.

## Data Organisation

`OutputData.zip` has a structure as follows:
- `OutputData/DriverSpace/`
  - `OutputData/DriverSpace/Inference/` contains the results of the driver space inference experiments.
  - `OutputData/DriverSpace/Intersection/` contains data related to intersection detection.
  - `OutputData/DriverSpace/SurroundingSampling/` contains sampled data for inference.

## Data Variables and Column Headings

### Trajectory & Sampling Data
- **x**: Global x-coordinate (meters) after coordinate transformation.
- **y**: Global y-coordinate (meters) after coordinate transformation.
- **v**: Vehicle speed in m/s.
- **round_v**: Discretised, rounded speed value used for grouping samples.
- **cangle**: Lateral interaction indicator; typically 0 (non-lateral) or 1 (with lateral interaction).
- **track_id**: Unique identifier for each vehicle trajectory.
- **frame_id**: Frame index marking the time sequence for each record.

### Inference Output Data (from experiments)
- **x_plus_lb** / **x_plus_ub**: Lower and upper bounds for spacing estimate in the positive x-direction.
- **x_minus_lb** / **x_minus_ub**: Lower and upper bounds for spacing estimate in the negative x-direction.
- **y_plus_lb** / **y_plus_ub**: Lower and upper bounds for spacing estimate in the positive y-direction.
- **y_minus_lb** / **y_minus_ub**: Lower and upper bounds for spacing estimate in the negative y-direction.
- **rx_plus_hat**, **rx_minus_hat**, **ry_plus_hat**, **ry_minus_hat**: Parameter estimates for driver space in x and y directions.
- Additional columns include standard errors (`stderr_*`) and p-values (`pval_*`) for the corresponding estimates.

## Usage Notes

- Data files are stored in standard formats (HDF5 or CSV) and can be loaded using Python packages such as `pandas` (e.g., `pd.read_hdf` or `pd.read_csv`).
- The dataset is organised to support analysis of vehicle spacing and interaction dynamics at intersections under different experimental settings.
- For replication details, experiment code is open-sourced at <https://github.com/Yiru-Jiao/DriverSpaceInference>.
