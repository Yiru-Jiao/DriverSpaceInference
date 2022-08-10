# Code for "Probabilistic representation for driver space and its inference from urban trajectory data"

## In order to repeat the experiments:

__Step 1.__ Download raw data from <https://open-traffic.epfl.ch/index.php/downloads/> and save them in the folder "RawDatasets".

__Step 2.__ Run `PreprocessingAndSampling.py` to process the raw data, transform and correct coordinates, and sample pairs of vehicles.

__Step 3.__ Run `Exp_pNEUMA.py` to repeat our experiments in the article.

__Step 4.__ Use `ResultEvaluation.ipynb` to save accepted driver space inference results.

## In order to apply the method to another dataset:

__Step 1.__ Save raw data in the folder "RawDatasets".

__Step 2.__ Create code to align the format of the new dataset to the format of the data to be saved in the folder "InputData".

__Step 3.__ Design your application according to the code in `Exp_pNEUMA.py`.

__Step 4.__ Accept and reject inference results according to the code in `ResultEvaluation.ipynb`.


# Abstract
Interaction between vehicles in the urban environment is a complex spatial and temporal phenomenon. Particularly in lane-free scenarios such as unsignalised intersections, shared spaces, and parking lots, modelling urban vehicle interaction is required to be two-dimensional. Current approaches to modelling two-dimensional vehicle interaction are predominantly based on the distance between vehicles. However, equally distant vehicles can interact differently when they are at different positions and drive towards different directions. As a result, distance-based approaches are ill-suited for modelling two-dimensional vehicle interaction. In this study, we propose a new method to infer probabilistic driver space from urban trajectory data. More specifically, our method postulates that drivers tend to keep different comfortable distances from other vehicles in different scenarios and directions, and hence maintain a scenario-dependent asymmetric driver space around their vehicles. Based on this notion, the method infers the probabilistic driver space of a vehicle by estimating the accumulative density of other vehicles surrounding the vehicle. To test the method, we apply it to a well-known large-scale urban trajectory dataset called pNEUMA. Through the experiments, this method produces stable and behaviorally intuitive  driver space representations. Our results show asymmetric driver spaces which generate plausible two-dimensional fundamental diagrams, giving insights into urban traffic. Therefore, this study offers a contribution to the growing research field on two-dimensional modelling of urban vehicle interaction.

# Citation
The paper is under review.
