# Code for "Probabilistic representation for driver space and its inference from urban trajectory data"

## In order to repeat the experiments:

__Step 1.__ Download raw data from <https://open-traffic.epfl.ch/index.php/downloads/> and save them in the folder "RawDatasets".

__Step 2.__ Run `Preprocessing.py` to process the raw data.

__Step 3.__ Run `Sampling.py` to transform and correct coordinates, and sample pairs of vehicles.

__Step 4.__ Run `Exp_pNEUMA.py` to repeat our experiments in the article.


## In order to apply the method to another dataset:

__Step 1.__ Save raw data in the folder "RawDatasets".

__Step 2.__ Create code to align the format of the new dataset to the format of the data to be saved in the folder "InputData".

__Step 3.__ Design your application according to the code in `Exp_pNEUMA.py`.


# Abstract
Interaction between vehicles in the urban environment is a complex spatial and temporal phenomenon. Current modelling of urban traffic is predominantly focused on microscopic driving operations and macroscopic network dynamics. The mesoscopic modelling of spacing distribution, which links microscopic and macroscopic traffic flow models, has been considered only in queues. However, vehicle spacing may vary towards different directions, particularly in urban scenarios such as unsignalised intersections, shared spaces, and parking lots. In this study, we propose a new method to infer probabilistic driver space from urban trajectory data. More specifically, our method postulates that drivers tend to keep different comfortable distances from other vehicles in different scenarios and directions, and hence maintain a scenario-dependent asymmetric driver space. Based on this notion, the method infers the probabilistic driver space of a vehicle by estimating the accumulative density of other vehicles surrounding the vehicle. To test the method, we apply it to a well-known large-scale urban trajectory dataset called pNEUMA. Through the experiments, the proposed method produces consistent and behaviourally intuitive driver space representations. These representations quantify the distribution of two-dimensional vehicle spacing, which allows for generating probabilistic Fundamental Diagrams for urban scenarios. Thereby, this study offers a new method and new insights that lead to better understanding urban vehicle interaction.

# Citation
The paper is under peer-review.
