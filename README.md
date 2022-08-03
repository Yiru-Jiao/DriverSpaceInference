# Code for "Probabilistic representation for driver space and its inference from urban trajectory data"

## In order to repeat the experiments:

__Step 1.__ Download raw data from [pNEUMA]<https://open-traffic.epfl.ch/index.php/downloads/> and save them in the folder "RawDatasets".

__Step 2.__ Run `PreprocessingAndSampling.py` to process the raw data, transform and correct coordinates, and sample pairs of vehicles.

__Step 3.__ Run `Exp_pNEUMA.py` to repeat our experiments in the article.

__Step 4.__ Use `ResultEvaluation.ipynb` to save accepted driver space inference results.

## In order to apply the method to another dataset:

__Step 1.__ Save raw data in the folder "RawDatasets".

__Step 2.__ Create code to align the format of the new dataset to 


# Citation
The paper has been published on the peer-review journal named "Information Fusion". Please cite it as:

Yiru Jiao, Yongli Li. (2021). An active opinion dynamics model: the gap between the voting result and group opinion. Information Fusion, 65, 128-146. doi: https://doi.org/10.1016/j.inffus.2020.08.009
 
# Abstract
Originally developed to simulate the evolution of public opinion, opinion dynamics models have also been successfully applied to market prices and advertising. However, passive interactions initiated by locational or social relationships in these models are insufficient to characterize purposeful behaviors such as canvass or trade, where people are driven by their specific inner causes. Here we propose an active model in which people tend to communicate with someone who is more likely to be an ally, and game theoretically decide whether to interact. Simulations of the model highlight the macroscopic development of opinion evolution, showing the ubiquitous gap between people’s voting result and their collective opinion, and how it narrows with the stabilization of opinion evolution. Our results help explain why group opinion rarely reverse its stance and the significance of an inclusiveness that is neither too high nor too low. Additionally, we find and testify the probability distribution of group opinion change, which contributes to predict how much the collective opinion of a group will change after full discussion.
