import os
import sys
import glob
import numpy as np
import DriverSpaceApproximation as ds

parent_dir = os.path.abspath('..')

# pNEUMA
for dx in ['d'+str(did+1) for did in range(10)]:
    data_files = sorted(glob.glob(parent_dir + '/InputData/pNEUMA/' + dx + '/*.csv'))
    if dx!='d10':
        data_titles = [data_file[-25:-4] for data_file in data_files]
    else:
        data_titles = [data_file[-26:-4] for data_file in data_files]

# ////////////////////
#      Linux       ///
# ////////////////////

# 1 Preprocessing

    for data_title in data_titles:
        sys.stdout.write('---- Preprocessing ' + data_title + ' ----\n')

        ### 1.1 Dataset transformation
        dX = ds.preprocess(dataset='pNEUMA', open_path=parent_dir+'/RawDatasets/pNEUMA/'+dx, data_title=data_title, head_period=5)
        dX.transform(save_path=parent_dir+'/InputData/pNEUMA/'+dx)
        data = dX.data.copy()
        data_overview = dX.data_overview.copy()
        dX = []

        sys.stdout.write('---- Sampling ' + data_title + ' ----\n')

        ### 1.2 Sampling
        #// Exclude motorcycles and pedestrians
        data = data.loc[data_overview[np.logical_and(data_overview['type']!='Motorcycle',data_overview['type']!='Undefined')].index]
        
        samp = ds.sample(dataset='pNEUMA', data=data, data_overview=data_overview, data_title=data_title)
        samp.sampling(save_path=parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/'+dx, frame_interval=25)



# //////////////////
#     Windows    ///
# //////////////////

    # if __name__ == '__main__':

    #     for data_title in data_titles:
    #         sys.stdout.write('---- Preprocessing ' + data_title + ' ----\n')

    #         ### 1.1 Dataset transformation
    #         dX = ds.preprocess(dataset='pNEUMA', open_path=parent_dir+'/RawDatasets/pNEUMA/'+dx, data_title=data_title, head_period=5)
    #         dX.transform(save_path=parent_dir+'/InputData/pNEUMA/'+dx)
    #         data = dX.data.copy()
    #         data_overview = dX.data_overview.copy()
    #         dX = []

    #         sys.stdout.write('---- Sampling ' + data_title + ' ----\n')

    #         ### 1.2 Sampling
    #         #// Exclude motorcycles and pedestrians
    #         data = data.loc[data_overview[np.logical_and(data_overview['type']!='Motorcycle',data_overview['type']!='Undefined')].index]
            
    #         samp = ds.sample(dataset='pNEUMA', data=data, data_overview=data_overview, data_title=data_title)
    #         samp.sampling(save_path=parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/'+dx, frame_interval=25)
