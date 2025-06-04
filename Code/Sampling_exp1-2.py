'''
Sampling for experiment 1: driver space at the unsignalized intersections in pNEUMA
(Section 4.1 and 4.2)

'''

import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.

def angle(vec1x, vec1y, vec2x, vec2y):
    sin = vec1x * vec2y - vec2x * vec1y  
    cos = vec1x * vec2x + vec1y * vec2y
    return -np.arctan2(sin, cos) * (180 / np.pi)


target_intersections = pd.read_csv(parent_dir + '/OutputData/DriverSpace/Intersection/target_intersections.csv')
target_intersections = target_intersections[target_intersections['signalized']<0.5]

data_files = {dx:sorted(glob.glob(parent_dir + '/InputData/pNEUMA/' + dx + '/data_*.h5')) for dx in ['d'+str(did+1) for did in range(10)]}
save_path = parent_dir + '/OutputData/DriverSpace/SurroundingSampling/Intersection'


# Select data in the unsignalized intersections
interval = 5 # reduce the recording frequency to 0.2s
for file_order in range(15):
    print('---- Data loading ----')
    data_all = []
    for dx in tqdm(['d'+str(did+1) for did in range(10)]):
        data_file = data_files[dx][file_order]
        data = pd.read_hdf(data_file, key='data')
        to_del = data.groupby('track_id',as_index=False).nth([0,1,2,3,4])
        data = pd.concat([data,to_del]).drop_duplicates(keep=False) # remove data that are unstable estimated at the beginning of a track
        to_del = []
        
        selected_frames = data.frame_id.drop_duplicates().values
        selected_frames = selected_frames[np.arange(0,len(selected_frames),interval)]
        data = data.set_index('frame_id').loc[selected_frames].reset_index()
        data['hx'] = np.cos(data.psi_kf)
        data['hy'] = np.sin(data.psi_kf)

        data['track_id'] = data.track_id.astype(str)+dx
        data['unique_id'] = data.frame_id.astype(str)+'-'+data.track_id
        data = data[(data.agent_type!='Motorcycle')&(data.agent_type!='Undefined')][['unique_id','frame_id','track_id','x','y','vx_kf','vy_kf','hx','hy','length','width']]
        data_all.append(data)
        data = []
    data_all = pd.concat(data_all)


    print('---- Sampling ----')
    pairs_order = []
    for int_id in tqdm(target_intersections.index.values):
        x, y, radius = target_intersections.loc[int_id][['x','y','radius']].values
        int_data = data_all[((data_all['x']-x)**2+(data_all['y']-y)**2)<radius**2]

        # Sampling
        intersection_name = target_intersections.loc[int_id]['dx'] + '-' + str(target_intersections.loc[int_id]['id'])
        # print('---- ' + intersection_name + ' data size ' + str(len(int_data['unique_id'].unique())) + ' ----')
        pair_idx = int_data.groupby('frame_id').apply(lambda x : pd.DataFrame.from_records(combinations(x['unique_id'], 2)))
        if len(pair_idx)>0:
            pairs = pd.DataFrame({'frame_id':pair_idx.index.get_level_values(0).values, 'uniqi':pair_idx[0].values, 'uniqj':pair_idx[1].values})
            pairs[['x_i','y_i','vx_i','vy_i','hx_i','hy_i','length_i','width_i']] = int_data.set_index('unique_id').reindex(index=pair_idx[0].values)[['x','y','vx_kf','vy_kf','hx','hy','length','width']].values
            pairs[['x_j','y_j','vx_j','vy_j','hx_j','hy_j','length_j','width_j']] = int_data.set_index('unique_id').reindex(index=pair_idx[1].values)[['x','y','vx_kf','vy_kf','hx','hy','length','width']].values
            pairs['i'] = int_data.set_index('unique_id').reindex(index=pair_idx[0].values)['track_id'].values
            pairs['j'] = int_data.set_index('unique_id').reindex(index=pair_idx[1].values)['track_id'].values
            pairs = pairs.drop(columns=['uniqi','uniqj'])
            pair_idx = []
            int_data = []

            dvx = pairs['vx_i']-pairs['vx_j']
            dvy = pairs['vy_i']-pairs['vy_j']
            pairs = pairs[(dvx!=0)|(dvy!=0)] # proximity resistance cannot be computed for two vehicles with a relative velocity of (0,0)
            pairs['x'] = dvy/np.sqrt(dvx**2+dvy**2)*(pairs.x_j-pairs.x_i)-dvx/np.sqrt(dvx**2+dvy**2)*(pairs.y_j-pairs.y_i)
            pairs['y'] = dvx/np.sqrt(dvx**2+dvy**2)*(pairs.x_j-pairs.x_i)+dvy/np.sqrt(dvx**2+dvy**2)*(pairs.y_j-pairs.y_i)
            pairs['v'] = np.sqrt(dvx**2 + dvy**2)
            pairs['vi'] = np.sqrt(pairs.vx_i**2+pairs.vy_i**2)
            pairs['vj'] = np.sqrt(pairs.vx_j**2+pairs.vy_j**2)

            # pairs = pairs[(abs(pairs.x)<15)&(abs(pairs.y)<100)]

            # remove overlapping (<0.5m) vehicles
            remove_list = pairs[(pairs.x**2+pairs.y**2)<0.25].j.drop_duplicates().values
            pairs = pairs[(~np.isin(pairs.j, remove_list))&(~np.isin(pairs.i, remove_list))]

            # with or without lateral interaction
            canglev = angle(pairs.hx_i, pairs.hy_i, pairs.hx_j, pairs.hy_j)
            cangle = np.empty(canglev.shape)
            cangle[abs(canglev)<5] = 0
            cangle[abs(canglev)>175] = 0
            cangle[(canglev<=175)&(canglev>=5)] = 1
            cangle[(canglev>=-175)&(canglev<=-5)] = 1
            pairs['cangle'] = cangle
            pairs['cangle_deg'] = canglev

            pairs = pairs.reset_index(drop=True)[['x','y','v','vi','vj','cangle','cangle_deg']]
            pairs['intersection'] = intersection_name
            pairs_order.append(pairs)
            pairs = []

    pairs_order = pd.concat(pairs_order, ignore_index=True)
    pairs_order.to_hdf(save_path + '/sample_' + str(file_order) + '.h5', key='sample')
    pairs_order = []
    
    
# Merge samples
print('---- Merging ----')
sample_files = sorted(glob.glob(save_path + '/sample_*.h5'))
samples = []
for sample_file in sample_files:
    sample = pd.read_hdf(sample_file, key='sample')
    samples.append(sample)
samples = pd.concat(samples, ignore_index=True)
samples = samples[samples.v<=20]

print('---- Grouping ----')
vehnum = 50000 # the least number of vehicle pairs in a group
test = []
for cangle in [0,1]:
    sample = samples[(samples.cangle==cangle)]
    sample = sample.sort_values(by='v')
    sample['round_v'] = np.round(sample.v,1)
    groups = sample.groupby('round_v').x.count()
    threshold = groups[groups>=vehnum].index[-1]
    sample1 = []
    for roundv in sample[sample.round_v<=threshold].round_v.unique():
        samp = sample[sample.round_v==roundv]
        if len(samp)>vehnum:
            samp = samp.loc[np.random.choice(samp.index.values, vehnum)]
        sample1.append(samp)
    sample1 = pd.concat(sample1, axis=0)
    sample2 = sample[sample.round_v>threshold].copy()
    sample2['round_v'] = np.arange(len(sample2))//vehnum
    sample2['round_v'] = (np.round(sample2.groupby('round_v').v.mean(),1)).reindex(sample2.round_v).values
    sample = pd.concat((sample1, sample2))
    test.append(sample)
    print('--- '+str(cangle)+' -- '+str(len(sample.round_v.unique()))+' ----')
samples = pd.concat(test, axis=0)
test = []
samples[['x','y','v','round_v','cangle']].to_hdf(save_path + '/unsignalized_samples_toinfer.h5', key='samples')