import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def angle(vec1x, vec1y, vec2x, vec2y):
    sin = vec1x * vec2y - vec2x * vec1y  
    cos = vec1x * vec2x + vec1y * vec2y
    return -np.arctan2(sin, cos) * (180 / np.pi)

parent_dir = os.path.abspath('..')

# Exp1:
for dx,interval in zip(['d'+str(did+1) for did in range(10)],[5,15,10,10,15,20,20,25,20,15]):
    data_files = sorted(glob.glob(parent_dir + '/InputData/pNEUMA/' + dx + '/*.h5'))
    save_path = parent_dir + '/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + dx
    
    if dx!='d10':
        data_titles = [data_file[-24:-3] for data_file in data_files]
    else:
        data_titles = [data_file[-25:-3] for data_file in data_files]

    for data_file, data_title in zip(data_files,data_titles):
        print('---- Sampling ' + data_title + ' ----')
        
        data = pd.read_hdf(data_file, key='data').reset_index()
        to_del = data.groupby('track_id',as_index=False).nth([0,1,2,3,4])
        data = pd.concat([data,to_del]).drop_duplicates(keep=False) # remove data that are unstable estimated at the beginning of a track
        to_del = []
        selected_frames = data.frame_id.drop_duplicates().values
        selected_frames = selected_frames[np.arange(0,len(selected_frames),interval)]
        data = data.set_index('frame_id').loc[selected_frames].reset_index()

        data['unique_id'] = data.frame_id.astype(str)+'-'+data.track_id.astype(str)
        pair_idx = data.groupby('frame_id').apply(lambda x : pd.DataFrame.from_records(combinations(x.unique_id, 2)))

        data = data.set_index('unique_id')
        pairs = pd.DataFrame({'frame_id':pair_idx.index.get_level_values(0).values, 'uniqi':pair_idx[0].values, 'uniqj':pair_idx[1].values})

        pairs[['i','x_i','y_i','vx_i','vy_i','length_i','width_i']] = data.reindex(index=pair_idx[0].values)[['track_id','x','y','vx_kf','vy_kf','length','width']].values
        pairs[['j','x_j','y_j','vx_j','vy_j','length_j','width_j']] = data.reindex(index=pair_idx[1].values)[['track_id','x','y','vx_kf','vy_kf','length','width']].values
        pairs = pairs.drop(columns=['uniqi','uniqj'])
        pair_idx = []
        data = []
        print('---- Data released ----')

        dvx = pairs.vx_i-pairs.vx_j
        dvy = pairs.vy_i-pairs.vy_j
        pairs = pairs[(dvx!=0)|(dvy!=0)] # proximity resistance cannot be computed for two vehicles with a relative velocity of (0,0)
        pairs['x'] = dvy/np.sqrt(dvx**2+dvy**2)*(pairs.x_j-pairs.x_i)-dvx/np.sqrt(dvx**2+dvy**2)*(pairs.y_j-pairs.y_i)
        pairs['y'] = dvx/np.sqrt(dvx**2+dvy**2)*(pairs.x_j-pairs.x_i)+dvy/np.sqrt(dvx**2+dvy**2)*(pairs.y_j-pairs.y_i)
        pairs['v'] = np.sqrt(dvx**2 + dvy**2)

        pairs = pairs[(abs(pairs.x)<30)&(abs(pairs.y)<100)]

        # remove overlapping (<0.5m) vehicles
        remove_list = pairs[(pairs.x**2+pairs.y**2)<0.25].j.drop_duplicates().values
        pairs = pairs[(~np.isin(pairs.j, remove_list))&(~np.isin(pairs.i, remove_list))]

        # with or without lateral interaction
        cangles = angle(pairs.vx_i, pairs.vy_i, (pairs.vx_i-pairs.vx_j), (pairs.vy_i-pairs.vy_j))
        cangle = np.empty(cangles.shape)
        cangle[abs(cangles)<10] = 0
        cangle[abs(cangles)>170] = 0
        cangle[(cangles<=170)&(cangles>=10)] = 1
        cangle[(cangles>=-170)&(cangles<=-10)] = 1
        pairs['cangle'] = cangle

        pairs = pairs[['x','y','v','cangle']]        
        pairs.to_hdf(save_path + '/sample_' + data_title + '.h5', key='sample')
        pairs = []
    
    sample_files = sorted(glob.glob(save_path + '/sample_*.h5'))
    samples = []
    for sample_file in sample_files:
        sample = pd.read_hdf(sample_file, key='sample')
        samples.append(sample)
    samples = pd.concat(samples, ignore_index=True)
    samples = samples[samples.v<=20]
    
    print('---- Grouping ----')
    test = []
    for cangle in [0,1]:
        sample = samples[samples.cangle==cangle]
        sample = sample.sort_values(by='v')
        sample['round_v'] = np.round(sample.v,1)
        groups = sample.groupby('round_v').x.count()
        threshold = groups[groups<200000].index[0]
        sample1 = sample[sample.round_v<threshold].copy()
        sample2 = sample[sample.round_v>=threshold].copy()
        sample2['round_v'] = np.arange(len(sample2))//200000
        sample2['round_v'] = (np.round(sample2.groupby('round_v').v.mean(),1)).reindex(sample2.round_v).values
        sample = pd.concat((sample1, sample2))
        test.append(sample)
    samples = pd.concat(test)
    test = []
    samples.to_hdf(save_path + '/samples_toinfer_' + dx + '.h5', key='samples')

# Exp2:
# for dx,interval,c in zip(['d4', 'd9'],[5,10],[1,0]):
for dx,interval,c in zip(['d1'],[2],[1]):
    data_files = sorted(glob.glob(parent_dir + '/InputData/pNEUMA/' + dx + '/*.h5'))
    save_path = parent_dir + '/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + dx
    
    data_titles = [data_file[-25:-3] for data_file in data_files]

    for data_file, data_title in zip(data_files,data_titles):
        print('---- Sampling ' + data_title + ' ----')
        
        data = pd.read_hdf(data_file, key='data').reset_index()
        to_del = data.groupby('track_id',as_index=False).nth([0,1,2,3,4])
        data = pd.concat([data,to_del]).drop_duplicates(keep=False) # remove data that are unstable estimated at the beginning of a track
        to_del = []
        selected_frames = data.frame_id.drop_duplicates().values
        selected_frames = selected_frames[np.arange(0,len(selected_frames),interval)]
        data = data.set_index('frame_id').loc[selected_frames].reset_index()

        data['unique_id'] = data.frame_id.astype(str)+'-'+data.track_id.astype(str)
        pair_idx = data.groupby('frame_id').apply(lambda x : pd.DataFrame.from_records(combinations(x.unique_id, 2)))

        data = data.set_index('unique_id')
        pairs = pd.DataFrame({'frame_id':pair_idx.index.get_level_values(0).values, 'uniqi':pair_idx[0].values, 'uniqj':pair_idx[1].values})

        pairs[['i','x_i','y_i','vx_i','vy_i','length_i','width_i']] = data.reindex(index=pair_idx[0].values)[['track_id','x','y','vx_kf','vy_kf','length','width']].values
        pairs[['j','x_j','y_j','vx_j','vy_j','length_j','width_j']] = data.reindex(index=pair_idx[1].values)[['track_id','x','y','vx_kf','vy_kf','length','width']].values
        pairs = pairs.drop(columns=['uniqi','uniqj'])
        pair_idx = []
        data = []
        print('---- Data released ----')

        dvx = pairs.vx_i-pairs.vx_j
        dvy = pairs.vy_i-pairs.vy_j
        pairs = pairs[(dvx!=0)|(dvy!=0)] # proximity resistance cannot be computed for two vehicles with a relative velocity of (0,0)
        pairs['x'] = dvy/np.sqrt(dvx**2+dvy**2)*(pairs.x_j-pairs.x_i)-dvx/np.sqrt(dvx**2+dvy**2)*(pairs.y_j-pairs.y_i)
        pairs['y'] = dvx/np.sqrt(dvx**2+dvy**2)*(pairs.x_j-pairs.x_i)+dvy/np.sqrt(dvx**2+dvy**2)*(pairs.y_j-pairs.y_i)
        pairs['v'] = np.sqrt(dvx**2 + dvy**2)

        pairs = pairs[(abs(pairs.x)<30)&(abs(pairs.y)<100)]

        # remove overlapping (<0.5m) vehicles
        remove_list = pairs[(pairs.x**2+pairs.y**2)<0.25].j.drop_duplicates().values
        pairs = pairs[(~np.isin(pairs.j, remove_list))&(~np.isin(pairs.i, remove_list))]

        # with or without lateral interaction
        cangles = angle(pairs.vx_i, pairs.vy_i, (pairs.vx_i-pairs.vx_j), (pairs.vy_i-pairs.vy_j))
        cangle = np.empty(cangles.shape)
        cangle[abs(cangles)<10] = 0
        cangle[abs(cangles)>170] = 0
        cangle[(cangles<=170)&(cangles>=10)] = 1
        cangle[(cangles>=-170)&(cangles<=-10)] = 1
        pairs['cangle'] = cangle

        # from front or back
        directions = angle(pairs.vx_i, pairs.vy_i, (pairs.x_j-pairs.x_i), (pairs.y_j-pairs.y_i))
        direction = np.empty(directions.shape)
        direction[abs(directions)<=90] = 1
        direction[abs(directions)>90] = -1
        pairs['direction'] = direction
        
        pairs = pairs[['x','y','v','cangle','direction']]
        pairs = pairs[pairs.cangle==c]
        pairs.to_hdf(save_path + '/fb_sample_' + data_title + '.h5', key='sample')
        pairs = []
    
    sample_files = sorted(glob.glob(save_path + '/fb_sample_*.h5'))
    samples = []
    for sample_file in sample_files:
        sample = pd.read_hdf(sample_file, key='sample')
        samples.append(sample)
    samples = pd.concat(samples, ignore_index=True)
    samples = samples[samples.v<=20]
    
    print('---- Grouping front/back----')
    test = []
    for direction in [1,-1]:
        print(samples.cangle.unique())
        sample = samples[(samples.direction==direction)]
        sample = sample.sort_values(by='v')
        sample['round_v'] = np.round(sample.v,1)
        groups = sample.groupby('round_v').x.count()
        threshold = groups[groups<200000].index[0]
        sample1 = sample[sample.round_v<threshold].copy()
        sample2 = sample[sample.round_v>=threshold].copy()
        sample2['round_v'] = np.arange(len(sample2))//200000
        sample2['round_v'] = (np.round(sample2.groupby('round_v').v.mean(),1)).reindex(sample2.round_v).values
        sample = pd.concat((sample1, sample2))
        test.append(sample)
    samples = pd.concat(test, axis=0)
    test = []
    samples.to_hdf(save_path + '/samples_toinfer_frontback_' + dx + '.h5', key='samples')
