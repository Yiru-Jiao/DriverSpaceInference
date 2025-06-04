'''
Sampling for experiment 4: impact of absolute speed
(Section 5.2)

'''

import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.


def angle(vec1x, vec1y, vec2x, vec2y):
    sin = vec1x * vec2y - vec2x * vec1y  
    cos = vec1x * vec2x + vec1y * vec2y
    return -np.arctan2(sin, cos) * (180 / np.pi)


# Merge samples
print('---- Merging ----')
save_path = parent_dir + '/OutputData/DriverSpace/SurroundingSampling/Intersection'
sample_files = sorted(glob.glob(save_path + '/sample_*.h5'))
samples = []
for sample_file in sample_files:
    sample = pd.read_hdf(sample_file, key='sample')
    samples.append(sample)
samples = pd.concat(samples, ignore_index=True)
samples = samples[samples.v<=20]

print('---- Grouping ----') ## for different relative speeds
vehnum = 10000 # the least number of vehicle pairs in a group
for cangle, lat in zip([0,1],['nonlat','withlat']):
    print('---- '+lat+' ----')
    for rvmin, rvmax, suffix in tqdm(zip([0,1.5,3.5,5.5],[0.5,2.5,4.5,6.5],['0','2','4','6']), total=4):
        sample = samples[(samples.cangle==cangle)&(samples.v>rvmin)&(samples.v<=rvmax)]
        sample = sample.sort_values(by='vi')
        sample['round_v'] = np.round(sample['vi'],1)
        groups = sample.groupby('round_v').x.count()
        try:
            threshold = groups[groups>=vehnum].index[-1]
            sample1 = []
            for roundv in sample[sample.round_v<=threshold].round_v.unique():
                samp = sample[sample.round_v==roundv]
                if len(samp)>vehnum:
                    samp = samp.loc[np.random.choice(samp.index.values, vehnum)]
                sample1.append(samp)
            sample1 = pd.concat(sample1, axis=0)
        except:
            threshold = 0
            sample1 = samples[samples.round_v<0].copy()
        sample2 = sample[sample.round_v>threshold].copy()
        sample2['round_v'] = np.arange(len(sample2))//vehnum
        sample2['round_v'] = (np.round(sample2.groupby('round_v').vi.mean(),1)).reindex(sample2.round_v).values
        sample = pd.concat((sample1, sample2))
        print('---- '+suffix+' '+str(len(sample.round_v.unique()))+' ----')
        sample['v'] = sample['vi']
        sample[['x','y','v','round_v']].to_hdf(save_path + '/samples_toinfer_rv' + suffix + '_' + lat + '.h5', key='samples')