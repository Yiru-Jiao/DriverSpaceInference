import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import DriverSpaceInference as dsi


parent_dir = os.path.abspath('..')
width = 2 
length = 4.5 # Width and length to calculate initial parameters

def exp(speed, samples):
    num_sample = int(len(samples)*0.9)
    if num_sample==0 or len(samples[samples.x<0])/len(samples[samples.x>0])>100 or len(samples[samples.x<0])/len(samples[samples.x>0])<0.01:
        return np.zeros(28)*np.nan
    else:
        # Infer driver space
        randomstate = 0
        reject = True
        while reject and randomstate<=10:
            sample = samples.loc[np.random.RandomState(randomstate).choice(samples.index.values,num_sample,replace=False)]
            # sys.stdout.write('---- speed = ' + str(speed) + ', randomstate = ' + str(randomstate) + ' ----\n')
            
            try:
                ry_plus = np.percentile(sample.y[(sample.y>=0)&(abs(sample.x)<width/2)], 0.1)
                ry_minus = np.percentile(sample.y[(sample.y<0)&(abs(sample.x)<width/2)], 0.1)
            except:
                randomstate = 35
                continue

            try:
                rx_plus = np.percentile(sample.x[(sample.x>0)&(sample.y<=max(length/2,ry_plus/2))&(sample.y>=-max(length/2,ry_minus/2))], 0.1)
                rx_minus = np.percentile(-sample.x[(sample.x<0)&(sample.y<=max(length/2,ry_plus/2))&(sample.y>=-max(length/2,ry_minus/2))], 0.1)
            except:
                randomstate = 35
                continue
            
            driver_space = dsi.drspace(sample, eps=1e-4, width=width, length=length, percentile=0.1)
            corrcoef_xy = driver_space.corrcoef_xy
            bounds, estimation, stderror_beta, pvalue_beta = driver_space.Inference(max_iter=30, workers=10)
            bounds = np.round(bounds,1)
            estimation = np.round(estimation,1)
            driver_space = [] # release occupied memory

            if np.all(np.isnan(bounds)):
                pfailure = False
                rxfailure = False
                ryfailure = False
            else:
                if np.all(np.isnan(estimation)):
                    pfailure = False
                    rxfailure = False
                    ryfailure = False
                else:
                    pfailure = np.any(pvalue_beta>0.05)
                    rxfailure = (estimation[0]==bounds[0])|(estimation[1]==bounds[2])
                    ryfailure = (estimation[2]==bounds[4])|(estimation[2]==bounds[5])|(estimation[3]==bounds[6])|(estimation[3]==bounds[7])
                
            reject = np.any([pfailure, rxfailure, ryfailure])|np.any(np.isnan(estimation))
            randomstate += 1

        if randomstate<35:
            return np.concatenate(([reject, randomstate, len(sample), corrcoef_xy],
                                    bounds,
                                    estimation,
                                    stderror_beta,
                                    pvalue_beta))
        else:
            return np.zeros(28)*np.nan


# Exp1: Driver space inference (corresponding to subsection 3.2 in our article)
for dx in ['d'+str(did+1) for did in range(10)]:
    sample_path = parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + dx

    for cangle, abbr in zip([0,1], ['nonlat','withlat']):
        surrounding = pd.read_hdf(sample_path + '/samples_toinfer_' + dx + '.h5', key='samples')
        surrounding = surrounding[surrounding.cangle==cangle]
        sys.stdout.write('---- ' + abbr + ' -- ' + dx + ' ----\n')
        
        roundvs = surrounding.round_v.unique()
        savefile = np.zeros(roundvs.shape).astype(bool)
        savefile[np.arange(10,len(roundvs),10)] = True
        savefile[-1] = True
        
        results = pd.DataFrame(np.empty((len(roundvs),28)), columns=['reject','randomstate','num_sample','corrcoef_xy',
                                                                     'x_plus_lb','x_plus_ub','x_minus_lb','x_minus_ub','y_plus_lb','y_plus_ub','y_minus_lb','y_minus_ub', 
                                                                     'rx_plus_hat','rx_minus_hat','ry_plus_hat','ry_minus_hat','bx_plus_hat','bx_minus_hat','by_plus_hat','by_minus_hat',
                                                                     'stderr_bx_plus','stderr_bx_minus','stderr_by_plus','stderr_by_minus',
                                                                     'pval_bx_plus','pval_bx_minus','pval_by_plus','pval_by_minus'])
        results['round_v'] = roundvs
        results = results.set_index('round_v')

        for speed, savef in tqdm(zip(roundvs, savefile), total=len(roundvs)):
            result = exp(speed, surrounding[surrounding.round_v==speed])
            results.loc[speed] = result
            if savef:
                results.reset_index().to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Inference/' + abbr + '_' + dx + '.csv')
            
    sys.stdout.write('Results saved successfully\n')


# Exp2: Bootstrapping (corresponding to subsection 3.4 in our article)
dx = 'd10'
sample_path = parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + dx

for cangle, abbr in zip([0], ['nonlat']):
    surrounding = pd.read_hdf(sample_path + '/samples_toinfer_' + dx + '.h5', key='samples')
    surrounding = surrounding[surrounding.cangle==cangle]
    sys.stdout.write('---- ' + abbr + ' -- ' + dx + ' ----\n')
    
    roundvs = np.array([3.6,5.6,6.4,7.5,8.3,10.6])
    results = pd.DataFrame(np.empty((20*len(roundvs),28)), columns=['reject','randomstate','num_sample','corrcoef_xy',
                                                                    'x_plus_lb','x_plus_ub','x_minus_lb','x_minus_ub','y_plus_lb','y_plus_ub','y_minus_lb','y_minus_ub', 
                                                                    'rx_plus_hat','rx_minus_hat','ry_plus_hat','ry_minus_hat','bx_plus_hat','bx_minus_hat','by_plus_hat','by_minus_hat',
                                                                    'stderr_bx_plus','stderr_bx_minus','stderr_by_plus','stderr_by_minus',
                                                                    'pval_bx_plus','pval_bx_minus','pval_by_plus','pval_by_minus'])
    results['round_v'] = np.repeat(roundvs,20)
    results = results.set_index('round_v')
    
    randomstates = []
    for speed in tqdm(roundvs):
        iter_count = 0
        randomstate = 0
        result_speed = []
        while iter_count<20:
            samples = surrounding[surrounding.round_v==speed]
            samples = samples.loc[np.random.RandomState(randomstate).choice(samples.index.values,int(0.9*len(samples)),replace=False)]
            result = exp(speed, samples)
            if result[0]==0:
                result_speed.append(result)
                randomstates.append(randomstate)
                iter_count += 1
            randomstate += 1
        results.loc[speed] = np.array(result_speed)
        results.reset_index().to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Inference/Bootstrapping_' + abbr + '_' + dx + '.csv')
    
    results['RandState'] = randomstates
    results.reset_index().to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Inference/Bootstrapping_' + abbr + '_' + dx + '.csv')
            
sys.stdout.write('Results saved successfully\n')


# Exp3: Asymmetric driver space (corresponding to subsection 4.1 in our article)

for dx, cangle, abbr1 in zip(['d4', 'd9'], [1,0], ['withlat','nonlat']):
    sample_path = parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + dx
    for direction, abbr2 in zip([1,-1],['front','back']):
        surrounding = pd.read_hdf(sample_path + '/samples_toinfer_frontback_' + dx + '.h5', key='samples')
        surrounding = surrounding[(surrounding.cangle==cangle)&(surrounding.direction==direction)]
        sys.stdout.write('---- ' + abbr1 + '--' + abbr2 + ' -- ' + dx + ' ----\n')

        roundvs = surrounding.round_v.unique()
        savefile = np.zeros(roundvs.shape).astype(bool)
        savefile[np.arange(10,len(roundvs),10)] = True
        savefile[-1] = True

        results = pd.DataFrame(np.empty((len(roundvs),28)), columns=['reject','randomstate','num_sample','corrcoef_xy',
                                                                        'x_plus_lb','x_plus_ub','x_minus_lb','x_minus_ub','y_plus_lb','y_plus_ub','y_minus_lb','y_minus_ub', 
                                                                        'rx_plus_hat','rx_minus_hat','ry_plus_hat','ry_minus_hat','bx_plus_hat','bx_minus_hat','by_plus_hat','by_minus_hat',
                                                                        'stderr_bx_plus','stderr_bx_minus','stderr_by_plus','stderr_by_minus',
                                                                        'pval_bx_plus','pval_bx_minus','pval_by_plus','pval_by_minus'])
        results['round_v'] = roundvs
        results = results.set_index('round_v')

        for speed, savef in tqdm(zip(roundvs, savefile), total=len(roundvs)):
            result = exp(speed, surrounding[surrounding.round_v==speed])
            results.loc[speed] = result
            if savef:
                results.reset_index().to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Inference/' + abbr1 + '_' + abbr2 + '_' + dx + '.csv')
            
sys.stdout.write('Results saved successfully\n')
