import os
import sys
import glob
import numpy as np
import pandas as pd
import DriverSpaceApproximation as ds


parent_dir = os.path.abspath('..')
width = 2 
length = 5 # Width and length of the averaged ego vehicle in the dataset


# Exp1: Driver space inference (corresponding to subsection 3.2 in our article)

def experiment(i):
    samples = surrounding[surrounding['vy_relative']>v_relative[i]-speed_interval/2]
    samples = samples[samples['vy_relative']<=v_relative[i]+speed_interval/2]
    
    driver_space = ds.drspace(samples, width=width, length=length)
    corrcoef_xy = driver_space.corrcoef_xy

    ini_rx_plus = driver_space.r_x_plus
    ini_rx_minus = driver_space.r_x_minus
    ini_ry_plus = driver_space.r_y_plus
    ini_ry_minus = driver_space.r_y_minus

    lower_bounds, estimation, stderror_beta, pvalue_beta = driver_space.Inference(max_iter=50, workers=20)
    
    r_x_plus, r_x_minus, r_y_plus, r_y_minus, beta_x_plus, beta_x_minus, beta_y_plus, beta_y_minus = estimation
    r_x = (1+np.sign(samples['relative_x']))/2*r_x_plus + (1-np.sign(samples['relative_x']))/2*r_x_minus
    r_y = (1+np.sign(samples['relative_y']))/2*r_y_plus + (1-np.sign(samples['relative_y']))/2*r_y_minus
    beta_x = (1+np.sign(samples['relative_x']))/2*beta_x_plus + (1-np.sign(samples['relative_x']))/2*beta_x_minus
    beta_y = (1+np.sign(samples['relative_y']))/2*beta_y_plus + (1-np.sign(samples['relative_y']))/2*beta_y_minus

    num_pr = np.zeros(10)
    if np.all(np.isnan(estimation)):
        num_pr[:] = np.nan
    else:
        samples['proximity_resistance'] = np.exp(-abs(samples['relative_x']/r_x)**(beta_x) - abs(samples['relative_y']/r_y)**(beta_y))
        for j in range(10):
            num_pr[j] = samples[np.logical_and(samples['proximity_resistance']>=j/10, samples['proximity_resistance']<(j+1)/10)].shape[0]

    estimates = np.concatenate(([v_relative[i], corrcoef_xy, ini_rx_plus, ini_rx_minus, ini_ry_plus, ini_ry_minus],
                                lower_bounds,
                                estimation,
                                stderror_beta,
                                pvalue_beta,
                                num_pr), axis=None)

    return estimates

for dx in ['d'+str(did) for did in range(1,11)]:
    sample_path = parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + dx

    all_files = glob.glob(sample_path + '/*.h5')
    surrounding = []
    for filename in all_files:
        df = pd.read_hdf(filename, key='sample')
        surrounding.append(df)
    surrounding = pd.concat(surrounding, axis=0, ignore_index=True)
    sys.stdout.write('---- Samples loaded successfully ----\n')

    # Position correction
    surrounding['relative_x'] = surrounding['relative_x'] / surrounding['width_j'] * width
    surrounding['relative_y'] = surrounding['relative_y'] / surrounding['length_j'] * length

    speed_interval = 0.2
    v_relative = np.round(np.arange(0, 13 + speed_interval, speed_interval), 1)

    surrounding = surrounding[surrounding.vy_relative <= 13+speed_interval/2]

    result = np.zeros((len(v_relative), 34))
    for i in range(len(v_relative)):
        sys.stdout.write('---- relative speed = ' + str(v_relative[i]) + ', start estimation ----\n')
        result[i] = experiment(i)
        ## Save result
        result_tosave = pd.DataFrame(result, columns=['vy_relative',
                                                      'corrcoef_xy',
                                                      'ini_rx_plus',
                                                      'ini_rx_minus',
                                                      'ini_ry_plus',
                                                      'ini_ry_minus',
                                                      'x_lower_bound',
                                                      'y_lower_bound',
                                                      'r_x_plus_hat',
                                                      'r_x_minus_hat',
                                                      'r_y_plus_hat',
                                                      'r_y_minus_hat',
                                                      'beta_x_plus_hat',
                                                      'beta_x_minus_hat',
                                                      'beta_y_plus_hat',
                                                      'beta_y_minus_hat',
                                                      'stderr_beta_x_plus',
                                                      'stderr_beta_x_minus',
                                                      'stderr_beta_y_plus',
                                                      'stderr_beta_y_minus',
                                                      'pval_beta_x_plus',
                                                      'pval_beta_x_minus',
                                                      'pval_beta_y_plus',
                                                      'pval_beta_y_minus',
                                                      'num_pr_1',
                                                      'num_pr_2',
                                                      'num_pr_3',
                                                      'num_pr_4',
                                                      'num_pr_5',
                                                      'num_pr_6',
                                                      'num_pr_7',
                                                      'num_pr_8',
                                                      'num_pr_9',
                                                      'num_pr_10'])

        result_tosave.to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Approximation/RelativeSpeed/exp_' + dx + '.csv')

    sys.stdout.write('All results saved successfully\n')


# Exp2: Bootstrapping (corresponding to subsection 3.3 in our article)

sample_path = parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/d6'

all_files = glob.glob(sample_path + '/*.h5')
surrounding = []
for filename in all_files:
    df = pd.read_hdf(filename, key='sample')
    surrounding.append(df)
surrounding = pd.concat(surrounding, axis=0, ignore_index=True)
sys.stdout.write('---- Samples loaded successfully ----\n')

# Position correction
surrounding['relative_x'] = surrounding['relative_x'] / surrounding['width_j'] * width
surrounding['relative_y'] = surrounding['relative_y'] / surrounding['length_j'] * length

speed_interval = 0.2
v_relative = np.round(np.arange(0, 14, 1), 1)

def experiment(i):
    samples = surrounding[surrounding['vy_relative']>v_relative[i]-speed_interval/2]
    samples = samples[samples['vy_relative']<=v_relative[i]+speed_interval/2]
    subsample_size = int(np.round(len(samples) * 0.8)) # 80% of the samples

    estimates = np.zeros((20, 22))
    iter_exp = 0
    iter_count = 0
    while iter_count < 20:
        np.random.seed(iter_exp)
        sub_samples = samples.loc[np.random.choice(samples.index, subsample_size)]
        
        driver_space = ds.drspace(sub_samples, width=width, length=length)
        corrcoef_xy = driver_space.corrcoef_xy

        sys.stdout.write('---- relative velocity = ' + str(v_relative[i]) + ', iteration = ' + str(iter_exp) + ', iter_count = ' + str(iter_count) + ' ----\n')
        lower_bounds, estimation, stderror_beta, pvalue_beta = driver_space.Inference(max_iter=50, workers=20)

        pfailure = np.any(pvalue_beta>0.05)
        rxfailure = np.logical_or(estimation[0]==lower_bounds[0], estimation[1]==lower_bounds[0])
        ryfailure = np.logical_or(estimation[2]==lower_bounds[1], estimation[3]==lower_bounds[1])
        
        if ~np.any(np.array([pfailure, rxfailure, ryfailure])):
            estimates[iter_count] = np.concatenate(([iter_exp, v_relative[i], subsample_size, corrcoef_xy, lower_bounds[0], lower_bounds[1]],
                                                     estimation,
                                                     stderror_beta,
                                                     pvalue_beta))
            iter_count += 1
        
        iter_exp += 1

    return estimates

result = np.zeros((len(v_relative)*20, 22))
for i in range(len(v_relative)):
    result[i*20:(i+1)*20] = experiment(i)

    ## Save result
    result_tosave = pd.DataFrame(result, columns=['iter_exp',
                                                  'vy_relative',
                                                  'num_samples',
                                                  'corrcoef_xy',
                                                  'x_lower_bound',
                                                  'y_lower_bound',
                                                  'r_x_plus_hat',
                                                  'r_x_minus_hat',
                                                  'r_y_plus_hat',
                                                  'r_y_minus_hat',
                                                  'beta_x_plus_hat',
                                                  'beta_x_minus_hat',
                                                  'beta_y_plus_hat',
                                                  'beta_y_minus_hat',
                                                  'stderr_beta_x_plus',
                                                  'stderr_beta_x_minus',
                                                  'stderr_beta_y_plus',
                                                  'stderr_beta_y_minus',
                                                  'pval_beta_x_plus',
                                                  'pval_beta_x_minus',
                                                  'pval_beta_y_plus',
                                                  'pval_beta_y_minus'])

    result_tosave.to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Approximation/Bootstrapping_d6.csv')

sys.stdout.write('All results saved successfully\n')


# Exp3: Asymmetric driver space (corresponding to subsection 4.1 in our article)

for did in ['d6', 'd7']:

    sample_path = parent_dir+'/OutputData/DriverSpace/pNEUMA/SurroundingSampling/' + did

    for position in ['front', 'back']:

        all_files = glob.glob(sample_path + '/*.h5')
        surrounding = []
        for filename in all_files:
            df = pd.read_hdf(filename, key='sample')
            surrounding.append(df)
        surrounding = pd.concat(surrounding, axis=0, ignore_index=True)
        sys.stdout.write('---- Samples loaded successfully ----\n')

        # Position correction
        surrounding['relative_x'] = surrounding['relative_x'] / surrounding['width_j'] * width
        surrounding['relative_y'] = surrounding['relative_y'] / surrounding['length_j'] * length

        # Vehicles filtering
        if position=='front':
            surrounding = surrounding[(surrounding.gx_v_i*(surrounding.gx_O_j-surrounding.gx_O_i)+surrounding.gy_v_i*(surrounding.gy_O_j-surrounding.gy_O_i))/np.sqrt(surrounding.gx_v_i**2+surrounding.gy_v_i**2)/np.sqrt((surrounding.gx_O_j-surrounding.gx_O_i)**2+(surrounding.gy_O_j-surrounding.gy_O_i)**2)>0]
        elif position=='back':
            surrounding = surrounding[(surrounding.gx_v_i*(surrounding.gx_O_j-surrounding.gx_O_i)+surrounding.gy_v_i*(surrounding.gy_O_j-surrounding.gy_O_i))/np.sqrt(surrounding.gx_v_i**2+surrounding.gy_v_i**2)/np.sqrt((surrounding.gx_O_j-surrounding.gx_O_i)**2+(surrounding.gy_O_j-surrounding.gy_O_i)**2)<0]

        speed_interval = 0.2
        v_relative = np.round(np.arange(0, 13 + speed_interval, speed_interval), 1)

        def experiment(i):
            samples = surrounding[surrounding['vy_relative']>v_relative[i]-speed_interval/2]
            samples = samples[samples['vy_relative']<=v_relative[i]+speed_interval/2]
            
            driver_space = ds.drspace(samples, width=width, length=length)
            corrcoef_xy = driver_space.corrcoef_xy

            lower_bounds, estimation, stderror_beta, pvalue_beta = driver_space.Inference(max_iter=50, workers=20)
            
            estimates = np.concatenate(([v_relative[i], corrcoef_xy, lower_bounds[0], lower_bounds[1]],
                                        estimation,
                                        stderror_beta,
                                        pvalue_beta))

            return estimates

        result = np.zeros((len(v_relative), 20))
        for i in range(len(v_relative)):
            sys.stdout.write('---- relative speed = ' + str(v_relative[i]) + ', start estimation ----\n')
            result[i] = experiment(i)

            ## Save result
            result_tosave = pd.DataFrame(result, columns=['vy_relative',
                                                          'corrcoef_xy',
                                                          'x_lower_bound',
                                                          'y_lower_bound',
                                                          'r_x_plus_hat',
                                                          'r_x_minus_hat',
                                                          'r_y_plus_hat',
                                                          'r_y_minus_hat',
                                                          'beta_x_plus_hat',
                                                          'beta_x_minus_hat',
                                                          'beta_y_plus_hat',
                                                          'beta_y_minus_hat',
                                                          'stderr_beta_x_plus',
                                                          'stderr_beta_x_minus',
                                                          'stderr_beta_y_plus',
                                                          'stderr_beta_y_minus',
                                                          'pval_beta_x_plus',
                                                          'pval_beta_x_minus',
                                                          'pval_beta_y_plus',
                                                          'pval_beta_y_minus'])

            result_tosave.to_csv(parent_dir+'/OutputData/DriverSpace/pNEUMA/Approximation/' + position + '_' + did + '.csv')

    sys.stdout.write('All results saved successfully\n')
