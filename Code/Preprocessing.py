'''
Preprocessing of the pNEUMA dataset
(Section 3.1 Preprocessing)

'''

import os
import csv
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed

parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.

## 1.1 pNEUMA data extraction
def pNEUMA(open_path, data_title):
    data_path = open_path + '/' + data_title + '.csv'
    data_overview_cols = ['track_id', 'type', 'traveled_d', 'avg_speed']
    data_cols = ['lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time', 'track_id'] # check the original csv to ensure the order of the variables is right
    
    data_overview_colsize = len(data_overview_cols)
    data_colsize = len(data_cols)-1
    data_overview_rows = []
    data_rows = []
    
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)
        data_overview = dict()
        for row in csv_reader:
            row = [item.strip() for item in row]
            track_id = int(row[0])
            data_overview_rows.append(row[:data_overview_colsize])
            
            data_traj = [float(item) for item in row[data_overview_colsize:] if len(item)>0]
            for i in range(0, len(data_traj), data_colsize):
                data_row = data_traj[i:i+data_colsize] + [track_id]
                data_rows.append(data_row)
                
    data_overview = pd.DataFrame(data_overview_rows, columns=data_overview_cols)
    data_overview['track_id'] = data_overview['track_id'].astype(int)
    data_overview = data_overview.set_index('track_id')

    #//Complement data of vehicle size (provided by the creator of pNEUMA)
    length = np.zeros(data_overview.shape[0])
    width = np.zeros(data_overview.shape[0])
    for i in range(data_overview.shape[0]):
        vehicle_type = data_overview['type'].iloc[i]
        if vehicle_type == 'Motorcycle':
            length[i] = 2.5
            width[i] = 1
        elif np.logical_or(vehicle_type == 'Car', vehicle_type == 'Taxi'):
            length[i] = 5
            width[i] =2
        elif vehicle_type == 'Medium Vehicle':
            length[i] = 5.83
            width[i] = 2.67
        elif vehicle_type == 'Heavy Vehicle':
            length[i] = 12.5
            width[i] = 3.33
        elif vehicle_type == 'Bus':
            length[i] = 12.5
            width[i] = 4
    data_overview['length'] = length
    data_overview['width'] = width
    
    data = pd.DataFrame(data_rows, columns=data_cols)
    #//Transform timestamps into integers
    time_interval = data['time'].iloc[1]-data['time'].iloc[0] #//time intervals in pNEUMA is not always 0.04
    data['frame_id'] = round(data['time']/time_interval).astype(int)
    data.set_index(['track_id', 'frame_id'], drop=True, inplace=True)
    #//Change the unit of speed from km/h to m/s
    data_overview[['traveled_d', 'avg_speed']] = data_overview[['traveled_d', 'avg_speed']].astype(float)
    data_overview['avg_speed'] = data_overview['avg_speed'] / 3.6
    data['speed'] = data['speed'] / 3.6
    
    #//Geographical coordinates
    utm_crs_list = query_utm_crs_info(datum_name='WGS84',
                                      area_of_interest=AreaOfInterest(
                                      west_lon_degree=np.floor(data['lon'].min()), 
                                      south_lat_degree=np.floor(data['lat'].min()), 
                                      east_lon_degree=np.ceil(data['lon'].max()), 
                                      north_lat_degree=np.ceil(data['lat'].max())),)
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    crs=CRS.from_epsg(4326)
    geo_coordinates = Transformer.from_crs(crs.geodetic_crs, utm_crs)
    x, y = geo_coordinates.transform(data['lat'].values, data['lon'].values)
    data['x'] = x - 739000
    data['y'] = y - 4206500

    return data_overview, data

## 1.2 Kalman Filter
def KFV(track_id):
    veh = data.loc[track_id].copy()

    # Extended Kalman Filter for Constant Heading and Velocity
    ## Initialize
    numstates = 4
    P = np.eye(numstates)*50
    dt = np.diff(veh.time)
    dt = np.hstack([dt[0],dt])
    R = np.diag([3.,3.,2])
    I = np.eye(numstates)
    mx = veh.x.values
    my = veh.y.values
    mv = veh.speed.values
    dx = np.diff(veh.x)
    dx = np.hstack([0,dx])
    dy = np.diff(veh.y)
    dy = np.hstack([0,dy])
    ds = np.sqrt(dx**2+dy**2)
    GPS=(ds!=0.0).astype('bool') # GPS Trigger for Kalman Filter

    # Measurement vector
    measurements = np.vstack((mx, my, mv))
    m = measurements.shape[1] 

    head = veh[['x','y']].diff(10)
    head = head[~np.isnan(head.x)]
    head = head[(head.x!=0)|(head.y!=0)].values
    if len(head)==0:
        estimates = np.zeros((m,4))*np.nan
    else:
        psi0 = np.arctan2(head[0][1], head[0][0])
        x = np.matrix([[mx[0], my[0], mv[0], psi0]], dtype=float).T

        # Estimated vector
        estimates = np.zeros((m,4))

        for filterstep in range(m):
            # Time Update (Prediction)
            # ========================
            # Project the state ahead
            # see "Dynamic Matrix"

            x[0] = x[0] + dt[filterstep]*x[2]*np.cos(x[3])
            x[1] = x[1] + dt[filterstep]*x[2]*np.sin(x[3])
            x[2] = x[2]
            x[3] = (x[3]+ np.pi) % (2.0*np.pi) - np.pi

            # Calculate the Jacobian of the Dynamic Matrix A
            # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
            a13 = dt[filterstep]*np.cos(x[3])
            a14 = -dt[filterstep]*x[2]*np.sin(x[3])
            a23 = dt[filterstep]*np.sin(x[3])
            a24 = dt[filterstep]*x[2]*np.cos(x[3])
            JA = np.matrix([[1.0, 0.0, a13, a14],
                            [0.0, 1.0, a23, a24],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], dtype=float)

            # Calculate the Process Noise Covariance Matrix
            sGPS     = 4*dt[filterstep]**2
            sCourse  = 1.0*dt[filterstep]
            sVelocity= 18.0*dt[filterstep]

            Q = np.diag([sGPS**2, sGPS**2, sVelocity**2, sCourse**2])

            # Project the error covariance ahead
            P = JA*P*JA.T + Q

            # Measurement Update (Correction)
            # ===============================
            # Measurement Function
            hx = np.matrix([[float(x[0])],
                            [float(x[1])],
                            [float(x[2])]], dtype=float)

            if GPS[filterstep]:
                JH = np.matrix([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0]], dtype=float)
            else:
                JH = np.matrix([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]], dtype=float)        

            S = JH*P*JH.T + R
            K = (P*JH.T) * np.linalg.inv(S.astype('float'))

            # Update the estimate via
            Z = measurements[:,filterstep].reshape(JH.shape[0],1)
            y = Z - (hx)                         # Innovation or Residual
            x = x + (K*y)

            # Update the error covariance
            P = (I - (K*JH))*P

            # Save states for Plotting
            x = x.astype('float')
            estimates[filterstep,:] = np.array(x).reshape(-1)

    veh[['x_kf','y_kf','speed_kf','psi_kf']] = estimates
    veh['track_id'] = track_id
    return veh.reset_index()

## 1.3 Zero heading check
def zeroheading(df):
    if np.all(np.logical_and(abs(df['vx_kf'].iloc[5:])<=0.5, abs(df['vy_kf'].iloc[5:])<=0.5)):
        return True
    else:
        return False

## 1.4 Data cleaning
def cleaning(data):

    ##//remove vehicles that never change location
    remove_list = data.groupby('track_id').apply(zeroheading)
    remove_list = remove_list.index[remove_list]
    data = data.set_index(['track_id','frame_id'])
    data = data.drop(remove_list, level=0)
    print('There are ' + str(len(remove_list)) + ' parking vehicles')
    
    # ##//remove vehicles overlapping(<0.5m) another vehicle's trajectory, which is caused by computer vision inaccuracy
    # ##//with these code commented, overlapping vehicles will be removed during sampling
    # data_round = data.copy()
    # data_round['x']=np.round(data['x']*2,0)/2
    # data_round['y']=np.round(data['y']*2,0)/2
    # duplicates = data_round[data_round[['time', 'x', 'y']].duplicated(keep='first')]
    
    # remove_list = duplicates.index.get_level_values(0).drop_duplicates()
    # data = data.drop(remove_list, level=0)
    
    return data, remove_list

# 1.5 Transform
for dx in ['d'+str(did+1) for did in range(10)]:
    data_files = sorted(glob.glob(parent_dir + '/RawDatasets/pNEUMA/' + dx + '/*.csv'))
    open_path = parent_dir+'/RawDatasets/pNEUMA/'+dx
    save_path = parent_dir+'/InputData/pNEUMA/'+dx
    
    if dx!='d10':
        data_titles = [data_file[-25:-4] for data_file in data_files]
    else:
        data_titles = [data_file[-26:-4] for data_file in data_files]

    for data_title in data_titles:
        print('---- Preprocessing ' + data_title + ' ----')

        data_overview, data = pNEUMA(open_path, data_title)
        track_ids = data.reset_index().groupby('track_id').frame_id.count()
        track_ids = track_ids[track_ids>=25] # remove vehicles appearing less than 1 second
        data = data.loc[track_ids.index]
        track_ids = track_ids.index.values
        data = pd.concat(Parallel(n_jobs=25)(delayed(KFV)(id) for id in tqdm(track_ids)))
        data['vx_kf'] = data.speed_kf*np.cos(data.psi_kf)
        data['vy_kf'] = data.speed_kf*np.sin(data.psi_kf)

        data, nevermove = cleaning(data)

        if len(nevermove)>0:
            pd.DataFrame(nevermove).to_csv(save_path + '/nevermove_' + data_title + '.csv')
        
        data[['length','width']] = data_overview.reindex(index=data.index.get_level_values(0))[['length','width']].values
        data['agent_type'] = data_overview.reindex(index=data.index.get_level_values(0))['type'].values
        
        data.reset_index().to_hdf(save_path + '/data_' + data_title + '.h5', key='data')
