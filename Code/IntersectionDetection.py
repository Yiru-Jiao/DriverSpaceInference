'''
Part I of intersection detection, the other part is in `IntersectionData.ipynb`

'''

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize

parent_dir = os.path.abspath('..') # Set your parent directory here. 
                                   # Without change the current setting is the parent directory of this file.
save_path = parent_dir + '/OutputData/DriverSpace/Intersection/trajectories/'

def read_data(dx):

    files = sorted(glob.glob(parent_dir + '/InputData/pNEUMA/' + dx + '/*.h5'))
    data = []
    for file in files[3:4]:
        df = pd.read_hdf(file, key='data')
        df = df.reset_index()
        data.append(df)
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    data = data[(data.agent_type!='Motorcycle')&(data.agent_type!='Undefined')]
    selected_frames = data['frame_id'].drop_duplicates().values
    selected_frames = selected_frames[np.arange(5,len(selected_frames),25)]
    data = data[data.frame_id.isin(selected_frames)]
    data['psi_kf'] = (data['psi_kf'] + np.pi)%(2*np.pi)-np.pi
    data = data.sort_values(['track_id', 'frame_id'])
    data['hx'] = np.cos(data['psi_kf'])
    data['hy'] = np.sin(data['psi_kf'])

    return data


def simplify_trajectory(track_id, trajectories):
    veh = trajectories[trajectories.track_id == track_id]
    newveh = [veh.iloc[0]]
    for i in range(1, len(veh)):
        if np.linalg.norm(newveh[-1][['x','y']] - veh.iloc[i][['x','y']]) > 3:
            newveh.append(veh.iloc[i])
    newveh = pd.DataFrame(newveh)[['track_id','frame_id','x','y','hx','hy','psi_kf']]
    return newveh


def balance(lateral_movement, lateral_dist, costheta):
    sigma = 2.
    var = lateral_dist - lateral_movement
    probabilities = 1/sigma/np.sqrt(2*np.pi)*np.exp(-var**2/2/sigma**2)
    # F1 = np.sum(probabilities*var*costheta)/np.sum(probabilities)
    # F2 = lateral_movement
    return (np.sum(probabilities*var*costheta)/np.sum(probabilities) - lateral_movement)**2


def dist_p2l(point, line_start, line_end):
    return np.absolute((line_end[0]-line_start[0])*(line_start[1]-point[1])-(line_start[0]-point[0])*(line_end[1]-line_start[1]))/np.sqrt((line_end[0]-line_start[0])**2+(line_end[1]-line_start[1])**2)


def update_trajectory(idx, toupdate_trajectories):
    x,y,hx,hy = toupdate_trajectories.loc[idx][['x','y','hx','hy']].values
    radius = 15
    points = toupdate_trajectories[((toupdate_trajectories['x']-x)**2+(toupdate_trajectories['y']-y)**2)<radius**2]
    if len(points)<2:
        return (x,y)
    else:
        # d1 = np.sqrt((points['x']-x)**2+(points['y']-y)**2)
        d2 = dist_p2l(points[['x','y']].values.T, [x+radius*(-hy),y+radius*hx], [x+radius*hy, y+radius*(-hx)])
        lateral_dist = (points['x']-x)*hy+(points['y']-y)*(-hx)
        points = points[d2<5]
        lateral_dist = lateral_dist[d2<5]
        costheta = (hx*points['hx']+hy*points['hy'])
        costheta[(costheta<0)&(lateral_dist>0)] = 0
        res = minimize(balance, 
                       x0=0., 
                       args=(lateral_dist, costheta), 
                       method='BFGS',)
        result = np.array([res.x]).reshape(-1)[0]
        return (x + result*hy, y + result*(-hx))


def compute_G(idx, updated_trajectories):
    radius = 8.
    x,y = updated_trajectories[['x','y']].loc[idx]
    neighbours = updated_trajectories[((updated_trajectories['x']-x)**2+(updated_trajectories['y']-y)**2)<radius**2]
    neighbours = neighbours[neighbours.index!=idx]
    if len(neighbours)<10:
        G_star = 0.
    else:
        weights = np.sqrt((neighbours['x']-x)**2+(neighbours['y']-y)**2)
        weights = 2*weights.max() - weights
        weights = weights/np.sum(weights)
        angles = abs(neighbours['psi_kf'])
        angles[angles>np.pi/2] = np.pi - angles[angles>np.pi/2]
        x_bar = np.mean(angles)
        S = np.sqrt(abs(np.mean(angles**2)-x_bar**2))
        G_star = (np.sum(weights*angles) - x_bar*np.sum(weights))/(S*np.sqrt(abs((len(angles)*np.sum(weights**2)-np.sum(weights)**2)/(len(angles)-1))))
    return G_star


# Main
intersection_list = []
for dx in ['d'+str(did+1) for did in range(10)]:
    print('Reading data', dx)
    trajectories = read_data(dx)

    # Simplify trajectories by removing points that are too close to each other

    print('Simplifying trajectories...')
    simplified_trajectories = pd.concat(Parallel(n_jobs=15)(delayed(simplify_trajectory)(track_id, trajectories) for track_id in tqdm(trajectories.track_id.unique())))
    simplified_trajectories = simplified_trajectories.reset_index(drop=True)
    ## Save simplified trajectories
    simplified_trajectories.to_hdf(save_path+'simplified_trajectories_'+dx+'.h5', key='data', mode='w')

    # Clustering trajectories by the method of Cao and Krumm (2010)

    print('Clustering trajectories...')
    toupdate_trajectories = simplified_trajectories.copy()
    loop_count = 0
    displacement = 1.
    while loop_count<5 and displacement>0.2:
        newxy = Parallel(n_jobs=25)(delayed(update_trajectory)(idx, toupdate_trajectories) for idx in tqdm(toupdate_trajectories.index))
        newxs = [xy[0] for xy in newxy]
        newys = [xy[1] for xy in newxy]
        updated_trajectories = toupdate_trajectories.copy()
        updated_trajectories['x'] = newxs
        updated_trajectories['y'] = newys
        traj_diff = updated_trajectories[['x','y']] - toupdate_trajectories[['x','y']]
        displacement = np.mean(np.sqrt(traj_diff['x']**2+traj_diff['y']**2))
        print('Loop', loop_count, 'displacement', displacement)
        toupdate_trajectories = updated_trajectories.copy()
        loop_count += 1

    # Registring intersection points to road intersections by hotspot analysis

    print('Computing G...')
    G_stars = Parallel(n_jobs=15)(delayed(compute_G)(idx, updated_trajectories) for idx in tqdm(updated_trajectories.index))
    updated_trajectories['G_star'] = np.array(G_stars).astype(float)
    ## Save updated trajectories
    updated_trajectories.to_hdf(save_path + dx + '_roads.h5', key='data', mode='w')
