import sys
import csv
import random
import numpy as np
import pandas as pd
from itertools import permutations
from itertools import combinations
from scipy.misc import derivative
from scipy import optimize
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


# 1 Coordinate transformation
# This function is used to transform geographical coordinates of a pair of vehicles into local coordinates between them.
# Please refer to subsubsection 2.1.1 in our article 
def geo2vehicle(x, y, gx_O, gy_O, gx_v_i, gy_v_i, gx_v_j, gy_v_j):
    a = -gx_O
    b = -gy_O
    gx_v_ij = gx_v_i - gx_v_j
    gy_v_ij = gy_v_i - gy_v_j
    cos = gy_v_ij / np.sqrt(gx_v_ij**2 + gy_v_ij**2)
    sin = gx_v_ij / np.sqrt(gx_v_ij**2 + gy_v_ij**2)
    
    transformed_x = cos*(x+a)-sin*(y+b)
    transformed_y = sin*(x+a)+cos*(y+b)

    return transformed_x, transformed_y


# 2 Preprocessing
# Please refer to subsection 3.1 in our article

class preprocess:
    """This class is used to transform raw datasets"""

    def __init__(self, dataset, open_path, data_title, head_period):
        self.dataset = dataset
        self.open_path = open_path
        self.data_title = data_title
        self.head_period = head_period
    
    ## 2.1 pNEUMA data extraction
    def pNEUMA(self,):
        data_path = self.open_path + '/' + self.data_title + '.csv'
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
        E, N = geo_coordinates.transform(data['lat'].values, data['lon'].values)
        data['E'] = E
        data['N'] = N

        #//Assign data
        self.data_overview = data_overview
        self.data = data

    ## 2.2 Heading direction
    def heading(self, df):
        #//If the existing time of the vehicle is less than head_period, set the heading direction according to the location change directly
        if len(df) <= self.head_period:
            Headings = pd.DataFrame([[df['E'].values[-1] - df['E'].values[0], df['N'].values[-1] - df['N'].values[0]],] * len(df), index=df.frame_id, columns=['E_heading','N_heading'])
        else:
            Headings = pd.DataFrame(np.zeros((len(df),2)), index=df.frame_id, columns=['E_heading','N_heading'])  
            Headings.iloc[:-self.head_period] = df[['E','N']].iloc[self.head_period:].values - df[['E','N']].iloc[:-self.head_period].values
            
            data_heading0 = df[np.logical_and(Headings.E_heading.values==0, Headings.N_heading.values==0)]

            if len(data_heading0) != 0 and len(data_heading0) != len(df):
                #//Complement the nearest non-zero heading direction for stationary moments
                frames = data_heading0.frame_id.drop_duplicates().values
                starts = np.array([0])
                ends = np.array([])

                for count in range(len(frames)):
                    if not frames[count] - frames[starts[-1]] == count - starts[-1]:
                        ends = np.append(ends, count - 1).astype(int)
                        starts = np.append(starts, count)
                ends = np.append(ends, count).astype(int)

                if frames[starts[0]] == df.frame_id.min():
                    arg_nearest_nonzero = np.where(np.logical_or(Headings.E_heading!=0, Headings.N_heading!=0))[0].min()
                    Headings.loc[frames[starts[0]]] = Headings.iloc[arg_nearest_nonzero][['E_heading', 'N_heading']]
                    if ends[0]==starts[0]:
                        starts = np.delete(starts, 0)
                        ends = np.delete(ends, 0)
                    else:
                        starts[0] = 1

                for j in range(starts.shape[0]):                        
                    Headings.loc[frames[starts[j]]:frames[ends[j]]] = np.array([Headings.loc[frames[starts[j]]-1].values,] * (ends[j] + 1 - starts[j]))
        return Headings

    ## 2.3 Zero heading check
    def zeroheading(self, df):
        if np.all(np.logical_and(df['E_heading']==0, df['N_heading']==0)):
            return True
        else:
            return False
    
    ## 2.4 Data cleaning
    def cleaning(self,):
        ##//remove vehicles that never change location
        remove_list = self.data.reset_index().groupby('track_id').apply(self.zeroheading)
        remove_list = self.data.index.get_level_values(0).drop_duplicates()[remove_list]

        self.data = self.data.drop(remove_list, level=0)
        self.data_overview = self.data_overview.drop(remove_list)

        ##//remove vehicles overlapping(<0.5m) another vehicle's trajectory, which is caused by computer vision inaccuracy
        data_round = self.data.copy()
        data_round['E']=np.round(self.data['E']*2,0)/2
        data_round['N']=np.round(self.data['N']*2,0)/2
        duplicates = data_round[data_round[['time', 'E', 'N']].duplicated(keep='first')]
        
        remove_list = duplicates.index.get_level_values(0).drop_duplicates()
        self.data = self.data.drop(remove_list, level=0)
        self.data_overview = self.data_overview.drop(remove_list)

        ##//remove no-more-existing vehicles in Data
        non_existing = self.data_overview.index.get_level_values(0).drop_duplicates()[np.logical_not(np.isin(self.data_overview.index.get_level_values(0).drop_duplicates(), self.data.index.get_level_values(0).drop_duplicates()))]
        self.data_overview = self.data_overview.drop(non_existing)

    ## 2.5 Transform
    def transform(self, save_path):
        self.pNEUMA()
        
        sys.stdout.write('---- heading direction ----\n')
        self.data = self.data.join(self.data.reset_index().groupby('track_id').apply(self.heading))
        
        sys.stdout.write('---- data cleaning ----\n')
        self.cleaning()

        self.data['vx'] = self.data['speed'] * self.data['E_heading'] / np.sqrt(self.data['E_heading']**2 + self.data['N_heading']**2)
        self.data['vy'] = self.data['speed'] * self.data['N_heading'] / np.sqrt(self.data['E_heading']**2 + self.data['N_heading']**2)
        self.data[['length','width']] = self.data_overview.reindex(index=self.data.index.get_level_values(0))[['length','width']].values
        self.data['agent_type'] = self.data_overview.reindex(index=self.data.index.get_level_values(0))['type'].values

        self.data_overview.to_csv(save_path + '/data_overview_' + self.data_title + '.csv')
        self.data.to_hdf(save_path + '/data_' + self.data_title + '.h5', key='data')
        sys.stdout.write('---- transformed data saved sucessfully ----\n')


# 3 Sampling
# Please refer to subsection 3.1 in our article

class sample:
    """This class is used to randomly select samples that could be used to approximate driver space."""

    def __init__(self, dataset, data, data_overview, data_title):
        
        self.dataset = dataset
        self.data = data
        self.data_overview = data_overview        
        self.data_title = data_title

    def sampling(self, save_path, frame_interval):
        self.data = self.data.reset_index()
        frames = self.data['frame_id'].drop_duplicates().values
        frame_ids = frames[random.randint(0, frame_interval-1) + np.array(range(len(frames) // frame_interval)) * frame_interval]
        self.data = self.data.loc[np.isin(self.data['frame_id'], frame_ids)]
        
        pair_idx = self.data.groupby('frame_id').apply(lambda x : pd.DataFrame.from_records(combinations(x.track_id, 2)))
        pairs = pd.DataFrame({'frame_id':pair_idx.index.get_level_values(0).values, 'i':pair_idx[0].values, 'j':pair_idx[1].values}, dtype='int32')
        pair_idx = []
        sys.stdout.write('---- pair_idx released ----\n')

        self.data = self.data.set_index(['track_id','frame_id'])
        
        pairs[['gx_O_i','gy_O_i','gx_v_i','gy_v_i','length_i','width_i','speed_i']] = self.data.reindex(index=pairs.set_index(['i','frame_id']).index)[['E','N','vx','vy','length','width','speed']].values
        pairs[['gx_O_j','gy_O_j','gx_v_j','gy_v_j','length_j','width_j','speed_j']] = self.data.reindex(index=pairs.set_index(['j','frame_id']).index)[['E','N','vx','vy','length','width','speed']].values

        self.surrounding = pairs[np.logical_or(pairs['gx_v_i'] - pairs['gx_v_j'] != 0, pairs['gy_v_i'] - pairs['gy_v_j'] != 0)] # proximity resistance cannot be computed for two vehicles with a relative velocity of (0,0)
        pairs = []
        sys.stdout.write('---- pairs released ----\n')

        # transformed_xy = self.surrounding.apply(lambda x: geo2vehicle(x.gx_O_j, x.gy_O_j, x.gx_O_i, x.gy_O_i, x.gx_v_i, x.gy_v_i, x.gx_v_j, x.gy_v_j), axis=1)
        # self.surrounding[['relative_x', 'relative_y']] = [list(xy) for xy in transformed_xy]

        self.surrounding['relative_x'] = (self.surrounding.gy_v_i - self.surrounding.gy_v_j) / np.sqrt((self.surrounding.gx_v_i - self.surrounding.gx_v_j)**2 + (self.surrounding.gy_v_i - self.surrounding.gy_v_j)**2) * (self.surrounding.gx_O_j - self.surrounding.gx_O_i) - (self.surrounding.gx_v_i - self.surrounding.gx_v_j) / np.sqrt((self.surrounding.gx_v_i - self.surrounding.gx_v_j)**2 + (self.surrounding.gy_v_i - self.surrounding.gy_v_j)**2) * (self.surrounding.gy_O_j - self.surrounding.gy_O_i)
        self.surrounding['relative_y'] = (self.surrounding.gx_v_i - self.surrounding.gx_v_j) / np.sqrt((self.surrounding.gx_v_i - self.surrounding.gx_v_j)**2 + (self.surrounding.gy_v_i - self.surrounding.gy_v_j)**2) * (self.surrounding.gx_O_j - self.surrounding.gx_O_i) + (self.surrounding.gy_v_i - self.surrounding.gy_v_j) / np.sqrt((self.surrounding.gx_v_i - self.surrounding.gx_v_j)**2 + (self.surrounding.gy_v_i - self.surrounding.gy_v_j)**2) * (self.surrounding.gy_O_j - self.surrounding.gy_O_i)
        self.surrounding['vy_relative'] = np.sqrt((self.surrounding['gx_v_i'] - self.surrounding['gx_v_j'])**2 + (self.surrounding['gy_v_i'] - self.surrounding['gy_v_j'])**2)
        
        sys.stdout.write('---- sampling completed ----\n')

        self.surrounding = self.surrounding[np.logical_and(np.absolute(self.surrounding['relative_x'])<50, np.absolute(self.surrounding['relative_y'])<100)]

        self.surrounding.to_hdf(save_path + '/surrounding_' + self.data_title + '.h5', key='sample')
        sys.stdout.write('---- sampling data saved successfully ----\n')


# 4 Parameter estimation
# Please refer to subsection 2.3 in our article

class drspace:
    """The class is used to estimate parameters of driver space from samples."""

    def __init__(self, samples, eps=1e-4, width=2, length=5, percentile=0.5):

        #//Extract input variables
        self.width = width
        self.x = samples['relative_x'].values
        self.y = samples['relative_y'].values
        self.v_y = samples['vy_relative'].values
        self.corrcoef_xy = np.corrcoef(self.x, self.y)[1,0]

        #//Initialize parameters
        self.epsilon = eps
        self.r_x_plus = np.percentile(self.x[np.logical_and(self.x>0, np.absolute(self.y)<width/2)], percentile)
        self.r_x_minus = np.percentile(-self.x[np.logical_and(self.x<0, np.absolute(self.y)<width/2)], percentile)
        self.r_y_plus = np.percentile(self.y[np.logical_and(self.y>0, np.absolute(self.x)<width/2)], percentile)
        self.r_y_minus = np.percentile(-self.y[np.logical_and(self.y<0, np.absolute(self.x)<width/2)], percentile)
        self.beta_x_plus = 2
        self.beta_x_minus = 2
        self.beta_y_plus = 2
        self.beta_y_minus = 2
    
    ## 4.1 Log likelihood

    ### 4.1.1 Log likelihood to all parameters

    def LogL(self, para):
        
        r_x_plus, r_x_minus, r_y_plus, r_y_minus, beta_x_plus, beta_x_minus, beta_y_plus, beta_y_minus = para

        r_x = (1 + np.sign(self.x)) / 2 * r_x_plus + (1 - np.sign(self.x)) / 2 * r_x_minus
        beta_x = (1 + np.sign(self.x)) / 2 * beta_x_plus + (1 - np.sign(self.x)) / 2 * beta_x_minus
        r_y = (1 + np.sign(self.y)) / 2 * r_y_plus + (1 - np.sign(self.y)) / 2 * r_y_minus
        beta_y = (1 + np.sign(self.y)) / 2 * beta_y_plus + (1 - np.sign(self.y)) / 2 * beta_y_minus

        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / r_x), beta_x) - np.power(np.absolute(self.y / r_y), beta_y)))

        return sum(elements)

    ### 4.1.2 Log likelihood to r_x_plus

    def LogL_rx_plus(self, r_x_plus):
        
        r_x = (1 + np.sign(self.x)) / 2 * r_x_plus + (1 - np.sign(self.x)) / 2 * self.r_x_minus
        beta_x = (1 + np.sign(self.x)) / 2 * self.beta_x_plus + (1 - np.sign(self.x)) / 2 * self.beta_x_minus
        r_y = (1 + np.sign(self.y)) / 2 * self.r_y_plus + (1 - np.sign(self.y)) / 2 * self.r_y_minus
        beta_y = (1 + np.sign(self.y)) / 2 * self.beta_y_plus + (1 - np.sign(self.y)) / 2 * self.beta_y_minus

        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / r_x), beta_x) - np.power(np.absolute(self.y / r_y), beta_y)))

        return sum(elements)

    ### 4.1.3 Second-order derivative of the Log likelihood to r_x_plus
    
    def LogL_rx_plus_prime2(self, r_x_plus): 
        return derivative(self.LogL_rx_plus, r_x_plus, dx=0.5, n=2, order=3)

    ### 4.1.4 Log likelihood to r_x_minus

    def LogL_rx_minus(self, r_x_minus):
        r_x = (1 + np.sign(self.x)) / 2 * self.r_x_plus + (1 - np.sign(self.x)) / 2 * r_x_minus
        beta_x = (1 + np.sign(self.x)) / 2 * self.beta_x_plus + (1 - np.sign(self.x)) / 2 * self.beta_x_minus
        r_y = (1 + np.sign(self.y)) / 2 * self.r_y_plus + (1 - np.sign(self.y)) / 2 * self.r_y_minus
        beta_y = (1 + np.sign(self.y)) / 2 * self.beta_y_plus + (1 - np.sign(self.y)) / 2 * self.beta_y_minus

        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / r_x), beta_x) - np.power(np.absolute(self.y / r_y), beta_y)))

        return sum(elements)

    ### 4.1.5 Second-order derivative of the Log likelihood to r_x_minus

    def LogL_rx_minus_prime2(self, r_x_minus):
        return derivative(self.LogL_rx_minus, r_x_minus, dx=0.5, n=2, order=3)

    ### 4.1.6 Log likelihood to r_y_plus

    def LogL_ry_plus(self, r_y_plus):
        r_x = (1 + np.sign(self.x)) / 2 * self.r_x_plus + (1 - np.sign(self.x)) / 2 * self.r_x_minus
        beta_x = (1 + np.sign(self.x)) / 2 * self.beta_x_plus + (1 - np.sign(self.x)) / 2 * self.beta_x_minus
        r_y = (1 + np.sign(self.y)) / 2 * r_y_plus + (1 - np.sign(self.y)) / 2 * self.r_y_minus
        beta_y = (1 + np.sign(self.y)) / 2 * self.beta_y_plus + (1 - np.sign(self.y)) / 2 * self.beta_y_minus
    
        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / r_x), beta_x) - np.power(np.absolute(self.y / r_y), beta_y)))

        return sum(elements)

    ### 4.1.7 Second-order derivative of the Log likelihood to r_y_plus

    def LogL_ry_plus_prime2(self, r_y_plus):
        return derivative(self.LogL_ry_plus, r_y_plus, dx=0.5, n=2, order=3)
    
    ### 4.1.8 Log likelihood to r_y_minus

    def LogL_ry_minus(self, r_y_minus):
        r_x = (1 + np.sign(self.x)) / 2 * self.r_x_plus + (1 - np.sign(self.x)) / 2 * self.r_x_minus
        beta_x = (1 + np.sign(self.x)) / 2 * self.beta_x_plus + (1 - np.sign(self.x)) / 2 * self.beta_x_minus
        r_y = (1 + np.sign(self.y)) / 2 * self.r_y_plus + (1 - np.sign(self.y)) / 2 * r_y_minus
        beta_y = (1 + np.sign(self.y)) / 2 * self.beta_y_plus + (1 - np.sign(self.y)) / 2 * self.beta_y_minus
    
        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / r_x), beta_x) - np.power(np.absolute(self.y / r_y), beta_y)))

        return sum(elements)

    ### 4.1.9 Second-order derivative of the Log likelihood to r_y_minus

    def LogL_ry_minus_prime2(self, r_y_minus):
        return derivative(self.LogL_ry_minus, r_y_minus, dx=0.5, n=2, order=3)

    ### 4.1.10 Negative Log likelihood to betas

    def NLogL_betas(self, para):
    
        beta_x_plus, beta_x_minus, beta_y_plus, beta_y_minus = para
        
        r_x = (1 + np.sign(self.x)) / 2 * self.r_x_plus + (1 - np.sign(self.x)) / 2 * self.r_x_minus
        beta_x = (1 + np.sign(self.x)) / 2 * beta_x_plus + (1 - np.sign(self.x)) / 2 * beta_x_minus
        r_y = (1 + np.sign(self.y)) / 2 * self.r_y_plus + (1 - np.sign(self.y)) / 2 * self.r_y_minus
        beta_y = (1 + np.sign(self.y)) / 2 * beta_y_plus + (1 - np.sign(self.y)) / 2 * beta_y_minus
    
        elements = np.log(1+self.epsilon-np.exp(-np.power(np.absolute(self.x / r_x), beta_x) - np.power(np.absolute(self.y / r_y), beta_y)))

        return -sum(elements)

    ## 4.2 Optimize betas
    
    def optimize_betas(self, initial_guess):    
        betas = optimize.minimize(self.NLogL_betas,
                                  x0=initial_guess,
                                  method='L-BFGS-B',
                                  bounds=((2,np.inf),(2,np.inf),(2,np.inf),(2,np.inf)),
                                  jac=None,
                                  tol=None,
                                  options={'disp': None, 
                                           'maxcor': 10, 
                                           'ftol': 2.220446049250313e-09, 
                                           'gtol': 1e-05, 
                                           'eps': 1e-08, 
                                           'maxfun': 15000, 
                                           'maxiter': 15000, 
                                           'iprint': - 1,
                                           'maxls': 25,
                                           'finite_diff_rel_step': None})
        return betas

    ## 4.3 Inference

    def Inference(self, max_iter=100, workers=1):
 
        parameters = np.zeros((2, 8))
        stderr_betas = np.zeros((0,4))
        pvalue_betas = np.zeros((0,4))
        parameters[1, :] = [self.r_x_plus, self.r_x_minus, self.r_y_plus, self.r_y_minus, self.beta_x_plus, self.beta_x_minus, self.beta_y_plus, self.beta_y_minus]
        
        relative_speed = self.v_y.mean()
        x_llimit = self.width/2
        x_ulimit = 15
        y_llmit = np.round(0.1 * relative_speed**2 + 0.2 * relative_speed + 1.5, 1)
        y_ulimit = np.round(0.3 * relative_speed**2 + 1.5 * relative_speed + 10, 1)
        
        iteration = 0
        while np.any(parameters[-1] != parameters[-2]):

            #//Stop when reaching the maximum number of iterations
            if iteration > max_iter:
                parameters = np.vstack((parameters, np.ones((1,8))*np.nan))
                stderr_betas = np.vstack((stderr_betas, np.ones((1,4))*np.nan))
                pvalue_betas = np.vstack((pvalue_betas, np.ones((1,4))*np.nan))
                warnings.warn('Max. iterations reached', RuntimeWarning)
                break
            
            #//Stop when several sets of values repeat and select the set of values with the largest sum of rs
            array2check = np.all(np.equal(np.array([parameters[-1],]*len(parameters[:-1])), parameters[:-1]), axis=1)
            if np.any(array2check):
                repeated_alternatives = parameters[:-1][array2check]
                idx_chosen = np.argmax(repeated_alternatives[:,0]+repeated_alternatives[:,1]+repeated_alternatives[:,2]+repeated_alternatives[:,3])
                parameters = np.vstack((parameters, repeated_alternatives[idx_chosen]))
                stderr_betas = np.vstack((stderr_betas, stderr_betas[idx_chosen-len(repeated_alternatives)-1]))
                pvalue_betas = np.vstack((pvalue_betas, pvalue_betas[idx_chosen-len(repeated_alternatives)-1]))
                warnings.warn('Repetitive alternatives; the alternative with largest sum of rs is chosen', RuntimeWarning)
                break

            r_x_plus = optimize.brute(self.LogL_rx_plus_prime2, (slice(x_llimit, x_ulimit, 0.1),), finish=None, workers=workers)
            r_x_minus = optimize.brute(self.LogL_rx_minus_prime2, (slice(x_llimit, x_ulimit, 0.1),), finish=None, workers=workers)
            r_y_plus = optimize.brute(self.LogL_ry_plus_prime2, (slice(y_llmit, y_ulimit, 0.1),), finish=None, workers=workers)
            r_y_minus = optimize.brute(self.LogL_ry_minus_prime2, (slice(y_llmit, y_ulimit, 0.1),), finish=None, workers=workers)
            self.r_x_plus, self.r_x_minus, self.r_y_plus, self.r_y_minus = [r_x_plus, r_x_minus, r_y_plus, r_y_minus]

            betas = self.optimize_betas([self.beta_x_plus, self.beta_x_minus, self.beta_y_plus, self.beta_y_minus])
            self.beta_x_plus, self.beta_x_minus, self.beta_y_plus, self.beta_y_minus = betas.x
            
            stderr = np.sqrt(np.diag(betas.hess_inv.todense()))
            tstat = betas.x / stderr
            pval = (1 - t.cdf(np.absolute(tstat), len(self.x)-1)) * 2

            parameters = np.vstack((parameters, np.round([self.r_x_plus, self.r_x_minus, self.r_y_plus, self.r_y_minus, self.beta_x_plus, self.beta_x_minus, self.beta_y_plus, self.beta_y_minus],1)))
            stderr_betas = np.vstack((stderr_betas, stderr))
            pvalue_betas = np.vstack((pvalue_betas, pval))

            iteration += 1
            #sys.stdout.write('---- iteration ' + str(iteration) + ' ----\n')
            sys.stdout.write(str(parameters[-1]) + '\n')
        
        return ([x_llimit, y_llmit], parameters[-1].copy(), stderr_betas[-1].copy(), pvalue_betas[-1].copy())