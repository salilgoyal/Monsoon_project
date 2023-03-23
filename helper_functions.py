# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: NERSC Python
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import pickle
import math

# +
# allows any data type to be stored in a file with extension .pickle
def store_as_pickle(data, filepath):
    pickle_out = open(filepath, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()
# -

# +
# allows a file with extension .pickle to be loaded
def load_from_pickle(filename):
    pickle_in = open(filename, 'rb')
    return pickle.load(pickle_in)
# -

# +
# does an inner join of all the dataframes inside the dictionaries 'model' and 'obs'
# careful: doing 'for key, df in obs.items()' or 'for key, df in model.items()' can give different
#    results when obs.items() != model.items(). This occurs when model and obs have only depressions
#    or lows (e.g., the set of depressions for ECMWF doesn't have to be the same as the set of 
#    depressions for ERA5).
# left data frame is obs, right data frame is model

def join_model_obs(model, obs, lsuffix, rsuffix):
    
    model = {key: df.set_index('date') for key, df in model.items()}
    obs = {key: df.set_index('date') for key, df in obs.items()}
    
    return {key: df.join(model[key], on='date', how='inner', lsuffix=lsuffix, rsuffix=rsuffix)
            for key, df in obs.items()}
# -

# +
# adds a column with the forecast lead time (starts at 0 and displays the number of 
#    hours elasped at each data point)

def add_leadtime_column(dic):
    
    return {key: pd.concat([df, 
                            (df.index.to_series().diff().astype('timedelta64[h]').
                             cumsum().fillna(0)).rename('ForecastLeadTime')
                           ],
                           axis=1) 
            for key,df in dic.items()}
# -


# +
# e.g. add_bias_columns(dic, 'ecmwf', 'era5', 'PressureDrop')
# pass in the output from add_leadtime_column

def add_bias_columns(dic, model_name, observation_name, var, relative=False):
    
    if relative==True: # adds an extra column with relative (i.e. (model-obs)/obs) bias
        modified_df = {key: pd.concat([df,
                                (df[var+'_'+model_name]-
                                 df[var+'_'+observation_name]).rename(var+'Bias'),
                                ((df[var+'_'+model_name]-
                                 df[var+'_'+observation_name])
                                 /df[var+'_'+observation_name]).rename('Relative'+var+'Bias'),
                           ],
                           axis=1) for key,df in dic.items()}
        
        # when calculating relative bias you might end up dividing by 0, so the line below
        #    takes care of those instances by removing those data points
        return {key: df.replace([np.inf, -np.inf], np.nan).dropna() 
                for key,df in modified_df.items()}
        
    else:
        return {key: pd.concat([df,
                                (df[var+'_'+model_name]-
                                     df[var+'_'+observation_name]).rename(var+'Bias'),
                               ],
                               axis=1) for key,df in dic.items()}
# -

# +
# adds the following columns to every df in your dictionary: 'LongitudeBias',
#    'LatitudeBias', 'DistanceBias'.
# pass in the output from add_leadtime_column

def add_position_bias_columns(dic, model_name, observation_name):
    new_dic = {key: pd.concat([df,
                            (df['Longitude_'+model_name]-
                                 df['Longitude_'+observation_name]).rename('LongitudeBias'),
                            (df['Latitude_'+model_name]-
                                 df['Latitude_'+observation_name]).rename('LatitudeBias'),
                             ],
                             axis=1) for key,df in dic.items()}
    
    # compute the distance in kilometers between two points, given the bias between them,
    #    using a cosine weighting formula supplied by Dr. Boos
    def compute_distance(lon_bias, lat_bias, lat_observation, lat_model):
    
        lat_avg = pd.concat([lat_observation, lat_model], axis=1).mean(axis=1)

        a = 6371e3 # radius of the earth, in meters

        dist = ( (a*(lon_bias*math.pi/180) * ((lat_avg*math.pi/180).apply(math.cos)))**2 
                 + 
                 (a*(lat_bias*math.pi/180))**2
               ).apply(math.sqrt) / 1000

        return dist # in kilometers

    return {key: pd.concat([df,
                            (compute_distance(df['LongitudeBias'], df['LatitudeBias'],
                                             df['Latitude_'+observation_name],
                                             df['Latitude_'+model_name])).rename('DistanceBias'),
                            ],
                            axis=1) for key,df in new_dic.items()}
# -


# +
# concat the relevant columns from all the dfs in the dictionary together into one df
# e.g. concat_bias(dic, 'ecmwf', 'era5', 'DistanceBias')
# pass in the output from add_bias_columns, or add_position_bias_columns

# concat is along axis=0 (i.e. rows are stacked on top of each other, num of columns = 2)

def concat_bias(dic, model_name, observation_name, var, relative=False):
    
    if relative==True:
        return pd.concat([df[['ForecastLeadTime', var+'Bias', 'Relative'+var+'Bias']] 
                          for key,df in dic.items()])
    else:
        return pd.concat([df[['ForecastLeadTime', var+'Bias']] for key,df in dic.items()])
# -


# +
# group a df on ForecastLeadTime
# pass in the output from concat_bias

# this will output columns titled 'count', 'mean', 'median', 'std', '<lambda_0>', '<lambda_1>'

def group_on_leadtime(df):
    return df.groupby('ForecastLeadTime').agg(['count', 'mean', 'median', 'std',
                                                                  lambda x: x.quantile(0.25),
                                                                  lambda x: x.quantile(0.75)])
# -


