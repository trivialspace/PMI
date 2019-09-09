#%%
import pandas as pd
from scipy.stats import poisson
import numpy as np
import geopandas as gpd

import hvplot
from hvplot import hvPlot
import holoviews as hv
import hvplot.pandas
hv.extension('bokeh')

#%% [markdown]

#The idea is to subdivide the map into a grid with cells of dimension 10m X 10m. For a 24h time period, we count, for each x min interval, how many pickups have happened in this region of the grid.
# We compute the average and use this as the lambda parameter lambda_i for a Poisson random variable (one random variable for each cell of the grid). 
# For each cell, we count the number of taxi pickups over the last x min. This allows us to rank the cell according to how unlikely (lowest probability) it was to observe this many pickups with respect to a Poisson random variable of mean lambda_i. 
# We take the 5 grid cells with the lowest probability. This is our prediction. 

#To validate this approach, we pick a timestamp at random, compute the 5 grid cells with lowest probability and check to see how many of these had a pickup in the following 10/15 minutes. 


# We look at 5, 15, 30, 45, 60 minutes for a time window before the timestamp to get the best estimates for lambda_i.


#%% 
# Setup the pickups dataframe that was stored as part of Question 1
df = pd.read_csv('pickups.csv') 
df.timestamp = pd.to_datetime(df.timestamp)
df.weekday = df.timestamp.dt.weekday<5
df.loc[(df.month==5) & (df.day==26),'weekday']=False # Memorial Day is a weekend

timestamp = df.iloc[161500,3]
df.sort_values(by='timestamp', inplace=True)

#%% 
# Compute the lambda parameters for the grid, as well as the number of pickups before the timestamp
def less_than_x_min(timestamp:pd.Timestamp, minutes:int, df:pd.DataFrame, weekday:bool=True)->pd.DataFrame:
    df = df[df.weekday==weekday]
    df['most_recent'] = (df.timestamp - timestamp).between(pd.Timedelta(0, unit='minutes'), pd.Timedelta(minutes, unit='minutes'))
    df = df[df.day != timestamp.day]
    if weekday:
        total = 15
    else:
        total = 8
    days = (df.timestamp-timestamp).values.astype('timedelta64[D]')
    df['all_time_period'] = ((df.timestamp-timestamp)-days).between(pd.Timedelta(0, unit='minutes',), pd.Timedelta(minutes, unit='minutes'))
    df2 = df.groupby(['x_round10', 'y_round10', 'weekday'], as_index=False)[['most_recent', 'all_time_period']].sum()
    df2['all_time_period'] = df2['all_time_period']/total
    df2['prob']=poisson.pmf(df2.most_recent, df2.all_time_period)
    df2.sort_values(by='prob', ascending=True, inplace=True)
    return df2

#%% 
# Obtain the next pickups (10 min and 15 minutes were computed)
def next_pickups(timestamp:pd.Timestamp, df:pd.DataFrame, weekday:bool=True):
    df2 = df[df.weekday==weekday]
    df2 = df2[df2.timestamp>timestamp]
    df2 = df2[df2.timestamp < timestamp + pd.Timedelta('10 min')]
    return df2


#%% 
# Compute how many of the 5 predictions had a pickup in the next timewindow
def check_timestamp(myindex:int, df:pd.DataFrame, time_window:int):
    timestamp=df.loc[myindex, 'timestamp']
    weekday = df.loc[myindex, 'weekday']
    df_recent = less_than_x_min(timestamp, time_window, df, weekday)
    df_pickups = next_pickups(timestamp, df, weekday)
    hits = df_recent.head(5).x_round10.isin(df_pickups.x_round10) & df_recent.head(5).y_round10.isin(df_pickups.y_round10)
    return sum(hits)
#%% 
# Draw random indices and check the predictions
def validate(size:int, time_window:int, df:pd.DataFrame)->list:
    draws = np.floor(len(df)*np.random.rand(size)).astype(int)
    hits = list(map(lambda z:check_timestamp(df=df, myindex=z, time_window=time_window), draws))
    return hits
#%%
temp_dict = {}
for time_window in [5, 15, 30, 45, 60]:
    hits =validate(100, time_window, df)
    temp_dict[time_window]=hits.copy()
#%%
#df_check = pd.DataFrame(temp_dict)
#df_check.to_csv('df_check.csv')
df_check = pd.read_csv('df_check.csv')
df_check_10min = pd.read_csv('df_check_10min.csv')
#%%
# Plot the number of hits with a 15 min prediction interval
df_counts_15min = df_check.apply(pd.Series.value_counts)
#%%
df_counts_15min.transpose().hvplot(kind='bar')
#%%
df_counts_15min.hvplot(kind='bar')
#%% 
# Plot the number of hits with a 10 min prediction interval
df_counts_10min = df_check_10min.apply(pd.Series.value_counts)
#%%
df_counts_10min.transpose().hvplot(kind='bar')
#%%
df_counts_10min.hvplot(kind='bar')

#%% [markdown]

# Let us call the time windows before the timestamp (used to compute lambda_i) TW.
# We value histograms that have their weights shifted towards 5 (hits/5). 
# Looking at the 15 min validation window, the longer TW has its weight shifted to the right, but also has quite a few complete misses (0/5 hits).
# If minimizing the number of complete misses is highly valued, we should pick time 15 min TW.

#%% [markdown]

# Let us look at the 10 min validation window, which is a harsher criterion. 
# Here, 15 and 30 min for TW are doing better: they have fewer total misses (0/5 hits). 
# As such, it would make sense to pick a value between 15 and 30 min for the smoothing value TW.

#%% [markdown]
# One more observation is that grid resolution might be too fine. This allows fine grained predictions, but involves many grid cells (110 000). Perhaps a taxi driver would be satisfied with predictions at 100m resolution, and this would involve 10 000 grid cells.
# It would be interesting to run the analysis again with a grid resolution of 100m.
#%% 
# Plotting on a map(unfinished)
test = next_pickups(timestamp, df, weekday=True)
gdf = gpd.GeoDataFrame(test, geometry=gpd.points_from_xy(test.long, test.lat))
gdf.crs="+init=epsg:4326"
gdf.hvplot()

#%%
