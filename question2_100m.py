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

#We redo the whole computatiton with a grid resolution of 100m.

#%% Setup the pickups dataframe that was stored as part of Question 1
df = pd.read_csv('pickups.csv') 
df.timestamp = pd.to_datetime(df.timestamp)
df.weekday = df.timestamp.dt.weekday<5
df.loc[(df.month==5) & (df.day==26),'weekday']=False # Memorial Day is a weekend

df['x_round100'] = df.x.round(-2)
df['y_round100'] = df.y.round(-2)

timestamp = df.iloc[161500,3]
df.sort_values(by='timestamp', inplace=True)

#%% Compute the lambda parameters for the grid, as well as the number of pickups before the timestamp
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
    df2 = df.groupby(['x_round100', 'y_round100', 'weekday'], as_index=False)[['most_recent', 'all_time_period']].sum()
    df2['all_time_period'] = df2['all_time_period']/total
    df2['prob']=poisson.pmf(df2.most_recent, df2.all_time_period)
    df2.sort_values(by='prob', ascending=True, inplace=True)
    return df2

#%% Obtain the next pickups (10 min and 15 minutes were computed)
def next_pickups(timestamp:pd.Timestamp, df:pd.DataFrame, weekday:bool=True):
    df2 = df[df.weekday==weekday]
    df2 = df2[df2.timestamp>timestamp]
    df2 = df2[df2.timestamp < timestamp + pd.Timedelta('10 min')]
    return df2


#%% Compute how many of the 5 predictions had a pickup in the next timewindow
def check_timestamp(myindex:int, df:pd.DataFrame, time_window:int):
    timestamp=df.loc[myindex, 'timestamp']
    weekday = df.loc[myindex, 'weekday']
    df_recent = less_than_x_min(timestamp, time_window, df, weekday)
    df_pickups = next_pickups(timestamp, df, weekday)
    hits = df_recent.head(5).x_round100.isin(df_pickups.x_round100) & df_recent.head(5).y_round100.isin(df_pickups.y_round100)
    return sum(hits)
#%% Draw random indices and check the predictions
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
df_check = pd.DataFrame(temp_dict)
df_check.to_csv('df_check_100m_10min.csv', index=False)
#df_check = pd.read_csv('df_check.csv')
df_check_100m_10min = pd.read_csv('df_check_100m_10min.csv')

#%% Plot the number of hits with a 10 min prediction interval
df_counts_100m_10min = df_check_100m_10min.apply(pd.Series.value_counts)
#%%
df_counts_100m_10min.transpose().hvplot(kind='bar')
#%%
df_counts_100m_10min.hvplot(kind='bar')

#%% [markdown]

#Clearly here, a smoothing window of 60 min would be beneficial. It would make sense to investigate further and consider other metrics to compare 10m and 100m. Domain knowledge would also be beneficial (is 100m grid resolution an acceptable size for the prediction resolution?)