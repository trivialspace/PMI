#%% [markdown]
# ##CO2 reduction computation  
# Let P be the total pollution produced in one month.   

# Without reduction, the total would be 12 times the pollution P  

# With reduction, we have a geometric ratio 0.9**0+0.9**1+0.9**2... 0.9**11  

# This evaluates to (1-0.9**12)/(1-0.9) = 7.1757  

# The difference is therefore 4.8243 .  

# The pollution produced is 404 g/mile * (5/8) mile/km = 252.5 g / km  

# The total pollution reduction is 252.5 * 4.8243 * Total km  

# This represents a reduction of about 40%.  


# To compute the distance travelled between two consecutive points, we use the L1 distance, which better mimics car travel in a city (the streets in San Francisco are more or less oriented according to the cardinal points)  

# The number of KM travelled while empty comes out to about 2.3 * 10^6 km (computing the distance between points on consecutive rows).  

# Total CO2 saved: 2.8 *10^6 kg  

#%%
import pandas as pd
import utm
import math
import os
from pathlib import Path
#%%
# Reading only one file for testing.
#df = pd.read_csv("./cabspottingdata/new_abboip.txt", sep=" ", names=['lat', 'long', 'busy','timestamp'])

#%% Convert to projected coordinates

#%% 
# Euclidean distance

def dist(a:float,b:float)->float:
    if math.isnan(a) or math.isnan(b):
        return 0.0
    else:
        return math.sqrt(a**2 + b**2) 

#%% 
# Distance on empty uses L1

def distance_on_empty(df:pd.DataFrame, col='distance_l1_km')->float:
    return df[df.busy==0][col].sum()

#%% 
# Prepare the dataframe, computing many derived pieces of information that will be useful for the second question

def prepare_dataframe(df:pd.DataFrame)->pd.DataFrame:
    df.lat = pd.to_numeric(df.lat, errors='coerce')
    df.long = pd.to_numeric(df.long, errors='coerce')
    df.busy = pd.to_numeric(df.busy, errors='coerce', downcast='integer')
    df.timestamp = pd.to_numeric(df.timestamp, errors='coerce', downcast='integer')
    df.timestamp = pd.to_datetime(df.timestamp, unit='s')
    df['x']=df.apply(lambda z: utm.from_latlon(z['lat'], z['long'])[0], axis=1)
    df['y']=df.apply(lambda z: utm.from_latlon(z['lat'], z['long'])[1], axis=1)
    df['day']=df.timestamp.dt.day
    df['month'] = df.timestamp.dt.month
    df['x_distance']=(df.x - df.x.shift())
    df['y_distance']=(df.y - df.y.shift())
    df['duration'] = df.timestamp - df.timestamp.shift()
    df['distance_km'] = df.apply(lambda z: dist(z['x_distance'], z['y_distance']), axis=1)/1000
    df['distance_l1_km'] = df.apply(lambda z: abs(z['x_distance']) + abs(z['y_distance']), axis=1)/1000
    df['time_15min']=df.timestamp.dt.round('15min')
    df['dropoff'] = (df.busy - df.busy.shift())==-1
    df['pickup'] = (df.busy - df.busy.shift())==1
    df['x_round10'] = df.x.round(-1)
    df['y_round10'] = df.y.round(-1)
    df['weekday'] = df.timestamp.dt.weekday<5
    df.loc[(df.month==5) & (df.day==26),'weekday']=False # Memorial Day is a weekend
    df['time_only'] = df['timestamp'].dt.time
    return df

#%% 
# Iterating through the files is quite time consuming. The UTM computations are costly but necessary: they allow to convert from the (latitude, longitude) system to a projected system where we can compute usual distances. 

def iterate_files()->(pd.DataFrame, int):
    files = os.listdir('./cabspottingdata')
    result_df = []
    result_km = []
    for file in files:
        df = pd.read_csv('./cabspottingdata/'+file, sep=" ", names=['lat', 'long', 'busy','timestamp'])
        df['driver'] = file.split(sep='.')[0]
        df = prepare_dataframe(df)
        total_km = distance_on_empty(df)
        result_km.append(total_km)
        df = df[df.pickup]
        result_df.append(df.copy())
    return pd.concat(result_df), result_km

def iterate_files2():
    result_df = []
    files = os.listdir('./cabspottingdata')
    for file in files:
        df = pd.read_csv('./cabspottingdata/'+file, sep=" ", names=['lat', 'long', 'busy','timestamp'])
        df['driver'] = file.split(sep='.')[0]
        result_df.append(df.copy())
    df_full = pd.concat(result_df)
    df_full.to_csv("full_df.csv", index=False)
    return df_full


#%% 
# Treat large file by chunks (for iterate_files2)
result_df = []
result_km = []
for chunk in pd.read_csv('full_df.csv', chunksize=10**6):
    df = prepare_dataframe(chunk)
    total_km = distance_on_empty(df)
    result_km.append(total_km)
    df = df[df.pickup]
    result_df.append(df.copy())

