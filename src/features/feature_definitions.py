# in real life the 10 input features will come as a raw on the server when the user query
# while in dfing 10 input features will be in the form of data frame.
# So, the code must be applicable on the one raw and dataset also.

# this file create function definitions for the features we will used by the build_features.py and the app.py

import pandas as pd
import numpy as np
import pathlib

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

### the function and the below two will be used in the app.py to make the model work.
def datetime_feature_fix(df):
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df.loc[:, 'pickup_date'] = df['pickup_datetime'].dt.date
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
###
def create_dist_features(df):
    df.loc[:, 'distance_haversine'] = haversine_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    df.loc[:, 'direction'] = bearing_array(df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
###
def create_datetime_features(df):
    df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
    df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
    df.loc[:, 'pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
    df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']

# this function will be used in the build_features.py to create the 15 features from the 8 input features
# removing certain features that we won't be using for the model.
# this function will also be used for the 15 features creation in the app.py in real time
def feature_build(df, tag):
    datetime_feature_fix(df)
    create_dist_features(df)
    create_datetime_features(df)
    do_not_use_for_training = ['id', 'pickup_datetime', 'dropoff_datetime',
                            'check_trip_duration', 'pickup_date', 'pickup_datetime_group']
    feature_names = [f for f in df.columns if f not in do_not_use_for_training]
    print(f'We have {len(feature_names)} features in {tag}.')
    return df[feature_names]

# for testing the working of feature_definitions.py if the same file will run
def test_feature_build(df):
    # fixing datetime features
    datetime_feature_fix(df)
    # creating distance features
    create_dist_features(df)
    # creating datetime features
    create_datetime_features(df)
    print(df.head())

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_path = home_dir.as_posix() + '/data/raw/test.csv'
    # testing for 10 rows only
    data = pd.read_csv(data_path, nrows=10)
    test_feature_build(data)