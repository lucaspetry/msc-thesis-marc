from ..logger import cur_date_time
from .geohash import bin_geohash

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def get_trajectories(file, tid_col='tid', label_col='label', geo_precision=8, drop=[]):
    print(cur_date_time(), "| Loading trajectories from file '{}'...".format(file))
    df = pd.read_csv(file)
    keys = list(df.keys())
    attr_sizes = {}
    attr_encoders = {}
    keys.remove(tid_col)

    for col in drop:
        if col in keys:
            keys.remove(col)
            print(cur_date_time(), "| Warning: Column '{}' dropped from input file!".format(col))
        else:
            print(cur_date_time(), "| Warning: Column '{}' cannot be dropped because it was not found!".format(col))

    count_attr = 0
    lat_lon = False

    if 'lat' in keys and 'lon' in keys:
        keys.remove('lat')
        keys.remove('lon')
        lat_lon = True
        count_attr += geo_precision * 5
        print(cur_date_time(), "| Attribute Lat/Lon: {}-bits value".format(geo_precision * 5))

    for attr in keys:
        attr_encoders[attr] = LabelEncoder()
        df[attr] = attr_encoders[attr].fit_transform(df[attr])
        attr_sizes[attr] = max(df[attr]) + 1

        if attr != label_col:
            values = len(set(df[attr]))
            count_attr += values
            print(cur_date_time(), "| Attribute '{}': {} unique values".format(attr, values))

    print(cur_date_time(), "| Total of attribute/value pairs: {}".format(count_attr))
    keys.remove(label_col)

    x = []
    y = []
    tids = df[tid_col].unique()

    for idx, tid in tqdm(enumerate(tids), desc='Processing trajectories', total=len(tids)):
        traj = df.loc[df[tid_col].isin([tid])]
        x.append(traj.loc[:, keys].values)

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj['lat'].values[i]
                lon = traj['lon'].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))

            x[-1] = np.hstack([x[-1], np.array(loc_list)]).tolist()

        label = traj[label_col].iloc[0]
        y.append(label)

    if lat_lon:
        keys.append('lat_lon')
        attr_sizes['lat_lon'] = geo_precision * 5

    label_encoder = LabelEncoder().fit(y)

    y = label_encoder.transform(y)
    # logger.log(Logger.INFO, "Loading data from files " + file_str + "... DONE!")
    
    x = np.array(x)
    y = np.array(y)

    print(cur_date_time(), '| Trajectories:   {: >6}'.format(len(tids)))
    print(cur_date_time(), '| Labels/classes: {: >6}'.format(len(label_encoder.classes_)))

    return (keys, attr_sizes, attr_encoders, tids, x, y)
