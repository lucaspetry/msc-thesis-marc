from core.utils import parse_file_prefix

import glob
from os import path
import os
import pandas as pd
import re


RESULTS_FOLDER = 'results/*[!\.csv]'
CV_FILE = 'cross_validation_results.csv'
MODEL_FILE = 'model_fold_*.csv'
TRAINING_DATA_FILE = 'results/summary_training_data_{}.csv'

runs = sorted(glob.glob(RESULTS_FOLDER))
all_data = {}


for run in runs:
    print("Processing run '{}'".format(run))
    params = parse_file_prefix(run)
    cv_file = path.join(run, CV_FILE)
    cv_data = pd.read_csv(cv_file, index_col='fold')
    cv_data.drop(['mean', 'interval'], axis=0, inplace=True)
    cv_data.drop('timestamp', axis=1, inplace=True)

    cols = cv_data.columns
    means = cv_data.mean().values * 100
    stds = cv_data.std().values * 100
    rate = params['embedding_rate']

    for i, c in enumerate(cols):
        if c not in all_data:
            all_data[c] = {}

        c_mean = 'avg_{}'.format(c)
        c_std = 'std_{}'.format(c)
        number = '{:.2f} $\\pm$ {:.2f}'.format(means[i], stds[i])

        if 'rate' in all_data[c]:
            all_data[c]['datafile'].append(params['datafile'])
            all_data[c]['embedder'].append(params['embedder'])
            all_data[c]['embedding_trainable'].append(params['embedding_trainable'])
            all_data[c]['rate'].append('{:.2f}'.format(rate))
            all_data[c]['value'].append(number)
        else:
            all_data[c]['datafile'] = [params['datafile']]
            all_data[c]['embedder'] = [params['embedder']]
            all_data[c]['embedding_trainable'] = [params['embedding_trainable']]
            all_data[c]['rate'] = ['{:.2f}'.format(rate)]
            all_data[c]['value'] = [number]


def multiindex_pivot(df, columns=None, values=None):
    #https://github.com/pandas-dev/pandas/issues/23955
    names = list(df.index.names)
    df = df.reset_index()
    list_index = df[names].values
    tuples_index = [tuple(i) for i in list_index] # hashable
    df = df.assign(tuples_index=tuples_index)
    df = df.pivot(index="tuples_index", columns=columns, values=values)
    tuples_index = df.index  # reduced
    index = pd.MultiIndex.from_tuples(tuples_index, names=names)
    df.index = index
    return df


for col, data in all_data.items():
    df_file = TRAINING_DATA_FILE.format(col)
    print("Saving training data to '{}'".format(df_file))
    df = pd.DataFrame(data).set_index(['datafile', 'embedder', 'embedding_trainable'])
    df = multiindex_pivot(df, columns='rate', values='value')
    df.to_csv(df_file)
