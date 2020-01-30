from core.utils import parse_file_prefix

import glob
from os import path
import os
import pandas as pd
import re


RESULTS_FOLDER = 'results/*[!\.csv]'
CV_FILE = 'cross_validation_results.csv'
MODEL_FILE = 'model_fold_*.csv'
TRAINING_DATA_FILE = 'results/log_overall_training_data.csv'

runs = sorted(glob.glob(RESULTS_FOLDER))
all_data = None

for run in runs:
    print("Processing run '{}'".format(run))
    params = parse_file_prefix(run)
    # cv_file = path.join(run, CV_FILE)
    model_files = sorted(glob.glob(path.join(run, MODEL_FILE)))

    # if not os.path.isfile(cv_file):
    #     continue

    for model_file in model_files:
        fold = int(re.compile("fold_(.+)\.csv").search(model_file).group(1))

        model_data = pd.read_csv(model_file)
        model_data['fold'] = fold

        for p in params:
            model_data[p] = params[p]

        if all_data is not None:
            all_data = pd.concat([all_data, model_data], axis=0, ignore_index=True)
        else:
            all_data = model_data

print("Saving training data to '{}'".format(TRAINING_DATA_FILE))
all_data.to_csv(TRAINING_DATA_FILE, index=False)
