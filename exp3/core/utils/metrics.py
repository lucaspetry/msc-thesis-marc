from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

import warnings


def _process_pred(y_pred):
    return np.argmax(y_pred, axis=1)


def precision_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return precision_score(y_true, proc_y_pred, average='macro')


def recall_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return recall_score(y_true, proc_y_pred, average='macro')


def f1_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f1_score(y_true, proc_y_pred, average='macro')


def accuracy(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return accuracy_score(y_true, proc_y_pred, normalize=True)


def accuracy_top_k(y_true, y_pred, k=5):
    order = np.argsort(y_pred, axis=1)
    correct = 0

    for i, sample in enumerate(y_true):
        if sample in order[i, -k:]:
            correct += 1

    return correct / len(y_true)


def compute_acc_acc5_f1_prec_rec(y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    acc_top5 = accuracy_top_k(y_true, y_pred, k=5)
    _f1_macro = f1_macro(y_true, y_pred)
    _prec_macro = precision_macro(y_true, y_pred)
    _rec_macro = recall_macro(y_true, y_pred)
    return acc, acc_top5, _f1_macro, _prec_macro, _rec_macro
