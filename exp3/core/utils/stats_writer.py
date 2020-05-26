from datetime import datetime
from scipy.stats import sem, t
import pandas as pd


def get_confidence_interval(data, confidence=0.95):
    """
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The batch of samples for which to compute the mean and the confidence
        interval. The stats are computed for each individual feature.
    confidence : float (default=0.95)
        The confidence of the interval.

    Returns
    -------
    stats : tuple of arrays of shape (n_features)
        A tuple where the first element contains the means and the second
        element the margins of the intervals.
    """
    n = len(data)
    m = data.mean(0)
    std_err = sem(data, axis=0)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return m, h


class MetricsLogger:

    def __init__(self, keys=None, timestamp=True):
        self._stats = {}
        self._timestamp = timestamp
        self._keys = keys

        if self._keys is not None:
            for key in self._keys:
                self._stats[key] = []

        if self._timestamp:
            self._stats['timestamp'] = []

    def log(self, file=None, **kwargs):
        if self._timestamp:
            self._stats['timestamp'].append(datetime.now())

        for key, value in kwargs.items():
            self._stats[key].append(value)
        
        if file is not None:
            self.save(file)

    def save(self, file):
        pd.DataFrame(self._stats).to_csv(file, index=False)
