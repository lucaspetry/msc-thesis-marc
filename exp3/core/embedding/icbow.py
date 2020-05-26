from .gcbow import GroupedCBOW
from ..logger import cur_date_time
from ..utils import MetricsLogger
from ..utils import compute_acc_acc5_f1_prec_rec

import torch
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed


class IsolatedCBOW(nn.Module):

    def __init__(self, attributes, vocab_size, embedding_size=100, window=5,
                 negative_sampling=5):
        super(IsolatedCBOW, self).__init__()
        self.attributes = attributes
        self.embedding_size = embedding_size
        self.window = window
        self.negative_sampling = negative_sampling
        
        if not isinstance(self.embedding_size, dict):
            self.embedding_size = {key: embedding_size for key in attributes}

        class_dim = sum(self.embedding_size.values())

        for idx, key in enumerate(attributes):
            setattr(self, 'embedder_' + key, GroupedCBOW(attributes=[key],
                                                         vocab_size={key: vocab_size[key]},
                                                         embedding_size={key: embedding_size[key]},
                                                         window=window))

    def embedding_layer(self, attribute):
        embedder = getattr(self, 'embedder_' + attribute)
        return getattr(embedder, 'embed_' + attribute).weight.data.cpu()

    def fit(self, x, lrate=0.025, min_lrate=0.0001, epochs=100,
            batch_size=1000, patience=-1, threshold=0.001, log_file=None,
            cuda=False, n_jobs=4, verbose=False):
        new_x = self._prepare_training_data(x)

        embedders = [getattr(self, 'embedder_' + key) for key in self.attributes]

        def train_embedding(e, emb_x):
            e.fit(x=emb_x, lrate=lrate, min_lrate=min_lrate, epochs=epochs,
                  batch_size=batch_size, patience=patience, threshold=threshold,
                  log_file=log_file, cuda=cuda, verbose=verbose)

        func = delayed(train_embedding)

        ret = Parallel(n_jobs=n_jobs, verbose=0)(
            func(e, new_x[i]) for i, e in enumerate(embedders))
        return self

    def _prepare_training_data(self, x):
        """
        Parameters
        ----------
        x : array-like of shape (n_samples, max_length, n_features)
            Description.
        """
        new_x = [[] for _ in x[0][0]]

        for sample in x:
            for i in range(0, len(new_x)):
                new_x[i].append(np.expand_dims(sample[:, i], 1))

        return np.array(new_x)
