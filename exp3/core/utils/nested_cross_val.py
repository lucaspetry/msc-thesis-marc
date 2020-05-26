import numpy as np
from sklearn.model_selection import StratifiedKFold


class NestedCrossValidator:
    """Cross-validator.
    Parameters
    ----------    
    n_splits : int
        The number of folds to split the data into.
    shuffle : bool (default=False)
        Whether or not the data should be shuffled before splitting.
    random_state : int (default=None)
        The seed used by the random number generator. If ``None``, the random 
        number generator is the RandomState instance used by ``np.random`` 
        (from sklearn).
    """

    def __init__(self, n_splits, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=random_state)

    def split(self, tids, labels):
        kfold1 = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                 random_state=self.random_state)
        kfold2 = StratifiedKFold(n_splits=self.n_splits - 1, shuffle=False)

        for trainval_idxs, test_idxs in kfold1.split(tids, labels):
            for train_idxs, val_idxs in kfold2.split(tids[trainval_idxs],
                                                     labels[trainval_idxs]):
                yield train_idxs, val_idxs, test_idxs
