# Evaluation of Embedding Size and Initialization Techniques

Source code for running MARC with different embedding sizes and initialization techniques, using a nested-cross validation approach.

### Setup

1. In order to run the code you first need to install all the Python dependencies listed in `requirements.txt`. You can do that with pip (works only with Python 3, tested with Python 3.7.5):
    ```
    pip install -r requirements.txt
    ```

2. Or if you use the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment manager you can just run the following to create an environment with Python 3.7.5 and all required dependencies (replace `ENV_NAME` with whatever name you'd like):
    ```
    conda env create -f environment.yml --name ENV_NAME
    ```
    And then activate it with:
    ```
    conda activate ENV_NAME
    ```

### Usage

If you prefer, all the Shell scripts built for running these experiments are in this folder (files `run_exp_*.sh`). Otherwise, please see below:

```
usage: run_cross_val.py [-h] [--data.tid_col DATA.TID_COL]
                        [--data.label_col DATA.LABEL_COL]
                        [--data.folds DATA.FOLDS]
                        [--embedder.type {random,gcbow,icbow,autoencoder,pca}]
                        [--embedder.window EMBEDDER.WINDOW]
                        [--embedder.lrate EMBEDDER.LRATE]
                        [--embedder.min_lrate EMBEDDER.MIN_LRATE]
                        [--embedder.epochs EMBEDDER.EPOCHS]
                        [--embedder.bs_train EMBEDDER.BS_TRAIN]
                        [--embedder.patience EMBEDDER.PATIENCE]
                        [--model.embedding_size MODEL.EMBEDDING_SIZE]
                        [--model.embedding_rate MODEL.EMBEDDING_RATE]
                        [--model.embedding_trainable]
                        [--model.merge_type {concatenate}]
                        [--model.rnn_type {lstm,gru}]
                        [--model.rnn_cells MODEL.RNN_CELLS]
                        [--model.dropout MODEL.DROPOUT]
                        [--model.lrate MODEL.LRATE]
                        [--model.epochs MODEL.EPOCHS]
                        [--model.bs_train MODEL.BS_TRAIN]
                        [--model.bs_test MODEL.BS_TEST]
                        [--model.patience MODEL.PATIENCE]
                        [--results.confidence RESULTS.CONFIDENCE]
                        [--results.folder RESULTS.FOLDER] [--n_jobs N_JOBS]
                        [--prefix PREFIX] [--seed SEED] [--save-models]
                        [--verbose] [--cuda]
                        data.file

positional arguments:
  data.file             The CSV data file to be used.

optional arguments:
  -h, --help            show this help message and exit
  --data.tid_col DATA.TID_COL
                        Name of the column in the data file that represents
                        trajectory IDs (default=tid).
  --data.label_col DATA.LABEL_COL
                        Name of the column in the data file that represents
                        trajectory labels (default=label).
  --data.folds DATA.FOLDS
                        The number of folds of the k x (k-1)-fold nested
                        cross-validation (default=5).
  --embedder.type {random,gcbow,icbow,autoencoder,pca}
                        The embedding technique used for pretraining
                        embeddings (default=random).
  --embedder.window EMBEDDER.WINDOW
                        The context window size of the embedder (for 'gcbow'
                        and 'icbow' only) (default=1).
  --embedder.lrate EMBEDDER.LRATE
                        The initial learning rate for training the embedder
                        model (default=0.025).
  --embedder.min_lrate EMBEDDER.MIN_LRATE
                        The minimum learning rate for training the embedder
                        model (default=0.0001).
  --embedder.epochs EMBEDDER.EPOCHS
                        The maximum number of epochs for which to train the
                        embedder model (default=500).
  --embedder.bs_train EMBEDDER.BS_TRAIN
                        The training batch size for the embedder model
                        (default=1000).
  --embedder.patience EMBEDDER.PATIENCE
                        The number of epochs without improvement to wait
                        before early stopping the training of the embedder
                        model (default=20).
  --model.embedding_size MODEL.EMBEDDING_SIZE
                        The embedding size of each attribute. This option
                        overrides 'embedding-rate' (default=0).
  --model.embedding_rate MODEL.EMBEDDING_RATE
                        The compression rate of the input (default=1.0).
  --model.embedding_trainable
                        Whether or not the embedding layer is trainable in the
                        classifier model.
  --model.merge_type {concatenate}
                        The type of merge operation for merging attribute
                        embeddings in the classifier model
                        (default=concatenate).
  --model.rnn_type {lstm,gru}
                        The type of RNN cells of the classifier model
                        (default=lstm).
  --model.rnn_cells MODEL.RNN_CELLS
                        The number of RNN cells used in the recurrent layer of
                        the classifier model (default=100).
  --model.dropout MODEL.DROPOUT
                        The dropout rate of the classifier model
                        (default=0.5).
  --model.lrate MODEL.LRATE
                        The initial learning rate for training the classifier
                        model (default=0.001).
  --model.epochs MODEL.EPOCHS
                        The maximum number of epochs for which to train the
                        classifier model (default=1000).
  --model.bs_train MODEL.BS_TRAIN
                        The training batch size for the classifier model
                        (default=128).
  --model.bs_test MODEL.BS_TEST
                        The validation/testing batch size for the classifier
                        model (default=512).
  --model.patience MODEL.PATIENCE
                        The number of epochs without improvement to wait
                        before early stopping the training of the classifier
                        model (default=-1).
  --results.confidence RESULTS.CONFIDENCE
                        The confidence of the intervals for the statiscs means
                        (default=0.95).
  --results.folder RESULTS.FOLDER
                        The folder where all models and results files will be
                        saved (default=results).
  --n_jobs N_JOBS       The number of jobs to use when parallelization is
                        possible (default=1).
  --prefix PREFIX       A prefix to add to all files saved (default=).
  --seed SEED           The random seed used in the cross-validation split and
                        for running the models (default=1234).
  --save-models         If passed, saves the embedder and classifier models
                        for each fold.
  --verbose             If passed, prints more detailed information during the
                        execution of the script.
  --cuda                If passed, trains the models on any available GPUs.
```