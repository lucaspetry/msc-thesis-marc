import argparse
from os import path
import re


ARGS = [
    {
        'name' : 'data.file', 'type': str,
        'help': 'The CSV data file to be used'
    },
    {
        'name' : '--data.tid_col', 'type': str, 'default': 'tid',
        'help': 'Name of the column in the data file that represents trajectory IDs'
    },
    {
        'name' : '--data.label_col', 'type': str, 'default': 'label',
        'help': 'Name of the column in the data file that represents trajectory labels'
    },
    {
        'name' : '--data.folds', 'type': int, 'default': 5,
        'help': 'The number of folds of the k x (k-1)-fold nested cross-validation'
    },
    {
        'name' : '--embedder.type', 'type': str, 'default': 'random',
        'choices': ['random', 'gcbow', 'icbow', 'autoencoder', 'pca'],
        'help': "The embedding technique used for pretraining embeddings"
    },
    {
        'name' : '--embedder.window', 'type': int, 'default': 1,
        'help': "The context window size of the embedder (for 'gcbow' and 'icbow' only)"
    },
    {
        'name' : '--embedder.lrate', 'type': float, 'default': 0.025,
        'help': 'The initial learning rate for training the embedder model'
    },
    {
        'name' : '--embedder.min_lrate', 'type': float, 'default': 0.0001,
        'help': 'The minimum learning rate for training the embedder model'
    },
    {
        'name' : '--embedder.epochs', 'type': int, 'default': 500,
        'help': 'The maximum number of epochs for which to train the embedder model'
    },
    {
        'name' : '--embedder.bs_train', 'type': int, 'default': 1000,
        'help': 'The training batch size for the embedder model'
    },
    {
        'name' : '--embedder.patience', 'type': int, 'default': 20,
        'help': 'The number of epochs without improvement to wait before early' + \
                ' stopping the training of the embedder model'
    },
    {
        'name' : '--model.embedding_size', 'type': int, 'default': 0,
        'help': "The embedding size of each attribute. This option overrides 'embedding-rate'"
    },
    {
        'name' : '--model.embedding_rate', 'type': float, 'default': 1.0,
        'help': 'The compression rate of the input'
    },
    {
        'name' : '--model.embedding_trainable', 'type': bool,
        'help': 'Whether or not the embedding layer is trainable in the classifier model'
    },
    {
        'name' : '--model.merge_type', 'type': str, 'default': 'concatenate',
        'choices': ['concatenate'],
        'help': 'The type of merge operation for merging attribute embeddings in the classifier model'
    },
    {
        'name' : '--model.rnn_type', 'type': str, 'default': 'lstm',
        'choices': ['lstm', 'gru'],
        'help': 'The type of RNN cells of the classifier model'
    },
    {
        'name' : '--model.rnn_cells', 'type': int, 'default': 100,
        'help': 'The number of RNN cells used in the recurrent layer of the classifier model'
    },
    {
        'name' : '--model.dropout', 'type': float, 'default': 0.5,
        'help': 'The dropout rate of the classifier model'
    },
    {
        'name' : '--model.lrate', 'type': float, 'default': 0.001,
        'help': 'The initial learning rate for training the classifier model'
    },
    {
        'name' : '--model.epochs', 'type': int, 'default': 1000,
        'help': 'The maximum number of epochs for which to train the classifier model'
    },
    {
        'name' : '--model.bs_train', 'type': int, 'default': 128,
        'help': 'The training batch size for the classifier model'
    },
    {
        'name' : '--model.bs_test', 'type': int, 'default': 512,
        'help': 'The validation/testing batch size for the classifier model'
    },
    {
        'name' : '--model.patience', 'type': int, 'default': -1,
        'help': 'The number of epochs without improvement to wait before early' + \
                ' stopping the training of the classifier model'
    },
    {
        'name' : '--results.confidence', 'type': float, 'default': 0.95,
        'help': 'The confidence of the intervals for the statiscs means'
    },
    {
        'name' : '--results.folder', 'type': str, 'default': 'results',
        'help': "The folder where all models and results files will be saved"
    },
    {
        'name' : '--n_jobs', 'type': int, 'default': 1,
        'help': 'The number of jobs to use when parallelization is possible'
    },
    {
        'name' : '--prefix', 'type': str, 'default': '',
        'help': 'A prefix to add to all files saved'
    },
    {
        'name' : '--seed', 'type': int, 'default': 1234,
        'help': 'The random seed used in the cross-validation split and for' + \
                ' running the models'
    },
    {
        'name' : '--save-models', 'type': bool,
        'help': 'If passed, saves the embedder and classifier models for each fold'
    },
    {
        'name' : '--verbose', 'type': bool,
        'help': 'If passed, prints more detailed information during the execution of the script'
    },
    {
        'name' : '--cuda', 'type': bool,
        'help': 'If passed, trains the models on any available GPUs'
    }
]


def parse_args():
    argparser = argparse.ArgumentParser()

    for arg in ARGS:
        if 'choices' not in arg:
            arg['choices'] = None
        if 'help' not in arg:
            arg['help'] = ''
        if 'default' not in arg:
            arg['default'] = None

        if arg['default'] is not None:
            arg['help'] += ' (default={}).'.format(arg['default'])
        else:
            arg['help'] += '.'

        if arg['type'] == bool:
            argparser.add_argument(arg['name'], action='store_true',
                                   help=arg['help'])
        else:
            argparser.add_argument(arg['name'], type=arg['type'],
                                   choices=arg['choices'],
                                   default=arg['default'],
                                   help=arg['help'])

    return argparser.parse_args()


def get_file_prefix(args):
    separator = '-'
    file = '{}'.format(args.prefix + separator if args.prefix != '' else '')
    file += 'datafile_{}{}'.format(path.basename(args.data_file), separator)
    file += 'folds_{}{}'.format(args.data_folds, separator)
    file += 'embedder_{}{}'.format(args.embedder_type, separator)

    if args.model_embedding_size > 0:
        file += 'embedding_size_{}{}'.format(args.model_embedding_size, separator)
    else:
        file += 'embedding_rate_{}{}'.format(args.model_embedding_rate, separator)

    file += 'embedding_trainable_{}{}'.format(args.model_embedding_trainable, separator)
    file += 'rnn_cells_{}{}'.format(args.model_rnn_cells, separator)
    file += 'seed_{}'.format(args.seed)
    return file


def parse_file_prefix(file):
    separator = '-'
    datafile = re.compile("datafile_(.+){}folds".format(separator)).search(file).group(1)
    folds = int(re.compile("folds_(.+){}embedder".format(separator)).search(file).group(1))

    if 'embedding_rate' in file:
        embedder = re.compile("embedder_(.+){}embedding_rate".format(separator)).search(file).group(1)
        embedding_size_rate = float(re.compile("embedding_rate_(.+){}embedding".format(separator)).search(file).group(1))
    else:
        embedder = re.compile("embedder_(.+){}embedding_size".format(separator)).search(file).group(1)
        embedding_size_rate = int(re.compile("embedding_size_(.+){}embedding".format(separator)).search(file).group(1))

    embedding_trainable = re.compile("embedding_trainable_(.+){}rnn_cells".format(separator)).search(file).group(1) == 'True'
    seed = int(re.compile("seed_(.+)$").search(file).group(1))

    return {'datafile': datafile,
            'folds': folds,
            'embedder': embedder,
            'embedding_rate': embedding_size_rate if 'embedding_rate' in file else None,
            'embedding_size': embedding_size_rate if 'embedding_size' in file else None,
            'embedding_trainable': embedding_trainable,
            'seed': seed}
