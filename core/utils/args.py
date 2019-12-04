import argparse
from os import path


ARGS = [
    {
        'name' : 'data.file', 'type': str,
        'help': ''
    },
    {
        'name' : '--data.tid_col', 'type': str, 'default': 'tid',
        'help': ''
    },
    {
        'name' : '--data.label_col', 'type': str, 'default': 'label',
        'help': ''
    },
    {
        'name' : '--data.folds', 'type': int, 'default': 5,
        'help': ''
    },
    {
        'name' : '--embedder.type', 'type': str, 'default': 'random',
        'choices': ['random', 'word2vec', 'autoencoder', 'pca'],
        'help': ""
    },
    {
        'name' : '--embedder.lrate', 'type': float, 'default': 0.025,
        'help': ''
    },
    {
        'name' : '--embedder.min_lrate', 'type': float, 'default': 0.0001,
        'help': ''
    },
    {
        'name' : '--embedder.epochs', 'type': int, 'default': 500,
        'help': ''
    },
    {
        'name' : '--embedder.bs_train', 'type': int, 'default': 1000,
        'help': ''
    },
    {
        'name' : '--embedder.patience', 'type': int, 'default': 20,
        'help': ''
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
        'help': ''
    },
    {
        'name' : '--model.merge_type', 'type': str, 'default': 'concatenate',
        'choices': ['add', 'average', 'concatenate'],
        'help': ''
    },
    {
        'name' : '--model.rnn_type', 'type': str, 'default': 'lstm',
        'choices': ['lstm', 'gru'],
        'help': ''
    },
    {
        'name' : '--model.rnn_cells', 'type': int, 'default': 100,
        'help': ''
    },
    {
        'name' : '--model.dropout', 'type': float, 'default': 0.5,
        'help': ''
    },
    {
        'name' : '--model.lrate', 'type': float, 'default': 0.001,
        'help': ''
    },
    {
        'name' : '--model.epochs', 'type': int, 'default': 1000,
        'help': ''
    },
    {
        'name' : '--model.bs_train', 'type': int, 'default': 128,
        'help': ''
    },
    {
        'name' : '--model.bs_test', 'type': int, 'default': 512,
        'help': ''
    },
    {
        'name' : '--model.patience', 'type': int, 'default': -1,
        'help': ''
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
        'name' : '--prefix', 'type': str, 'default': '',
        'help': 'A prefix to add to all files saved'
    },
    {
        'name' : '--seed', 'type': int, 'default': 1234,
        'help': ''
    },
    {
        'name' : '--verbose', 'type': bool,
        'help': ''
    },
    {
        'name' : '--cuda', 'type': bool,
        'help': ''
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