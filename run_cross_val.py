from core.logger import cur_date_time, print_header
from core.utils import get_trajectories
from core.utils import NestedCrossValidator
from core.utils import MetricsLogger
from core.utils import compute_acc_acc5_f1_prec_rec
from core.utils import get_confidence_interval
from core.utils import parse_args, get_file_prefix
from core.models import MARC
from core.embedding import get_embedder

import random
import re
import numpy as np
import torch
import torch.nn as nn
import os
from os import path


###############################################################################
#   PARSE ARGS
###############################################################################
args = parse_args()

# Print args
print_header('PARAMS')

max_arg_len = max([len(n) for n in args.__dict__.keys()])

for arg, value in dict(args.__dict__).items():
    print('{: <{len}} = {}'.format(arg, value, len=max_arg_len))
    setattr(args, arg.replace('-', '_').replace('.', '_'), value)

print('')

# Set seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)


###############################################################################
#   RESULTS FODLER AND FILES
###############################################################################
print_header('RESULTS FOLDER AND FILES')
ROOT_FOLDER = path.join(path.dirname(path.realpath(__file__)),
                        args.results_folder)
EXP_FOLDER = path.join(ROOT_FOLDER, get_file_prefix(args))

print('{} | Exp. results folder: {}'.format(cur_date_time(), EXP_FOLDER))

try:
    os.makedirs(EXP_FOLDER)
except Exception as e:
    print('{} | Warning: {}'.format(cur_date_time(), e))

RES_FILE_CV = path.join(EXP_FOLDER, 'cross_validation_results.csv')

print('')


###############################################################################
#   LOAD DATA
###############################################################################
print_header('LOAD DATA')
(attr, attr_sizes, attr_encoders,
 max_length, tids, all_x, all_y) = get_trajectories(file=args.data_file,
                                                    tid_col=args.data_tid_col,
                                                    label_col=args.data_label_col)
num_classes = len(set(all_y))
print('')


###############################################################################
#   BUILD EMBEDDINGS
###############################################################################
print_header('EMBEDDING DIMENSIONS')
EMBEDDING_SIZE = {key: args.model_embedding_size for key in attr}

if args.model_embedding_size == 0:
    EMBEDDING_SIZE = {key: round(attr_sizes[key] * args.model_embedding_rate) for key in attr}    

max_attr_len = max([len(n) for n in attr])

for key in attr:
    print('{: <{len}} = {: >3}'.format(key, EMBEDDING_SIZE[key], len=max_attr_len))

print('{: <{len}} = {: >3}'.format('TOTAL', sum(EMBEDDING_SIZE.values()), len=max_attr_len))
print('')


###############################################################################
#   NESTED CROSS-VALIDATION
###############################################################################
print_header('CROSS-VALIDATION')
cross_val = NestedCrossValidator(args.data_folds, shuffle=True,
                                 random_state=args.seed)
total_folds = args.data_folds * (args.data_folds - 1)
stats_names = ['Accuracy @ 1', 'Accuracy @ 5', 'Macro-F1', 'Macro-Precision', 'Macro-Recall']
stats = []

cv_logger = MetricsLogger(keys=np.concatenate([['fold'], stats_names]), timestamp=True)


for fold, (train_idxs, val_idxs, test_idxs) in enumerate(cross_val.split(tids, all_y)):
    train_x = all_x[train_idxs]
    train_y = all_y[train_idxs]
    val_x = all_x[val_idxs]
    val_y = all_y[val_idxs]
    test_x = all_x[test_idxs]
    test_y = all_y[test_idxs]

    model = MARC(attributes=attr,
                 vocab_size=attr_sizes,
                 embedding_size=EMBEDDING_SIZE,
                 rnn_cells=args.model_rnn_cells, output_size=num_classes,
                 merge_type=args.model_merge_type, rnn_cell=args.model_rnn_type,
                 dropout=args.model_dropout)

    embedder = get_embedder(attributes=attr, vocab_size=attr_sizes,
                            embedding_size=EMBEDDING_SIZE, embedder_type=args.embedder_type)

    if embedder is not None:
        if args.cuda:
            embedder = embedder.cuda()

        if args.embedder_type != 'pca':
            embedder.fit(x=np.concatenate([train_x, val_x]),
                         lrate=args.embedder_lrate,
                         min_lrate=args.embedder_min_lrate,
                         epochs=args.embedder_epochs,
                         batch_size=args.embedder_bs_train,
                         patience=args.embedder_patience, threshold=0.01,
                         cuda=args.cuda, verbose=False)
            model.init_embedding_layer({a: embedder.embedding_layer(a) for a in attr},
                                       trainable=args.model_embedding_trainable)
        else:
            raise Exception('PCA is currently not working!')

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lrate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.9)
    loss_func = nn.NLLLoss()

    best_val_loss = float('inf')
    best_val_epoch = -1
    best_model = None
    early_stop = args.model_patience > 0

    for epoch in range(1, args.model_epochs + 1):
        sample_idxs = torch.randperm(len(train_x)).long()
        
        train_loss = 0
        y_true = []
        y_pred = []

        model.train()

        for batch in sample_idxs.split(args.model_bs_train):
            optimizer.zero_grad()
            x = train_x[batch]
            y = torch.Tensor(train_y[batch]).long()
            y_true.append(y)

            lengths = torch.Tensor([len(seq) for seq in x]).long()
            x = nn.utils.rnn.pad_sequence([torch.Tensor(seq).long() for seq in x],
                                          batch_first=True, padding_value=0)

            if args.cuda:
                x = x.cuda()
                y = y.cuda()
                lengths = lengths.cuda()

            pred_y = model(x, lengths)
            y_pred.append(pred_y.data.cpu())

            loss = loss_func(pred_y, y)
            train_loss += loss
            loss.backward()
            optimizer.step()

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        (train_acc, train_acc5, train_f1,
         train_prec, train_rec) = compute_acc_acc5_f1_prec_rec(y_true, y_pred)

        val_loss = 0
        y_true = torch.Tensor(val_y).long()
        y_pred = []

        model.eval()
        sample_idxs = torch.Tensor(range(len(val_x))).long()
        for batch in sample_idxs.split(args.model_bs_test):
            x = val_x[batch]
            y = torch.Tensor(val_y[batch]).long()

            lengths = torch.Tensor([len(seq) for seq in x]).long()
            x = nn.utils.rnn.pad_sequence([torch.Tensor(seq).long() for seq in x],
                                          batch_first=True, padding_value=0)

            if args.cuda:
                x = x.cuda()
                y = y.cuda()
                lengths = lengths.cuda()

            pred_y = model(x, lengths)
            y_pred.append(pred_y.data.cpu())

            loss = loss_func(pred_y, y)
            val_loss += loss

        y_pred = torch.cat(y_pred)
        (val_acc, val_acc5, val_f1,
         val_prec, val_rec) = compute_acc_acc5_f1_prec_rec(y_true, y_pred)

        if args.verbose:
            print('{} | Epoch {: >4} |'.format(cur_date_time(), epoch),
                  'Train {{ Loss: {:8.4f}  Acc: {:.4f} F1: {:.4f} }}'.format(train_loss, train_acc, train_f1),
                  'Val {{ Loss: {:8.4f}  Acc: {:.4f} F1: {:.4f} }}'.format(val_loss, val_acc, val_f1))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_model = model.state_dict()

        if early_stop and epoch - best_val_epoch > args.model_patience:
            # print('Early stopping!')
            break

    # Test model
    model.load_state_dict(best_model)
    model.eval()

    y_true = torch.Tensor(test_y).long()
    y_pred = []

    sample_idxs = torch.Tensor(range(len(test_x))).long()
    for batch in sample_idxs.split(args.model_bs_test):
        x = test_x[batch]
        y = torch.Tensor(test_y[batch]).long()

        lengths = torch.Tensor([len(seq) for seq in x]).long()
        x = nn.utils.rnn.pad_sequence([torch.Tensor(seq).long() for seq in x],
                                      batch_first=True, padding_value=0)

        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            lengths = lengths.cuda()

        pred_y = model(x, lengths)
        y_pred.append(pred_y.data.cpu())

    y_pred = torch.cat(y_pred)
    (test_acc, test_acc5, test_f1,
     test_prec, test_rec) = compute_acc_acc5_f1_prec_rec(y_true, y_pred)
    stats.append([test_acc, test_acc5, test_f1, test_prec, test_rec])
    cv_logger.log(file=RES_FILE_CV,
                  **{'fold': fold + 1,
                     'Accuracy @ 1': test_acc,
                     'Accuracy @ 5': test_acc5,
                     'Macro-F1': test_f1,
                     'Macro-Precision': test_prec,
                     'Macro-Recall': test_rec})
    print('{} | Fold {: >2}/{: >2} | Acc: {:.4f}  Acc5: {:.4f} Prec: {:.4f} Rec: {:.4f} F1: {:.4f}'.format(
        cur_date_time(), fold + 1, total_folds, test_acc, test_acc5, test_prec, test_rec, test_f1))

print('')
print_header('SUMMARY')
means, intervals = get_confidence_interval(data=np.array(stats), confidence=args.results_confidence)
max_name_length = max([len(n) for n in stats_names])

cv_logger.log(file=RES_FILE_CV,
              **{'fold': 'mean',
                 'Accuracy @ 1': means[0],
                 'Accuracy @ 5': means[1],
                 'Macro-F1': means[2],
                 'Macro-Precision': means[3],
                 'Macro-Recall': means[4]})
cv_logger.log(file=RES_FILE_CV,
              **{'fold': 'interval',
                 'Accuracy @ 1': intervals[0],
                 'Accuracy @ 5': intervals[1],
                 'Macro-F1': intervals[2],
                 'Macro-Precision': intervals[3],
                 'Macro-Recall': intervals[4]})

for i, name in enumerate(stats_names):
    print("{} | {: <{len}}: {:.4f} +- {:.4f}".format(cur_date_time(), name, means[i], intervals[i], len=max_name_length))
