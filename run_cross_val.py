from core.logger import cur_date_time, print_header
from core.utils.data_loader import get_trajectories
from core.utils.nested_cross_val import NestedCrossValidator
from core.utils.metrics import compute_acc_acc5_f1_prec_rec
from core.models import MARC
from core.embedding.cbow import CBOW

import argparse
import random
import re
import numpy as np
import torch
import torch.nn as nn


###############################################################################
#   PARSE ARGS
###############################################################################
argparser = argparse.ArgumentParser()

argparser.add_argument('data_file', type=str)
argparser.add_argument('results_file', type=str)
argparser.add_argument('--tid-col', type=str, default='tid')
argparser.add_argument('--label-col', type=str, default='label')
argparser.add_argument('--folds', type=int, default=5)
argparser.add_argument('--merge-type', type=str, default='concatenate',
                       choices=['add', 'average', 'concatenate'])
argparser.add_argument('--rnn-cell', type=str, default='lstm',
                       choices=['lstm', 'gru'])
argparser.add_argument('--embedding-size', type=int, default=100)
argparser.add_argument('--rnn-state-size', type=int, default=100)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--lrate', type=float, default=0.001)
argparser.add_argument('--epochs', type=int, default=1000)
argparser.add_argument('--bs-train', type=int, default=128)
argparser.add_argument('--bs-test', type=int, default=512)
argparser.add_argument('--patience', type=int, default=-1)
argparser.add_argument('--seed', type=int, default=1234)
argparser.add_argument('--verbose', action='store_true')
argparser.add_argument('--cuda', action='store_true')

args = argparser.parse_args()

# Set seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# Print args
print_header('PARAMS')
input_args = re.findall("[\(\s)](.+?)=(.+?)[\,\)]", str(args))

for arg, value in input_args:
    print(arg, '=', value)

print('')


###############################################################################
#   LOAD DATA
###############################################################################
print_header('LOAD DATA')
(attr, attr_sizes,
 attr_encoders, tids, all_x, all_y) = get_trajectories(file=args.data_file,
                                                       tid_col=args.tid_col,
                                                       label_col=args.label_col)
num_classes = len(set(all_y))
print('')


###############################################################################
#   BUILD EMBEDDINGS
###############################################################################
print_header('EMBEDDING TRAINING')
embedder = CBOW(attributes=attr, vocab_size=attr_sizes,
                embedding_size=args.embedding_size,
                window=1,
                negative_sampling=5)

if args.cuda:
    embedder = embedder.cuda()

embedder.train_model(x=all_x, lrate=0.025, min_lrate=0.0001, epochs=500,
                     batch_size=1000, patience=20, threshold=0.01,
                     cuda=args.cuda, verbose=True)
print('')


###############################################################################
#   NESTED CROSS-VALIDATION
###############################################################################
print_header('CROSS-VALIDATION')
cross_val = NestedCrossValidator(args.folds, shuffle=True,
                                 random_state=args.seed)
total_folds = args.folds * (args.folds - 1)


for fold, (train_idxs, val_idxs, test_idxs) in enumerate(cross_val.split(tids, all_y)):
    train_x = all_x[train_idxs]
    train_y = all_y[train_idxs]
    val_x = all_x[val_idxs]
    val_y = all_y[val_idxs]
    test_x = all_x[test_idxs]
    test_y = all_y[test_idxs]

    model = MARC(attributes=attr,
                 vocab_size=attr_sizes,
                 embedding_size=args.embedding_size,
                 rnn_cells=args.rnn_state_size, output_size=num_classes,
                 merge_type=args.merge_type, rnn_cell=args.rnn_cell,
                 dropout=args.dropout)

    model.init_embedding_layer({a: embedder.embedding_layer(a) for a in attr},
                               trainable=True)

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.9)
    loss_func = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_epoch = -1
    best_model = None
    early_stop = args.patience > 0

    for epoch in range(1, args.epochs + 1):
        sample_idxs = torch.randperm(len(train_x)).long()
        
        train_loss = 0
        y_true = []
        y_pred = []

        model.train()

        for batch in sample_idxs.split(args.bs_train):
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
        for batch in sample_idxs.split(args.bs_test):
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

        if early_stop and epoch - best_val_epoch > args.patience:
            # print('Early stopping!')
            break

    # Test model
    model.load_state_dict(best_model)
    model.eval()

    y_true = torch.Tensor(test_y).long()
    y_pred = []

    sample_idxs = torch.Tensor(range(len(test_x))).long()
    for batch in sample_idxs.split(args.bs_test):
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
    print('{} | Fold {: >2}/{: >2} | Acc: {:.4f}  Acc5: {:.4f} Prec: {:.4f} Rec: {:.4f} F1: {:.4f}'.format(
        cur_date_time(), fold + 1, total_folds, test_acc, test_acc5, test_prec, test_rec, test_f1))
