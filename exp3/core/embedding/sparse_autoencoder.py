from .autoencoder import Autoencoder
from ..logger import cur_date_time
from ..utils import MetricsLogger
from ..utils import compute_acc_acc5_f1_prec_rec

import torch
import torch.nn as nn


class SparseAutoencoder(Autoencoder):

    def __init__(self, attributes, vocab_size, embedding_size, p=0.05):
        super(SparseAutoencoder,self).__init__(attributes, vocab_size, embedding_size)
        self.p = p

    def fit(self, x, lrate=0.025, min_lrate=0.0001, epochs=100,
            batch_size=1000, patience=-1, threshold=0.001, log_file=None,
            cuda=False, n_jobs=-1, verbose=False):
        model_logger = MetricsLogger(keys=['epoch', 'train_loss'], timestamp=True)

        train_x = nn.utils.rnn.pad_sequence([torch.Tensor(seq).long() for seq in x],
                                      batch_first=True, padding_value=-1)
        train_x = train_x.view(train_x.size(0) * train_x.size(1), -1)
        train_x = train_x[train_x.sum(dim=1) >= 0]

        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=10,
                                                               factor=0.1,
                                                               min_lr=min_lrate)
        loss_func = nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_epoch = -1
        best_model = None
        early_stop = patience > 0

        for epoch in range(1, epochs + 1):
            sample_idxs = torch.randperm(len(train_x)).long()
            train_loss = 0

            self.train()

            for batch in sample_idxs.split(batch_size):
                optimizer.zero_grad()
                x = train_x[batch]

                if cuda:
                    x = x.cuda()

                pred_y = self(x)
                loss = 0

                for i, pred in enumerate(pred_y):
                    loss += loss_func(pred, x[:, i])
                
                train_loss += loss
                loss.backward()
                optimizer.step()

            if verbose:
                print('{} | Epoch {: >4}/{} | Loss: {:.4f}'.format(
                    cur_date_time(), epoch, epochs, train_loss))

            if log_file is not None:
                model_logger.log(file=log_file,
                                 **{'epoch': epoch, 'train_loss': train_loss.item()})

            scheduler.step(train_loss)

            if best_loss - train_loss > threshold:
                best_loss = train_loss
                best_epoch = epoch
                best_model = self.state_dict()

            if early_stop and epoch - best_epoch > patience:
                self.load_state_dict(best_model)
                if verbose:
                    print('{} | Early stopping!'.format(cur_date_time()))
                break

        return self
