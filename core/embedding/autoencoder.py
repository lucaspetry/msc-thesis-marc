from ..logger import cur_date_time
from ..utils import MetricsLogger
from ..utils import compute_acc_acc5_f1_prec_rec

import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, attributes, vocab_size, embedding_size):
        super(Autoencoder,self).__init__()
        self.attributes = attributes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        if not isinstance(self.embedding_size, dict):
            self.embedding_size = {key: embedding_size for key in attributes}

        class_dim = sum(self.embedding_size.values())

        for idx, key in enumerate(attributes):
            if key == 'lat_lon':
                setattr(self, 'embed_' + key, nn.Linear(vocab_size[key],
                                                        self.embedding_size[key],
                                                        bias=False))
            else:
                setattr(self, 'embed_' + key, nn.Embedding(vocab_size[key],
                        self.embedding_size[key]))
            setattr(self, 'class_' + key, nn.Linear(class_dim, vocab_size[key]))

    def embedding_layer(self, attribute):
        return getattr(self, 'embed_' + attribute).weight.data.cpu()

    def encode(self, x):
        new_x = []
        
        for i, key in enumerate(self.attributes):
            embedder = getattr(self, 'embed_' + key)
            if key == 'lat_lon':
                new_x.append(embedder(x[:, i:].float()))
            else:
                new_x.append(embedder(x[:, i]))
            new_x[-1] = torch.sigmoid(new_x[-1])

        return torch.cat(new_x, dim=-1)

    def decode(self, x):
        output = []

        for i, key in enumerate(self.attributes):
            classifier = getattr(self, 'class_' + key)
            output.append(classifier(x))

        return output

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def fit(self, x, lrate=0.025, min_lrate=0.0001, epochs=100,
            batch_size=1000, patience=-1, threshold=0.001, log_file=None,
            cuda=False, verbose=False):
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
