from ..logger import cur_date_time
from ..utils import MetricsLogger
from ..utils import compute_acc_acc5_f1_prec_rec

import torch
import torch.nn as nn


class GroupedCBOW(nn.Module):

    def __init__(self, attributes, vocab_size, embedding_size=100, window=5,
                 negative_sampling=5):
        super(GroupedCBOW, self).__init__()
        self.attributes = attributes
        self.embedding_size = embedding_size
        self.window = window
        self.negative_sampling = negative_sampling
        
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

    def embed(self, x, lens):
        new_x = []
        beg = (lens < 2 * self.window).sum()
        
        for i, key in enumerate(self.attributes):
            embedder = getattr(self, 'embed_' + key)
            if key == 'lat_lon':
                new_x.append(embedder(x[:, :, i:].float()))
            else:
                new_x.append(embedder(x[:, :, i]))

        x = torch.cat(new_x, dim=-1)

        # To get around shorter context windows
        x_mean = torch.zeros(beg, x.size(2))
        for i, sample in enumerate(x[:beg]):
            x_mean[i] = sample[:lens[i]].mean(dim=0)

        if x.is_cuda:
            x_mean = x_mean.cuda()

        x = torch.cat([x_mean, x[beg:].mean(dim=1)])

        return x

    def classify(self, x):
        output = []

        for i, key in enumerate(self.attributes):
            classifier = getattr(self, 'class_' + key)
            output.append(classifier(x))

        return output

    def forward(self, x, lens):
        x = self.embed(x, lens)
        x = self.classify(x)
        return x

    def fit(self, x, lrate=0.025, min_lrate=0.0001, epochs=100,
            batch_size=1000, patience=-1, threshold=0.001, log_file=None,
            cuda=False, n_jobs=-1, verbose=False):
        model_logger = MetricsLogger(keys=['epoch', 'train_loss'], timestamp=True)

        train_x, train_y, lengths = self._prepare_training_data(x)

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
                lens = lengths[batch]
                x = train_x[batch][lens.argsort()].long()
                y = torch.Tensor(train_y[batch])[lens.argsort()].long()
                lens = lens[lens.argsort()].long()

                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                    lens = lens.cuda()

                pred_y = self(x, lens)
                loss = 0

                for i, pred in enumerate(pred_y):
                    loss += loss_func(pred, y[:, i])

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

    def _prepare_training_data(self, x):
        """
        Parameters
        ----------
        x : array-like of shape (n_samples, max_length, n_features)
            Description.
        """
        new_x, new_y, lengths = [], [], []

        for sample in x:
            for i in range(0, len(sample)):
                input_sample = []
                input_sample.extend(sample[i-self.window:i])
                input_sample.extend(sample[i+1:i+1+self.window])
                missed = 2 * self.window - len(input_sample)

                lengths.append(len(input_sample))

                if missed == self.window:
                    lengths[-1] = 2 * self.window
                    input_sample.extend(input_sample)
                elif missed > 0:
                    input_sample.extend(torch.zeros(missed, len(sample[i])).int())
                new_x.append(input_sample)
                new_y.append(sample[i])

        return torch.Tensor(new_x), torch.Tensor(new_y), torch.Tensor(lengths)
