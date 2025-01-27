import torch
import torch.nn as nn


class MARC(nn.Module):
    """Multiple-Aspect tRajectory Classifier (MARC).

    Parameters
    ----------
    attributes : list
        The names of the attributes describing the input data.
    vocab_size : dict
        Description.
    embedding_size : int or dict
        Description.
    rnn_cells : int
        Description.
    merge_type : str
        Description.
    rnn_cell : str (default='lstm')
        Description.
    dropout : float (default=0)
        Description.
    """

    def __init__(self, attributes, vocab_size, embedding_size, rnn_cells, output_size,
                 merge_type, rnn_cell='lstm', dropout=0, max_length=0):
        super(MARC, self).__init__()
        self.merge_type = merge_type
        self.dropout = dropout
        self.attributes = attributes
        self.embedding_size = embedding_size

        if not isinstance(self.embedding_size, dict):
            self.embedding_size = {key: embedding_size for key in attributes}

        for idx, key in enumerate(attributes):
            if key == 'lat_lon':
                setattr(self, 'embed_' + key, nn.Linear(vocab_size[key],
                        self.embedding_size[key], bias=False))
            else:
                setattr(self, 'embed_' + key, nn.Embedding(vocab_size[key],
                        self.embedding_size[key]))
        
        self.hidden_dropout = nn.Dropout(dropout)
        self.rnn_input = sum(self.embedding_size.values())

        if self.merge_type != 'concatenate':
            self.rnn_input = self.embedding_size.values()[0]

        if rnn_cell == 'lstm':
            self.rnn = nn.LSTM(input_size=self.rnn_input,
                               batch_first=True, hidden_size=rnn_cells)
        else:
            self.rnn = nn.GRU(input_size=self.rnn_input,
                              batch_first=True, hidden_size=rnn_cells)
        self.rnn_mean_weights = nn.Parameter(torch.Tensor(max_length).fill_(1.0))
        self.rnn_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_cells, output_size)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)

    def init_embedding_layer(self, weight_dict, trainable=True):
        for idx, key in enumerate(self.attributes):
            layer = getattr(self, 'embed_' + key)
            layer.weight.data = weight_dict[key]
            layer.weight.requires_grad = trainable
 
    def embed(self, x):
        new_x = []

        for i, key in enumerate(self.attributes):
            embedder = getattr(self, 'embed_' + key)
            if key == 'lat_lon':
                new_x.append(embedder(x[:, :, i:].float()))
            else:
                new_x.append(embedder(x[:, :, i]))
        x = new_x

        x = torch.cat(x, dim=-1)
        
        if self.merge_type == 'add':
            x = x.view(x.size(0), x.size(1),
                       len(self.attributes), self.embedding_size).sum(dim=2)
        elif self.merge_type == 'average':
            x = x.view(x.size(0), x.size(1),
                       len(self.attributes), self.embedding_size).mean(dim=2)

        return x

    def forward(self, x, lengths):
        x = self.embed(x)
        x = self.hidden_dropout(x)
        x, _ = self.rnn(x)

        # x = x[range(len(lengths)), lengths - 1]

        idx = (lengths - 1) // 3
        x = torch.cat([x[range(len(idx)), idx],
                       x[range(len(idx)), idx * 2],
                       x[range(len(idx)), lengths - 1]])

        # x = x.view(3, x.size(0) // 3, -1).mean(dim=0)
        x = x.view(3, x.size(0) // 3, -1)
        x = x * self.rnn_mean_weights[:3].expand([x.size(1), x.size(2), x.size(0)]).permute(2, 0, 1)
        x = x.mean(dim=0)

        # x_mean = torch.zeros(x.size(0), x.size(2))
        
        # for i, sample in enumerate(x):
        #     x_mean[i] = (sample[:lengths[i]].t() @ self.rnn_mean_weights[:lengths[i]]) / lengths[i]

        # if x.is_cuda:
        #     x = x_mean.cuda()

        x = self.rnn_dropout(x)
        x = self.classifier(x)
        return x.log_softmax(dim=-1)
