import torch
import torch.nn as nn


class BiTULER(nn.Module):

    def __init__(self, input_size, embedding_size, rnn_cells, output_size,
                 dropout=0):
        super(BiTULER, self).__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.requires_grad = False

        self.rnn = nn.LSTM(input_size=embedding_size, batch_first=True,
                           hidden_size=rnn_cells, bidirectional=True)

        self.rnn_dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(rnn_cells * 2, output_size)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)

    def init_embedding_layer(self, weights):
        self.embedding.weight.data = weights

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)

        if lengths is not None:
            x = x[range(len(lengths)), lengths - 1]
        else:
            x = x[:, -1]
        x = self.rnn_dropout(x)
        x = self.classifier(x)
        return x
