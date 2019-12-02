import torch
import torch.nn as nn


class TULVAE(nn.Module):

    def __init__(self, input_size, embedding_size, rnn_cells, latent_dim,
                 output_size, dropout=0):
        super(TULVAE, self).__init__()
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.requires_grad = False
        
        self.rnn = nn.LSTM(input_size=embedding_size, batch_first=True,
                           hidden_size=rnn_cells, bidirectional=True)

        self.rnn_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_cells * 2, output_size)

        self.z_mean = nn.Linear(1, latent_dim)
        self.z_log_sigma = Dense(1, latent_dim)

        self.hidden_encoder = nn.Linear(latent_dim, rnn_cells)
        self.rnn_encoder = nn.LSTM(input_size=embedding_size, batch_first=True,
                                   hidden_size=rnn_cells, bidirectional=True)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)

    def init_embedding_layer(self, weights):
        self.embedding.weight.data = weights

    def forward_classifier(self, x, lengths=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)

        if lengths is not None:
            x = x[range(len(lengths)), lengths - 1]
        else:
            x = x[:, -1]
        x = self.rnn_dropout(x)
        x = self.classifier(x)
        return x

    def forward_encoder(x, x_label):
        x = self.embedding(x)
        x, _ = self.rnn(x)

        if lengths is not None:
            x = x[range(len(lengths)), lengths - 1]
        else:
            x = x[:, -1]
        x = self.rnn_dropout(x)
        x = torch.cat([x, x_label], dim=-1)

        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)

        def sampling(z_mean, z_log_sigma):
            epsilon = torch.normal([z_mean.size(0), self.latent_dim],
                                   mean=0., std=1.)
            return z_mean + torch.exp(z_log_sigma) * epsilon

        z = sampling(z_mean, z_log_sigma)
        z = self.hidden_encoder(z)
        z = torch.nn.functional.softplus(z)
        return z

    def forward_decoder(z, name):
        z = RepeatVector(max_traj_length)(z)
        decoder_h = LSTM(hidden_cells,
                         return_sequences=True)(z)
        decoder_mean = LSTM(EMBEDDING_SIZE, return_sequences=True)
        decoded = decoder_mean(decoder_h)
        return Dense(vocab_size,
                     name=name)(decoded)
        # TO-DO

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
