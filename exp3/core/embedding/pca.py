import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


class PCAEmbedder:

    def __init__(self, attributes, vocab_size, embedding_size):
        self.attributes = attributes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_idxs = {}
        count = 0

        for attr in self.attributes:
            self.embedding_idxs[attr] = count
            count += self.vocab_size[attr]

        if not isinstance(self.embedding_size, dict):
            self.embedding_size = {key: embedding_size for key in attributes}

        categories = [range(self.vocab_size[a]) for a in self.attributes]
        self.one_hot_encoder = OneHotEncoder(categories)
        self.pca = PCA(n_components=sum(embedding_size.values()))

    def embedding_layer(self, attribute):
        return self.pca.components_.T

    def fit(self, x):
        flat_x = x.view(x.size(0) * x.size(1), -1)
        flat_x = flat_x[flat_x.nonzero()]

        one_hot_x = self.one_hot_encoder.transform(flat_x)
        self.pca.fit(one_hot_x)
        return self
