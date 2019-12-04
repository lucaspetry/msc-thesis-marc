from .autoencoder import Autoencoder
from .cbow import CBOW
from .pca import PCAEmbedder

__all__ = ['Autoencoder',
		   'CBOW',
           'PCAEmbedder']

def get_embedder(attributes, vocab_size, embedding_size, embedder_type):
	if embedder_type == 'word2vec':
		return CBOW(attributes=attributes, vocab_size=vocab_size,
                    embedding_size=embedding_size, window=3)
	elif embedder_type == 'autoencoder':
		return Autoencoder(attributes=attributes, vocab_size=vocab_size,
                           embedding_size=embedding_size)
	elif embedder_type == 'pca':
		return PCAEmbedder(attributes=attributes, vocab_size=vocab_size,
                           embedding_size=embedding_size)
	else:
		return None
