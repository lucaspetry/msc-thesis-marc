from .autoencoder import Autoencoder
from .gcbow import GroupedCBOW
from .icbow import IsolatedCBOW
from .pca import PCAEmbedder

__all__ = ['Autoencoder',
		   'GroupedCBOW',
		   'IsolatedCBOW',
           'PCAEmbedder']

def get_embedder(attributes, vocab_size, embedding_size, embedder_type, embedder_window=None):
	if embedder_type == 'gcbow':
		return GroupedCBOW(attributes=attributes, vocab_size=vocab_size,
                    	   embedding_size=embedding_size, window=embedder_window)
	elif embedder_type == 'icbow':
		return IsolatedCBOW(attributes=attributes, vocab_size=vocab_size,
                    	    embedding_size=embedding_size, window=embedder_window)
	elif embedder_type == 'autoencoder':
		return Autoencoder(attributes=attributes, vocab_size=vocab_size,
                           embedding_size=embedding_size)
	elif embedder_type == 'pca':
		return PCAEmbedder(attributes=attributes, vocab_size=vocab_size,
                           embedding_size=embedding_size)
	else:
		return None
