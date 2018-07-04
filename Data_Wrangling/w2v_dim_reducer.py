import gensim
path="../"
gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=matrix_size, size=50)