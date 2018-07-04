import gensim
path="../NN_work/PubMed-and-PMC-w2v.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True) 
model.save_word2vec_format('PubMed-and-PMC-w2v.txt', binary=False)
