# ----------------------------------------------------------------------------------------
# Step 1

# import gensim
# path='PubMed-and-PMC-w2v.txt'
# model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True) 
# model.save_word2vec_format('PubMed-and-PMC-w2v.txt', binary=False)

# ----------------------------------------------------------------------------------------
# Step 2 

# path='PubMed-and-PMC-w2v.txt'
# outpath='PubMed-and-PMC-w2v-50dim.txt'
# with open(path) as old, open(outpath, 'w') as new:
  # out = next(old)
  # vocab_size, dim = out.split()
  # new.write(vocab_size + " " + str(50) + "\n")
  # for line in old:
    # # print(line)
    # items = line.split(" ")
    # # print(items)
    # dims = items[1:51]
    # dim_string = " ".join(items[1:51])
    # # print(len(dims))
    # # print(dim_string)
    # out = items[0] + " " + dim_string + "\n"
    # # print("out", out)
    # new.write(out)
    # # break

# ----------------------------------------------------------------------------------------
# Step 3

# import gensim
# path='PubMed-and-PMC-w2v-50dim.txt'
# model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False) 
# model.save_word2vec_format('PubMed-and-PMC-w2v-50dim.bin', binary=True)

# ----------------------------------------------------------------------------------------
# Step 4 (test)

import gensim
path='PubMed-and-PMC-w2v-50dim.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True) 
vectors = model.wv
print(vectors['the'])