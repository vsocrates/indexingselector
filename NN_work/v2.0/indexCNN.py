import os
import string

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib

tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim

from data_utils import data_load
from data_utils import get_batch
from indexCNN import IndexClassCNN

# # Data loading Parameters
# TRAIN_SET_PERCENTAGE = 0.9

# # Model Hyperparameters
# EMBEDDING_DIM = 200 # default 128, pretrained => 200
# FILTER_SIZES = "3,4,5"
# NUM_FILTERS= 128 # this is per filter size; default = 128
# L2_REG_LAMBDA=0.0 # L2 regularization lambda
# DROPOUT_KEEP_PROB=0.65

# # Training Parameters
# ALLOW_SOFT_PLACEMENT=False
# LOG_DEVICE_PLACEMENT=False
# NUM_CHECKPOINTS = 5 # default 5
# BATCH_SIZE = 64 # default 64
# NUM_EPOCHS = 10 # default 200
# EVALUATE_EVERY = 5 # Evaluate the model after this many steps on the test set; default 100
# CHECKPOINT_EVERY = 5 # Save the model after this many steps, every time
# PRETRAINED_W2V_PATH = "PubMed-and-PMC-w2v.bin"

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model


def train_CNN(train_dataset,
              test_dataset,
              vocab_processor,
              max_doc_length,
              model=None,
              ):
  
  train_init_op = iterator.make_initializer(train_dataset)
  test_init_op = iterator.make_initializer(test_dataset)
  sess.run(train_init_op)
  input_x, input_y = iterator.get_next()

  main_input = Input(tensor=input_x, name="main_input")
  

def get_word_to_vec_model(model_path, vocab_length):
  matrix_size = 50
  model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=matrix_size)
  print(model.vector_size)
  print(len(model.index2word))
  # store the embeddings in a numpy array
  
  # embedding_matrix = np.zeros((len(model.wv.vocab) + 1, EMBEDDING_DIM))
  embedding_matrix = np.zeros((vocab_length, EMBEDDING_DIM))
  # for i in range(len(model.wv.vocab)):
  max_size = min(len(model.index2word), vocab_length)
  for i in range(max_size):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  
  print(embedding_matrix[0:2])
  # have to add one for some reason? Maybe cuz its length?
  model_length = matrix_size + 1
  # free up the memory
  del(model)
  
  return embedding_matrix


def main(argv=None):
  # xml_file = "pubmed_result.xml"
  xml_file = "small_data.xml"
  # xml_file = "cits.xml"
  text_list = []

  train_dataset, test_dataset, vocab_processor, max_doc_length = data_load(xml_file, text_list, BATCH_SIZE, TRAIN_SET_PERCENTAGE)

  model = None
  if PRETRAINED_W2V_PATH:
    model = get_word_to_vec_model(PRETRAINED_W2V_PATH, len(vocab_processor.vocab))
    train_CNN(train_dataset,
              test_dataset,
              vocab_processor,
              max_doc_length,
              model=model,
              )
  else:
    train_CNN(train_dataset, test_dataset, vocab_processor, max_doc_length)    
    
if __name__ == '__main__':
  tf.app.run(main=main)
