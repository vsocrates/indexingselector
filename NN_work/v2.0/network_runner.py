# Vanilla Python
import os
import string
from datetime import datetime
import re
import argparse
from profilehooks import profile
import time 

# Numpy
import numpy as np

# Tensorflow
import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorflow.python import debug as tf_debug
tf.logging.set_verbosity(tf.logging.INFO)
# print(tf.__version__)

# Keras
from keras.layers import Input, Embedding, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten, Concatenate
from keras.models import Model
from keras import backend

from keras.callbacks import CSVLogger
from keras.callbacks import ProgbarLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD
from keras.optimizers import Adam

# gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# own modules 
import globals 

from data_utils import data_load
from data_utils import get_word_to_vec_model
from data_utils import Datasets
from conditional_decorator import conditional_decorator
from indexCNN import train_CNN
from indexLSTM import train_LSTM
from indexCNNAux import train_CNNAux
from indexLSTMAux import train_LSTMAux

DO_TIMING_ANALYSIS = False

def main(argv=None):  
  text_list = []

  # The data_load method located in the data_utils.py does the brunt of the work and it starts here. 
  # This includes pulling data from the XML, preprocessing the text (tokenize, lowercase etc.), and creating tf.Datasets
  if globals.POS_XML_FILE:
    aug_text_list = []
    datasets, vocab_processors, max_doc_lengths, dataset_size = data_load(globals.XML_FILE, text_list, globals.BATCH_SIZE, globals.REMOVE_STOP_WORDS, globals.SHOULD_STEM, globals.LIMIT_VOCAB, globals.MAX_VOCAB_SIZE, globals.TRAIN_SET_PERCENTAGE, with_aux_info=globals.WITH_AUX_INFO, pos_text_list=aug_text_list, test_date=globals.SPLIT_WITH_DATE)
  else:
    datasets, vocab_processors, max_doc_lengths, dataset_size = data_load(globals.XML_FILE, text_list, globals.BATCH_SIZE, globals.REMOVE_STOP_WORDS, globals.SHOULD_STEM, globals.LIMIT_VOCAB, globals.MAX_VOCAB_SIZE, globals.TRAIN_SET_PERCENTAGE, with_aux_info=globals.WITH_AUX_INFO, pos_text_list=[], test_date=globals.SPLIT_WITH_DATE)
    
  # Gets a pretrained w2v model if we want it.
  model_list = {}
  if globals.PRETRAINED_W2V_PATH:
    model_list['text'] = get_word_to_vec_model(globals.PRETRAINED_W2V_PATH, globals.MATRIX_SIZE, vocab_processors, "text")
  # Creates one for all the other text categories
  if globals.WITH_AUX_INFO:
    model_list['affiliations'] = get_word_to_vec_model(globals.PRETRAINED_W2V_PATH, globals.MATRIX_SIZE, vocab_processors, "affiliations")
    model_list['keywords'] = get_word_to_vec_model(globals.PRETRAINED_W2V_PATH, globals.MATRIX_SIZE, vocab_processors, "keywords")
    model_list['article_title'] = get_word_to_vec_model(globals.PRETRAINED_W2V_PATH, globals.MATRIX_SIZE, vocab_processors, "article_title")
    
  # Run one of four types of models. "Aux" means that it will include at least one of the following: affiliations, journal title, article title, keywords with the abstract text.
  if globals.MODEL_TYPE == 'CNN':
    train_CNN(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=model_list['text'],
              )
  elif globals.MODEL_TYPE == "CNNAux":
    train_CNNAux(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=model_list,
              )
  elif globals.MODEL_TYPE == "LSTM":
    train_LSTM(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=model_list['text'],
              )
  elif globals.MODEL_TYPE == "LSTMAux":
    train_LSTMAux(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=model_list,
              )           
  else:
    print("I don't know how this happened, should be impossible")

"""Parse all input parameters and save to globals."""
def parse_arguments():

  def restricted_float(x):
      x = float(x)
      if x < 0.0 or x > 1.0:
          raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
      return x

  def valid_date(s):
      if s == "":
        return
      try:
          return datetime.strptime(s, "%Y-%m-%d").date()
      except ValueError:
          msg = "Not a valid date: '{0}'.".format(s)
          raise argparse.ArgumentTypeError(msg)
      
      
  parser = argparse.ArgumentParser()

  # Data loading params
  parser.add_argument("-f", "--data-file", help="location of data file", required=True)
  parser.add_argument("-pd", "--pos-data-file", help="location of fully indexed article file", default="")
  parser.add_argument("-w", "--w2v-path", help="location of pre-trained w2v model file")
  parser.add_argument("-x","--get-aux-info",help="retrieve the auxiliary information from the data file", action="store_true")
  parser.add_argument("-v", "--word2vec-size", help="get the first N words from pre-trained word2vec model")
  parser.add_argument("-c", "--no-limit-vocab", help="DON'T limit the size of the vocab (default true)", action="store_false")
  parser.add_argument("-j", "--max-vocab-size", help="get the first N words from pre-trained word2vec model", type=int, default=80000)
  parser.add_argument("-lw", "--lower-vocab", help="make vocab lowercase", action="store_true")
  parser.add_argument("-sd", "--split-by-date", help="Split test/train by date. Overrides percentage selection - format YYYY-MM-DD", type=valid_date, default="")
  parser.add_argument("-rn", "--run-number", help="Don't set this yourself, it will be set in slurm scripts", required=True)
  parser.add_argument("-ds", "--downsample", help="Downsample data to match pos and neg examples", action="store_true")
  
  # Common Model hyperparameters  
  parser.add_argument("-tv", "--same-vocab", help="Make the title and abstract use the same vocabulary", action="store_true")
  parser.add_argument("-at", "--aux-trainable", help="Auxiliary information trainable", action="store_true")
  parser.add_argument("-y", "--model-type", help="Which type of model to use", required=True, choices=['CNN', 'CNNAux', 'LSTM', 'LSTMAux'])
  parser.add_argument("-o", "--remove-stop-words", help="flag to remove stop words and punctuation from abstracts", action="store_true")
  parser.add_argument("-t", "--stem-words", help="flag to stem words in abstracts", action="store_true")
  parser.add_argument("-p", "--train-percentage", help="percentage of the dataset to train from 0 to 1", type=restricted_float, default=0.9)
  
  parser.add_argument("-l", "--embedding-dim", help="dimensionality of the learned word embeddings", type=int, default=200)
  parser.add_argument("-q", "--embedding-trainable", help="make the embedding trainable", action="store_true")
  parser.add_argument("-b", "--batch-size", help="set the batch size", type=int, default=64)
  parser.add_argument("-e", "--num-epochs", help="the number of epochs to train", type=int, default=200)
  parser.add_argument("-i", "--init-learning-rate", help="Initial Learning Rate", type=float, default=0.001)
  parser.add_argument("-k", "--main-dropout-prob", help='probability of dropout for MAIN 2 layers [e.g. "(0.5, 0.8)"]', default='"(0.5, 0.8)"')
  parser.add_argument("-r", "--aux-dropout-prob", help='probability of dropout for AUX 2 layers [e.g. "(0.5, 0.8)"]', default='"(0.5, 0.8)"')
  
  # LSTM Hyperparameters
  parser.add_argument("-m", "--main-lstm-dim", help="dimensionality of the MAIN lstm layer", type=int, default=64)
  parser.add_argument("-u", "--aux-lstm-dim", help="dimensionality of the AUX lstm layer", type=int, default=64) 
  parser.add_argument("-s", "--dense-dim", help="dimensionality combined data dense layer", type=int, default=64) 

  # CNN Hyperparameters 
  parser.add_argument("-g", "--hidden-dims", help="number of hidden nodes in the last dense layer", type=int, default=50)
  parser.add_argument("-z", "--filter-sizes", help='list of filter sizes [e.g. "(2,3,4)"]', default='"(2,4,5)"')
  parser.add_argument("-n", "--num-filters", help="number of filters per size", type=int, default=100)

  # Stdout params
  parser.add_argument("-d", "--debug", help="sets the debug flag providing extra output", action="store_true")
  parser.add_argument("-a", "--save", help="saves the model after training", action="store_true")

  arguments = parser.parse_args()
  globals.XML_FILE = arguments.data_file
  globals.RUN_NUMBER = arguments.run_number
  globals.POS_XML_FILE = arguments.pos_data_file
  globals.DOWNSAMPLE_TO_MATCH = arguments.downsample
  globals.WITH_AUX_INFO = arguments.get_aux_info
  globals.PRETRAINED_W2V_PATH = arguments.w2v_path

  try:
    w2v_size = int(arguments.word2vec_size)
  except TypeError:
    w2v_size = None
  globals.MATRIX_SIZE = w2v_size 
  print("globals.MATRIX_SIZE" , globals.MATRIX_SIZE)
  globals.LIMIT_VOCAB = arguments.no_limit_vocab
  globals.MAX_VOCAB_SIZE = arguments.max_vocab_size
  globals.VOCAB_LOWERCASE = arguments.lower_vocab
  globals.SPLIT_WITH_DATE = False
  
  # Common Model Hyperparameters
  globals.SAME_VOCAB = arguments.same_vocab
  globals.AUX_TRAINABLE = arguments.aux_trainable
  globals.MODEL_TYPE = arguments.model_type
  if globals.MODEL_TYPE.find("Aux"):
    globals.WITH_AUX_INFO = True
  globals.REMOVE_STOP_WORDS = arguments.remove_stop_words
  globals.SHOULD_STEM = arguments.stem_words
  globals.TRAIN_SET_PERCENTAGE = arguments.train_percentage
  globals.SPLIT_WITH_DATE = arguments.split_by_date
  if globals.SPLIT_WITH_DATE:
    globals.TRAIN_SET_PERCENTAGE = 0.0
    
  globals.EMBEDDING_DIM = arguments.embedding_dim # default 128, pretrained => 200 # not currently set
  globals.EMBEDDING_TRAINABLE = arguments.embedding_trainable
  globals.BATCH_SIZE = arguments.batch_size
  globals.NUM_EPOCHS = arguments.num_epochs
  
  dropout_prob_string = arguments.main_dropout_prob
  dropouts = re.findall(r"[-+]?\d*\.\d+|\d+", dropout_prob_string)
  globals.MAIN_DROPOUT_KEEP_PROB = [float(d) for d in dropouts]

  dropout_prob_string = arguments.aux_dropout_prob
  dropouts = re.findall(r"[-+]?\d*\.\d+|\d+", dropout_prob_string)
  globals.AUX_DROPOUT_KEEP_PROB = [float(d) for d in dropouts]

  globals.LEARNING_RATE = arguments.init_learning_rate
  
  # CNN Hyperparameters
  filter_size_string = arguments.filter_sizes
  filters = re.findall('\d+', filter_size_string)
  globals.FILTER_SIZES = [int(f) for f in filters]
  globals.NUM_FILTERS = arguments.num_filters# this is per filter size; default = 128
  globals.HIDDEN_DIMS = arguments.hidden_dims

  
  # LSTM Hyperparameters
  # L2_REG_LAMBDA=0.0 # L2 regularization lambda
  globals.MAIN_LSTM_SIZE = arguments.main_lstm_dim
  globals.AUX_LSTM_SIZE = arguments.aux_lstm_dim
  globals.COMBI_DENSE_LAYER_DIM = arguments.dense_dim

  
  # Stdout params
  globals.DEBUG = arguments.debug
  globals.SAVE_MODEL = arguments.save
  
  print("\n")
  print("All network parameters: ", arguments)
  print("\n\n")

if __name__ == '__main__':
  parse_arguments()
  tf.app.run(main=main)
