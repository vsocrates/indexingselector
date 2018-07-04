# Vanilla python
import os
import string
from profilehooks import profile

# Numpy
import numpy as np

# Tensorflow
import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorflow.python import debug as tf_debug
tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

# Gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# own modules
from data_utils import data_load
from conditional_decorator import conditional_decorator

# Keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras import backend

from keras.callbacks import CSVLogger
from keras.callbacks import ProgbarLogger
from keras.optimizers import SGD

def train_LSTM(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=None,
              ):
              
  session_conf = tf.ConfigProto(
          allow_soft_placement=ALLOW_SOFT_PLACEMENT, # determines if op can be placed on CPU when GPU not avail
          log_device_placement=LOG_DEVICE_PLACEMENT, # whether device placements should be logged, we don't have any for CPU
          #operation_timeout_in_ms=60000
          )
  session_conf.gpu_options.allow_growth = True
  sess = tf.Session(config=session_conf)

  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  backend.set_session(sess)
  
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  
  with sess.as_default():
  
    def make_iterator(dataset, batch_num):
        while True:
          iterator = dataset.make_one_shot_iterator()
          next_val = iterator.get_next()
          for i in range(batch_num):
            try:
              *inputs, labels = sess.run(next_val)
              yield inputs, labels  
            except tf.errors.OutOfRangeError:
              if DEBUG:
                print("OutOfRangeError Exception Thrown")          
              break
            except Exception as e: 
              if DEBUG:
                print(e)
                print("Unknown Exception Thrown")
              break

  train_batch_num = int((dataset_size*(TRAIN_SET_PERCENTAGE)) // BATCH_SIZE) + 1
  val_batch_num = int((dataset_size*(1-TRAIN_SET_PERCENTAGE)) // BATCH_SIZE)

  itr_train = make_iterator(datasets.abs_text_train_dataset, train_batch_num)
  itr_validate = make_iterator(datasets.abs_text_test_dataset, val_batch_num)
 
  main_input = Input(shape=(max_doc_lengths.abs_text_max_length,), dtype="int32", name="main_input")
  embedding_layer = Embedding(input_dim=len(vocab_processors['text'].vocab),
                              output_dim=EMBEDDING_DIM,
                              weights=[w2vmodel],
                              input_length=max_doc_lengths.abs_text_max_length,
                              trainable=False,
                              name="embedding")(main_input)
  dropout1 = Dropout(0.2, name="dropout1")(embedding_layer)
  lstm_out = LSTM(64)(dropout1)
  dropout2 = Dropout(0.2, name="dropout2")(lstm_out)
  auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(dropout2)

  
  # stochastic gradient descent algo, currently unused
  opt = SGD(lr=0.01)

  model = Model(inputs=main_input, outputs=auxiliary_output)
  model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
  # model._make_predict_function()
                # will be useful when we actually combine
                # loss_weights=[1., 0.2]
  callbacks = []
  verbosity = 2
  if DEBUG:
    callbacks.append(CSVLogger('training.log'))
    callbacks.append(ProgbarLogger(count_mode='steps'))
    verbosity = 1
  print(model.summary())

  model.fit_generator(generator=itr_train,
                      validation_data=itr_validate,
                      validation_steps=val_batch_num,
                      steps_per_epoch=train_batch_num,
                      epochs=NUM_EPOCHS,
                      verbose=verbosity,
                      workers=0,
                      callbacks=callbacks)
                      
@conditional_decorator(profile, DO_TIMING_ANALYSIS)            
def get_word_to_vec_model(model_path, vocab_proc, vocab_proc_tag):
  vocab = vocab_proc[vocab_proc_tag].vocab
  matrix_size = MATRIX_SIZE
  
  model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=matrix_size)
  print("Embedding Dims: ", model.vector_size)
  print("Number of Tokens in Model: ", len(model.index2word))
  # store the embeddings in a numpy array
  
  # embedding_matrix = np.zeros((len(model.wv.vocab) + 1, EMBEDDING_DIM))
  embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
  # for i in range(len(model.wv.vocab)):
  max_size = min(len(model.index2word), len(vocab))

  for word, idx in vocab.items():
    if word in model.wv:
      embedding_vector = model.wv[word]
      if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
    else:
    # I'm pretty sure something is supposed to happen here but idk what
      pass
    
  # # free up the memory
  del(model)
  return embedding_matrix

def main(argv=None):
  text_list = []

  datasets, vocab_processors, max_doc_lengths, dataset_size = data_load(xml_file, text_list, BATCH_SIZE, TRAIN_SET_PERCENTAGE, REMOVE_STOP_WORDS, with_aux_info=WITH_AUX_INFO)

  model = None
  if PRETRAINED_W2V_PATH:
    model = get_word_to_vec_model(PRETRAINED_W2V_PATH, vocab_processors, "text")
    train_LSTM(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=model,
              )
  else:
    train_LSTM(datasets, vocab_processors, max_doc_lengths, dataset_size)

def parse_arguments():
  # Data loading Parameters
  global XML_FILE#  = "pubmed_result.xml"
  global PRETRAINED_W2V_PATH# = "PubMed-and-PMC-w2v.bin"
  global WITH_AUX_INFO
  global MATRIX_SIZE#  = 9000

  # Model Hyperparameters
  global REMOVE_STOP_WORDS
  global TRAIN_SET_PERCENTAGE#  = 0.9
  
  global EMBEDDING_DIM # default 128, pretrained => 200 # not currently set
  global BATCH_SIZE 
  global NUM_EPOCHS 
  global FILTER_SIZES
  global NUM_FILTERS # this is per filter size; default = 128
  # L2_REG_LAMBDA=0.0 # L2 regularization lambda
  global DROPOUT_KEEP_PROB
  global HIDDEN_DIMS
  
  # Stdout params
  global DEBUG
  global DO_TIMING_ANALYSIS # Make sure to change in data_utils too
       

  global ALLOW_SOFT_PLACEMENT
  ALLOW_SOFT_PLACEMENT=False
  global LOG_DEVICE_PLACEMENT
  LOG_DEVICE_PLACEMENT=False
       


  def restricted_float(x):
      x = float(x)
      if x < 0.0 or x > 1.0:
          raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
      return x

  parser = argparse.ArgumentParser()

  # Data loading params
  parser.add_argument("-f", "--data-file", help="location of data file", required=True)
  parser.add_argument("-w", "--w2v-path", help="location of pre-trained w2v model file")
  parser.add_argument("-x","--get-aux-info",help="retrieve the auxiliary information from the data file", action="store_true")
  parser.add_argument("-ws", "--word2vec-size", help="get the first N words from pre-trained word2vec model", type=int, default=200)

  # Model hyperparams  
  parser.add_argument("-sw" "--remove-stop-words", help="flag to remove stop words from abstracts", action="store_true")
  parser.add_argument("-t", "--train-percentage", help="percentage of the dataset to train from 0 to 1", type=restricted_float, default=0.9)
  
  parser.add_argument("-l", "--embedding-dim", help="dimensionality of the learned word embeddings", type=int, default=200)
  parser.add_argument("-b", "--batch-size", help="set the batch size", type=int, default=64)
  parser.add_argument("-e", "--num-epochs", help="the number of epochs to train", type=int, default=200)
  parser.add_argument("-fs", "--filter-sizes", help='list of filter sizes [e.g. "(2,3,4)"]', default='"(2,4,5)"')
  parser.add_argument("-n", "--num-filters", help="number of filters per size", type=int, default=100)
  parser.add_argument("-dp", "--dropout-prob", help='probability of dropout for 2 layers [e.g. "(0.5, 0.8)"]', default='"(0.5, 0.8)"')
  parser.add_argument("-hd", "--hidden-dims", help="number of hidden nodes in the last dense layer", type=int, default=50)

  # Stdout params
  parser.add_argument("-d", "--debug", help="sets the debug flag providing extra output", action="store_true")
  parser.add_argument("-i", "--timing-analysis", help="perform and output timing on certain methods", action="store_true")
  
  arguments = parser.parse_args()
  print(arguments)
  XML_FILE = arguments.data_file
  PRETRAINED_W2V_PATH = arguments.w2v_path
  WITH_AUX_INFO = arguments.get_aux_info
  MATRIX_SIZE = arguments.word2vec_size

  # Model Hyperparameters
  REMOVE_STOP_WORDS = arguments.sw__remove_stop_words
  TRAIN_SET_PERCENTAGE = arguments.train_percentage
  
  EMBEDDING_DIM = arguments.embedding_dim # default 128, pretrained => 200 # not currently set
  BATCH_SIZE = arguments.batch_size
  NUM_EPOCHS = arguments.num_epochs
  
  filter_size_string = arguments.filter_sizes
  filters = re.findall('\d+', filter_size_string)
  FILTER_SIZES = [int(f) for f in filters]
  print("FILTER_SIZES", FILTER_SIZES)
  NUM_FILTERS = arguments.num_filters# this is per filter size; default = 128
  # L2_REG_LAMBDA=0.0 # L2 regularization lambda
  dropout_prob_string = arguments.dropout_prob
  
  dropouts = re.findall(r"[-+]?\d*\.\d+|\d+", dropout_prob_string)
  DROPOUT_KEEP_PROB = [float(d) for d in dropouts]
  print("DROPOUT_KEEP_PROB", DROPOUT_KEEP_PROB)
  HIDDEN_DIMS = arguments.hidden_dims
  
  # Stdout params
  DEBUG = arguments.debug
  DO_TIMING_ANALYSIS = arguments.timing_analysis # Make sure to change in data_utils too
      
if __name__ == '__main__':
  parse_arguments()
  tf.app.run(main=main)
