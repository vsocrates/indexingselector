import os
import string
from profilehooks import profile

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
from data_utils import Datasets
from conditional_decorator import conditional_decorator

# Data loading Parameters
TRAIN_SET_PERCENTAGE = 0.9
REMOVE_STOP_WORDS = True
WITH_AUX_INFO = True
MATRIX_SIZE = 50

# Model Hyperparameters
EMBEDDING_DIM = 200 # default 128, pretrained => 200
# FILTER_SIZES = "3,4,5"
# NUM_FILTERS= 128 # this is per filter size; default = 128
# L2_REG_LAMBDA=0.0 # L2 regularization lambda
# DROPOUT_KEEP_PROB=0.65

# # Training Parameters
ALLOW_SOFT_PLACEMENT=False
LOG_DEVICE_PLACEMENT=False
# NUM_CHECKPOINTS = 5 # default 5
BATCH_SIZE = 4 # default 64
NUM_EPOCHS = 2 # default 200
# EVALUATE_EVERY = 5 # Evaluate the model after this many steps on the test set; default 100
# CHECKPOINT_EVERY = 5 # Save the model after this many steps, every time
DEBUG = False
DO_TIMING_ANALYSIS = False # Make sure to change in data_utils too

# Data files
# xml_file = "../pubmed_result.xml"
# xml_file = "pubmed_result.xml"
# xml_file = "small_data.xml"
xml_file = "../small_data.xml"
# xml_file = "../cits.xml"
# xml_file = "pubmed_result_2012_2018.xml"
# PRETRAINED_W2V_PATH = "PubMed-and-PMC-w2v.bin"
PRETRAINED_W2V_PATH = "../PubMed-and-PMC-w2v.bin"

from tensorflow.python.keras.layers import Input, Embedding, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend

from tensorflow.python import debug as tf_debug
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import ProgbarLogger
from tensorflow.python.keras.optimizers import SGD

def train_CNN(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=None,
              ):

  # Model Hyperparameters
  filter_sizes = (2,4)#,5)
  num_filters = 5#100
  dropout_prob = (0.5, 0.8)
  hidden_dims = 50
       
  
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
              print("OutOfRangeError Exception Thrown")          
              break
            except Exception as e: 
              print(e)
              print("Unknown Exception Thrown")
              break

    train_batch_num = int((dataset_size*(TRAIN_SET_PERCENTAGE)) // BATCH_SIZE) + 1
    val_batch_num = int((dataset_size*(1-TRAIN_SET_PERCENTAGE)) // BATCH_SIZE)

    itr_train = make_iterator(datasets.abs_text_train_dataset, train_batch_num)
    itr_validate = make_iterator(datasets.abs_text_test_dataset, val_batch_num)
   
    main_input = Input(shape=(max_doc_lengths.abs_text_max_length,), dtype="int32", name="main_input")#, tensor=input_x)
    embedding_layer = Embedding(input_dim=len(vocab_processors['text'].vocab),
                                output_dim=EMBEDDING_DIM,
                                weights=[w2vmodel],
                                input_length=max_doc_lengths.abs_text_max_length,
                                trainable=False,
                                name="embedding")(main_input)

    dropout1 = Dropout(dropout_prob[0], name="dropout1")(embedding_layer)
    
    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
      conv_name = "conv1D-%s" % sz
      conv = Convolution1D(filters=num_filters,
                           kernel_size=sz,
                           padding="valid",
                           activation="relu",
                           strides=1,
                           name=conv_name)(dropout1)
      conv = MaxPooling1D(pool_size=2)(conv)
      conv = Flatten()(conv)
      conv_blocks.append(conv)
    conv_blocks_concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    dropout2 = Dropout(dropout_prob[1])(conv_blocks_concat)
    dense = Dense(hidden_dims, activation="relu")(dropout2)
    model_output = Dense(1, activation="sigmoid")(dense)

    # stochastic gradient descent algo, currently unused
    opt = SGD(lr=0.01)

    model = Model(inputs=main_input, outputs=model_output)
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
    train_CNN(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=model,
              )
  else:
    train_CNN(datasets, vocab_processors, max_doc_lengths, dataset_size)
      
      
if __name__ == '__main__':
  tf.app.run(main=main)
