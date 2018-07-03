# Vanilla Python
import os
import string
from profilehooks import profile

# Numpy
import numpy as np

# Tensorflow
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib
tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)
from tensorflow.python import debug as tf_debug

# Gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# Own Modules
from data_utils import data_load
from conditional_decorator import conditional_decorator

# Keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from keras.models import Model
from keras import backend

from keras.callbacks import CSVLogger
from keras.callbacks import ProgbarLogger
from keras.optimizers import SGD


# Data loading Parameters
TRAIN_SET_PERCENTAGE = 0.9
REMOVE_STOP_WORDS = True
WITH_AUX_INFO = True
MATRIX_SIZE = 9000

# Model Hyperparameters
EMBEDDING_DIM = 200 # default 128, pretrained => 200
# L2_REG_LAMBDA=0.0 # L2 regularization lambda
# DROPOUT_KEEP_PROB=0.65

# # Training Parameters
ALLOW_SOFT_PLACEMENT=False
LOG_DEVICE_PLACEMENT=False
# NUM_CHECKPOINTS = 5 # default 5
BATCH_SIZE = 64 # default 64
NUM_EPOCHS = 2 # default 200
# EVALUATE_EVERY = 5 # Evaluate the model after this many steps on the test set; default 100
# CHECKPOINT_EVERY = 5 # Save the model after this many steps, every time
DEBUG = False
DO_TIMING_ANALYSIS = False # Make sure to change in data_utils too

# Data files
# xml_file = "../pubmed_result.xml"
xml_file = "pubmed_result.xml"
# xml_file = "small_data.xml"
# xml_file = "../small_data.xml"
# xml_file = "../cits.xml"
# xml_file = "pubmed_result_2012_2018.xml"
PRETRAINED_W2V_PATH = "PubMed-and-PMC-w2v.bin"
# PRETRAINED_W2V_PATH = "../PubMed-and-PMC-w2v.bin"

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
  
    def make_multiple_iterator(dataset_list, batch_num):
        while True:
          itr_list = []
          next_val_list = []
          for dataset in dataset_list:
            iterator = dataset.make_one_shot_iterator()
            itr_list.append(iterator)
            next_val_list.append(iterator.get_next())
          # iterator = dataset.make_one_shot_iterator()
          # next_val = iterator.get_next() 
          for i in range(batch_num):
            value_list  = []
            labels_out = []
            for vals in next_val_list:
              try:
                *inputs, labels = sess.run(vals)
                value_list.append(inputs[0])
                labels_out.append(labels)
                # yield inputs, labels  
              except tf.errors.OutOfRangeError:
                if DEBUG:
                  print("OutOfRangeError Exception Thrown")          
                break
              except Exception as e: 
                if DEBUG:
                  print(e)
                  print("Unknown Exception Thrown")
                break
            yield value_list, labels_out
            
  train_batch_num = int((dataset_size*(TRAIN_SET_PERCENTAGE)) // BATCH_SIZE) + 1
  val_batch_num = int((dataset_size*(1-TRAIN_SET_PERCENTAGE)) // BATCH_SIZE)
  
  itr_train_abs = make_multiple_iterator([datasets.abs_text_train_dataset,datasets.affl_train_dataset], train_batch_num)
  itr_validate_abs = make_multiple_iterator([datasets.abs_text_test_dataset,datasets.affl_test_dataset], val_batch_num)
  
 
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

  # auxiliary information
  aux_input = Input(shape=(max_doc_lengths.affl_max_length,), dtype="int32", name="affl_input")
  affl_embedding_layer = Embedding(input_dim=len(vocab_processors['text'].vocab),
                              output_dim=EMBEDDING_DIM,
                              input_length=max_doc_lengths.abs_text_max_length,
                              name="affl_embedding")(aux_input)
  dropout3 = Dropout(0.2, name="dropout3")(affl_embedding_layer)
  aux_lstm_out = LSTM(64)(dropout3)
  dropout4 = Dropout(0.2, name="dropout4")(aux_lstm_out)
  
  concat = Concatenate()([dropout4, dropout2])
  x = Dense(64, activation='relu')(concat)
  x = Dense(64, activation='relu')(x)
  x = Dense(64, activation='relu')(x)

  # And finally we add the main logistic regression layer
  main_output = Dense(1, activation='sigmoid', name='main_output')(x)
  
  # stochastic gradient descent algo, currently unused
  opt = SGD(lr=0.01)

  model = Model(inputs=[main_input,aux_input] , outputs=[main_output, auxiliary_output])
  model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
  # model._make_predict_function()
                # will be useful when we actually combine
                # loss_weights=[1., 0.2]
  callbacks = []
  verbosity = 2
  if DEBUG:
    callbacks.append(CSVLogger('training.log'))
    # callbacks.append(ProgbarLogger(count_mode='steps'))
    verbosity = 1
  print(model.summary())

  model.fit_generator(generator=itr_train_abs,
                      validation_data=itr_validate_abs,
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
      
      
if __name__ == '__main__':
  tf.app.run(main=main)
