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

# # Data loading Parameters
TRAIN_SET_PERCENTAGE = 0.9

# # Model Hyperparameters
EMBEDDING_DIM = 200 # default 128, pretrained => 200
# FILTER_SIZES = "3,4,5"
# NUM_FILTERS= 128 # this is per filter size; default = 128
# L2_REG_LAMBDA=0.0 # L2 regularization lambda
# DROPOUT_KEEP_PROB=0.65

# # Training Parameters
# ALLOW_SOFT_PLACEMENT=False
# LOG_DEVICE_PLACEMENT=False
# NUM_CHECKPOINTS = 5 # default 5
BATCH_SIZE = 32 # default 64
NUM_EPOCHS = 50 # default 200
# EVALUATE_EVERY = 5 # Evaluate the model after this many steps on the test set; default 100
# CHECKPOINT_EVERY = 5 # Save the model after this many steps, every time
PRETRAINED_W2V_PATH = "../PubMed-and-PMC-w2v.bin"

from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend

from tensorflow.python import debug as tf_debug
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import ProgbarLogger
from tensorflow.python.keras.optimizers import SGD

def train_CNN(train_dataset,
              test_dataset,
              vocab_processor,
              max_doc_length,
              dataset_size,
              w2vmodel=None,
              ):
  
  sess = tf.Session()
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

    print(train_batch_num)
    
    itr_train = make_iterator(train_dataset, train_batch_num)
    itr_validate = make_iterator(test_dataset, val_batch_num)
   
    main_input = Input(shape=(max_doc_length,), dtype="int32", name="main_input")#, tensor=input_x)
    embedding_layer = Embedding(input_dim=len(vocab_processor.vocab),
                                output_dim=EMBEDDING_DIM,
                                weights=[w2vmodel],
                                input_length=max_doc_length,
                                trainable=False,
                                name="embedding")(main_input)
    lstm_out = LSTM(32)(embedding_layer)
    auxiliary_output = Dense(1, activation='relu', name='aux_output')(lstm_out)
    opt = SGD(lr=0.01)

    model = Model(inputs=main_input, outputs=auxiliary_output)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # model._make_predict_function()
                  # will be useful when we actually combine
                  # loss_weights=[1., 0.2]
    csv_logger = CSVLogger('training.log')
    progbar = ProgbarLogger(count_mode='steps')

    print(model.summary())

    print(train_batch_num)
    model.fit_generator(generator=itr_train,
                        validation_data=itr_validate,
                        validation_steps=val_batch_num,
                        steps_per_epoch=train_batch_num,
                        epochs=NUM_EPOCHS,
                        workers=0,
                        callbacks=[csv_logger, progbar])
                        


          
            
def get_word_to_vec_model(model_path, vocab_proc):
  vocab = vocab_proc.vocab
  matrix_size = 50
  model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=matrix_size)
  print(model.vector_size)
  print(len(model.index2word))
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
    
      
  # this doesn't correlate with our actual vocab indexes (terrible programming Vimig)
  
  # for i in range(max_size):
    # embedding_vector = model.wv[model.wv.index2word[i]]
    # if embedding_vector is not None:
      # embedding_matrix[i] = embedding_vector
  # # print(embedding_matrix[0:2])
  # # have to add one for some reason? Maybe cuz its length?
  
  # model_length = matrix_size + 1
  # # free up the memory
  del(model)
  return embedding_matrix
  
  
def main(argv=None):
  # xml_file = "pubmed_result.xml"
  # xml_file = "small_data.xml"
  # xml_file = "../small_data.xml"
  xml_file = "../cits.xml"
  # xml_file = "pubmed_result_2012_2018.xml"
  
  text_list = []

  train_dataset, test_dataset, vocab_processor, max_doc_length, dataset_size = data_load(xml_file, text_list, BATCH_SIZE, TRAIN_SET_PERCENTAGE)

  model = None
  if PRETRAINED_W2V_PATH:
    model = get_word_to_vec_model(PRETRAINED_W2V_PATH, vocab_processor)
    train_CNN(train_dataset,
              test_dataset,
              vocab_processor,
              max_doc_length,
              dataset_size,
              verbose=1,
              w2vmodel=model,
              )
  else:
    train_CNN(train_dataset, test_dataset, vocab_processor, max_doc_length, dataset_size)
  
    
      
if __name__ == '__main__':
  tf.app.run(main=main)