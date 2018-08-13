# Vanilla Python
import os
import re
import argparse
import string
from profilehooks import profile
import time

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
import globals 

from data_utils import data_load
from data_utils import get_word_to_vec_model
from conditional_decorator import conditional_decorator
from confusion_matrix_classes import BinaryTruePositives
from confusion_matrix_classes import BinaryTrueNegatives
from confusion_matrix_classes import BinaryFalsePositives 
from confusion_matrix_classes import BinaryFalseNegatives

# Keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from keras.models import Model
from keras import backend

from keras.callbacks import CSVLogger
from keras.callbacks import ProgbarLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD

def train_LSTMAux(datasets,
              vocab_processors,
              max_doc_lengths,
              dataset_size,
              w2vmodel=None,
              ):
              
  session_conf = tf.ConfigProto(
          allow_soft_placement=globals.ALLOW_SOFT_PLACEMENT, # determines if op can be placed on CPU when GPU not avail
          log_device_placement=globals.LOG_DEVICE_PLACEMENT, # whether device placements should be logged, we don't have any for CPU
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
                if globals.DEBUG:
                  print("OutOfRangeError Exception Thrown")          
                break
              except Exception as e: 
                if globals.DEBUG:
                  print(e)
                  print("Unknown Exception Thrown")
                break
            yield value_list, labels_out
            
    train_batch_num = int((dataset_size*(globals.TRAIN_SET_PERCENTAGE)) // globals.BATCH_SIZE) + 1
    val_batch_num = int((dataset_size*(1-globals.TRAIN_SET_PERCENTAGE)) // globals.BATCH_SIZE)
    
    itr_train_abs = make_multiple_iterator([datasets.abs_text_train_dataset,datasets.affl_train_dataset], train_batch_num)
    itr_validate_abs = make_multiple_iterator([datasets.abs_text_test_dataset,datasets.affl_test_dataset], val_batch_num)
    
   
    main_input = Input(shape=(max_doc_lengths.abs_text_max_length,), dtype="int32", name="main_input")
    embedding_layer = Embedding(input_dim=len(vocab_processors['text'].vocab),
                                output_dim=globals.EMBEDDING_DIM,
                                weights=[w2vmodel],
                                input_length=max_doc_lengths.abs_text_max_length,
                                trainable=False,
                                name="embedding")(main_input)
    dropout1 = Dropout(globals.MAIN_DROPOUT_KEEP_PROB[0], name="dropout1")(embedding_layer)
    lstm_out = LSTM(globals.MAIN_LSTM_SIZE)(dropout1)
    dropout2 = Dropout(globals.MAIN_DROPOUT_KEEP_PROB[1], name="dropout2")(lstm_out)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(dropout2)

    # auxiliary information
    aux_input = Input(shape=(max_doc_lengths.affl_max_length,), dtype="int32", name="affl_input")
    affl_embedding_layer = Embedding(input_dim=len(vocab_processors['text'].vocab),
                                output_dim=globals.EMBEDDING_DIM,
                                input_length=max_doc_lengths.abs_text_max_length,
                                name="affl_embedding")(aux_input)
    dropout3 = Dropout(globals.AUX_DROPOUT_KEEP_PROB[0], name="dropout3")(affl_embedding_layer)
    aux_lstm_out = LSTM(globals.AUX_LSTM_SIZE)(dropout3)
    dropout4 = Dropout(globals.AUX_DROPOUT_KEEP_PROB[1], name="dropout4")(aux_lstm_out)
    
    concat = Concatenate()([dropout4, dropout2])
    x = Dense(globals.COMBI_DENSE_LAYER_DIM, activation='relu')(concat)
    x = Dense(globals.COMBI_DENSE_LAYER_DIM, activation='relu')(x)
    x = Dense(globals.COMBI_DENSE_LAYER_DIM, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    
    # stochastic gradient descent algo, currently unused
    opt = SGD(lr=0.01)

    model = Model(inputs=[main_input,aux_input] , outputs=[main_output, auxiliary_output])
    
    truepos_metricfn = BinaryTruePositives()
    trueneg_metricfn = BinaryTrueNegatives()
    falsepos_metricfn = BinaryFalsePositives()
    falseneg_metricfn = BinaryFalseNegatives()
    
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy',
              truepos_metricfn,
              trueneg_metricfn,
              falsepos_metricfn,
              falseneg_metricfn])
    # model._make_predict_function()
                  # will be useful when we actually combine
                  # loss_weights=[1., 0.2]
    callbacks = []
    # callbacks.append(EarlyStopping(monitor="val_))
    callbacks.append(ReduceLROnPlateau())
    # Tensorboard in this version of Keras, broken. Need to update to latest version
    # callbacks.append(TensorBoard())
    callbacks.append(ModelCheckpoint("CNNweights.{epoch:02d}-{val_loss:.2f}.hdf5", period=5))
    
    verbosity = 2
    if globals.DEBUG:
      callbacks = []
      callbacks.append(CSVLogger('training.log'))
      # callbacks.append(ProgbarLogger(count_mode='steps'))
      verbosity = 1
    print(model.summary())

    model.fit_generator(generator=itr_train_abs,
                        validation_data=itr_validate_abs,
                        validation_steps=val_batch_num,
                        steps_per_epoch=train_batch_num,
                        epochs=globals.NUM_EPOCHS,
                        verbose=verbosity,
                        workers=0,
                        callbacks=callbacks)

    if globals.SAVE_MODEL:
      pattern = re.compile(r"[^\/]*$")
      outxml_path = pattern.search(globals.XML_FILE).group(0).split(".")[0]
      outw2v_path = pattern.search(globals.PRETRAINED_W2V_PATH).group(0).split(".")[0]
      model.save("LSTMAux_" + outxml_path + "_" + outw2v_path + "_saved_model.h5")
