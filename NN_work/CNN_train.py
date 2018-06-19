# Code heavily influenced by https://github.com/dennybritz/cnn-text-classification-tf


import numpy as np
import os
import time
import re
import math
import datetime
import nltk 

import lxml.etree as etree
import tensorflow as tf
from tensorflow.contrib.learn import preprocessing

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim

from data_utils import data_load
from data_utils import get_batch
from indexCNN import IndexClassCNN

# Data loading Parameters
TRAIN_SET_PERCENTAGE = 0.9

# Model Hyperparameters
EMBEDDING_DIM = 200 # default 128
FILTER_SIZES = "3,4,5"
NUM_FILTERS= 128 # this is per filter size; default = 128
L2_REG_LAMBDA=0.0 # L2 regularization lambda
DROPOUT_KEEP_PROB=0.5

# Training Parameters
ALLOW_SOFT_PLACEMENT=True
LOG_DEVICE_PLACEMENT=False
NUM_CHECKPOINTS = 2 # default 5
BATCH_SIZE = 64 # default 64
NUM_EPOCHS = 10 # default 200
EVALUATE_EVERY = 5 # Evaluate the model after this many steps on the test set; default 100
CHECKPOINT_EVERY = 5 # Save the model after this many steps, every time
PRETRAINED_W2V_PATH = "PubMed-and-PMC-w2v.bin"

# TODO: rename vars, Remember, these datasets below are already padded and batched
def train_CNN(train_dataset,
              test_dataset,
              vocab_processor,
              max_doc_length,
              model=None,
              ):

  # TODO GPU: when this is eventually run on a GPU setup, this some of what we'd change.
  session_conf = tf.ConfigProto(
            allow_soft_placement=ALLOW_SOFT_PLACEMENT, # determines if op can be placed on CPU when GPU not avail
            log_device_placement=LOG_DEVICE_PLACEMENT, # whether device placements should be logged, we don't have any for CPU
            operation_timeout_in_ms=60000
            )
  sess = tf.Session(config=session_conf)
  with sess.as_default():
    
    # We have to make the iterator here since we want to initialize the IndexClassCNN class and don't use   placeholders anymore
    # However, they are immediately reset when we actually do the training
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                         train_dataset.output_shapes)
                                         
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    sess.run(train_init_op)
    input_x, input_y = iterator.get_next()
    
    if model is not None:
      pretrained_model = True
    else:
      pretrained_model = False
      
    cnn = IndexClassCNN(
        input_x,
        input_y,
        DROPOUT_KEEP_PROB,
        sequence_length = max_doc_length,
        num_classes = 2, # TODO: change this so it's not set like this.
        vocab_size=len(vocab_processor.vocab),
        embedding_size=EMBEDDING_DIM,
        filter_sizes=list(map(int, FILTER_SIZES.split(","))),
        num_filters=NUM_FILTERS,
        has_pretrained_model=pretrained_model,
        l2_reg_lambda=L2_REG_LAMBDA
        )


    # define the training procedure
    global_step = tf.Variable(0,name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    # list of tuples with (gradients, [for each] variable)
    grad_var_pairs = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)
    
    # Keep track of the gradient values and sparsity (to see later)
    grad_summaries = []
    for gradient, var in grad_var_pairs:
      if gradient is not None:
        grad_hist_summary = tf.summary.histogram("{}/gradient/hist".format(var.name), gradient)
        sparsity_summary = tf.summary.scalar("{}/gradient/sparsity".format(var.name), tf.nn.zero_fraction(gradient))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
        
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    accuracy_summary = tf.summary.scalar("accuracy", cnn.accuracy)
    
    # Training Summaries
    train_summary_op = tf.summary.merge([loss_summary, accuracy_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
    # Test Summaries
    test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
    
    # Checkpoint Directory. Used to store the model at checkpoints.
    # Tensorflow assumes this already exists, so we'll make it here
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
    
    # Save the vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))
    
    # Initialize all vars to run model
    sess.run(tf.global_variables_initializer())
    
    # if model is not None:
      # print("made it here1:")
    
    # set up if we have an embedding already
    if model is not None:
      embedding_placeholder = tf.placeholder(tf.float32, [len(vocab_processor.vocab), EMBEDDING_DIM])
      embedding_init = cnn.words.assign(embedding_placeholder)    
      sess.run(embedding_init, feed_dict={embedding_placeholder:model})
    
    # I guess initializing the pretrained model? I"m not too sure if this is fine, or I should be using feed_dict somehow
    # sess.run(cnn.words.assign(model))
    
    def train_step(x_batch, y_batch):
      """
      A single training step
      """
      cnn.input_x, cnn.input_y = x_batch, y_batch
      cnn.dropout_keep_prob = DROPOUT_KEEP_PROB
      output = sess.run(x_batch)
      
      # training op doesn't return anything
      _, step, summaries, loss, accuracy = sess.run(
          [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy])

      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)
      
    def test_step(x_batch, y_batch, writer=None):
      """
      Evaluates model on a test set
      """
      cnn.input_x, cnn.input_y = x_batch, y_batch
      cnn.dropout_keep_prob = 1.0

      step, summaries, loss, accuracy = sess.run(
        [global_step, test_summary_op, cnn.loss, cnn.accuracy])
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      if writer:
        writer.add_summary(summaries, step)
   
    # Training loop. For each batch...
    for _ in range(NUM_EPOCHS):
      sess.run(train_init_op)
      input_x, input_y = iterator.get_next()
      train_step(input_x, input_y)
      current_step = tf.train.global_step(sess, global_step)
      
      if current_step % EVALUATE_EVERY == 0:
        print("\nEvaluation:")
        sess.run(test_init_op)
        test_x, test_y = iterator.get_next()
        while True:
          try:
            test_step(test_x,test_y, writer=test_summary_writer)
          except tf.errors.OutOfRangeError:
            break
            
      if current_step % CHECKPOINT_EVERY == 0:
        # uses the global step number as part of the file name
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
    
    # we use the simple save for now. update if it becomes an issue. 
    # TODO: make sure this is working right
    final_model_dir = os.path.abspath(os.path.join(out_dir, "final"))
    input_x, input_y = cnn.get_inputs()
    output = cnn.get_outputs()
    tf.saved_model.simple_save(
            sess,
            final_model_dir,
            inputs={"input_x":input_x,
                    "input_y":input_y},
            outputs={"predictions":output}
            )
    
    # legacy, in case we can't use simple_save (like for 1.4.1 tensorflow edition on hpc)
    
    # builder = tf.saved_model.builder.SavedModelBuilder(final_model_dir)

    # sess.run(train_init_op)
    # input_x, input_y = iterator.get_next()

    # text_input_tensor_info = tf.saved_model.utils.build_tensor_info(input_x)
    # predictions_output_tensor_info = tf.saved_model.utils.build_tensor_info(output)
    
    # prediction_signature = (
      # tf.saved_model.signature_def_utils.build_signature_def(
        # inputs={"text":text_input_tensor_info},
        # outputs={
          # "classes":predictions_output_tensor_info
        # },
        # method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      # )
    # )
    # builder.add_meta_graph_and_variables(
        # sess, [tf.saved_model.tag_constants.SERVING],
        # signature_def_map={
            # 'predict_indexing':
                # prediction_signature,
        # })    

def get_word_to_vec_model(model_path, vocab_length):
  matrix_size = 50
  model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
  # store the embeddings in a numpy array
  
  # embedding_matrix = np.zeros((len(model.wv.vocab) + 1, EMBEDDING_DIM))
  embedding_matrix = np.zeros((vocab_length, EMBEDDING_DIM))
  # for i in range(len(model.wv.vocab)):
  for i in range(vocab_length):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
   
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

  train_dataset, test_dataset, vocab_processor, max_doc_length = data_load(xml_file, text_list)

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
