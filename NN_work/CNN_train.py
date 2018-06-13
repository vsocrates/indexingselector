# Code heavily influenced by https://github.com/dennybritz/cnn-text-classification-tf


import numpy as np
import os
import time
import re
import math
import seaborn as sns
import datetime
import nltk 

import lxml.etree as etree
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.learn import preprocessing

from data_utils import data_load
from data_utils import get_batch
from indexCNN import IndexClassCNN

# Data loading Parameters
TRAIN_SET_PERCENTAGE = 0.9

# Model Hyperparameters
EMBEDDING_DIM = 128 # default 128
FILTER_SIZES = "3,4,5"
NUM_FILTERS= 128 # this is per filter size, default = 128
L2_REG_LAMBDA=0.0 # L2 regularization lambda
DROPOUT_KEEP_PROB=0.5

# Training Parameters
ALLOW_SOFT_PLACEMENT=True
LOG_DEVICE_PLACEMENT=False
NUM_CHECKPOINTS = 1 # default 5
BATCH_SIZE = 64 # default 64
NUM_EPOCHS = 10 # default 200
EVALUATE_EVERY = 100 # Evaluate the model after this many steps on the test set
CHECKPOINT_EVERY = 100 # Save the model after this many steps, every time


def train_CNN(train_dataset, test_dataset, vocab_processor):
# Training, Yay!
  print("x_train type: ", type(train_dataset))
  print("y_train type: ", type(test_dataset))

  with tf.Graph().as_default():
    # TODO GPU: when this is eventually run on a GPU setup, this some of what we'd change.
    session_conf = tf.ConfigProto(
              allow_soft_placement=ALLOW_SOFT_PLACEMENT, # determines if op can be placed on CPU when GPU not avail
              log_device_placement=LOG_DEVICE_PLACEMENT # whether device placements should be logged, we don't have any for CPU
              )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      cnn = IndexClassCNN(
          sequence_length = max_doc_length,
          num_classes = 1,
          vocab_size=len(vocab_processor.vocab),
          embedding_size=EMBEDDING_DIM,
          filter_sizes=list(map(int, FILTER_SIZES.split(","))),
          num_filters=NUM_FILTERS,
          l2_reg_lambda=L2_REG_LAMBDA
          )
      
      # define the training procedure
      global_step = tf.Variable(0,name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(1e-3)
      # list of tuples with (gradients, [for] variable)
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
      
      def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x:x_batch,
          cnn.input_y:y_batch,
          cnn.dropout_keep_prob: DROPOUT_KEEP_PROB
        }
        
        # we don't need the training op back
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)
        
      def test_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a test set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, test_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
          writer.add_summary(summaries, step)
     
      # Generate batches
      batches = get_batch(list(zip(X_train, Y_train)), BATCH_SIZE, NUM_EPOCHS)
      
      # Training loop. For each batch...
      for batch in batches:
        # turns out the * unpacks whatevers in batch
        x_batch, y_batch = zip(*batch)

        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % EVALUATE_EVERY == 0:
          print("\nEvaluation:")
          test_step(X_test, Y_test, writer=test_summary_writer)
        if current_step % CHECKPOINT_EVERY == 0:
          # uses the global step number as part of the file name
          path = saver.save(sess, checkpoint_prefix, global_step=current_step)
          print("Saved model checkpoint to {}\n".format(path))
      
      # we use the simple save for now. update if it becomes an issue. 
      # make sure this is working right
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

def main(argv=None):
  # xml_file = "pubmed_result.xml"
  xml_file = "small_data.xml"
  text_list = []

  train_dataset, test_dataset, vocab_processor, max_doc_length = data_load(xml_file, text_list)
  print("HELO MEP: ", train_dataset.output_shapes)
  # padded_train_dataset = train_dataset.padded_batch(4, padded_shapes=([None],[None]))
  # padded_test_dataset = test_dataset.padded_batch(4, padded_shapes=([None],[None]))
  
  train_CNN(train_dataset, test_dataset, vocab_processor, max_doc_length)
    
      
if __name__ == '__main__':
  tf.app.run(main=main)