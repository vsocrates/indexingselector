import tensorflow as tf
import numpy as np

class IndexClassCNN(object):
  """
  A CNN for document classification of medline citations using abstract text and metadata.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  
  shape(x) = [k, l, m, n] batch, row, column, channel? is this right?
  
  TODO: eventually switch to custom estimator probably.
  """
  def __init__(self, input_x, input_y, dropout, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
    # print("num class: ", num_classes)
    self.input_x = input_x#tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    self.input_y = input_y#tf.placeholder(tf.int32, [None, num_classes - 1], name="input_y")
    
    self.dropout_keep_prob = dropout#tf.placeholder(tf.float32, name="dropout_keep_prob")
    
    print("Inside CNN," , self.input_x)
    # we keep track of the l2 regularization loss
    l2_loss = tf.constant(0.0)
    
    # Embedding layer (input)
    # TODO GPU: eventually needs to be changed
    # with tf.device("/cpu:0"), tf.name_scope("embedding"):
    with tf.name_scope("embedding"):
      self.words = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0,1.0), name="words")
      
      # print("words shape: ", self.words.get_shape().as_list())
      self.embedded_chars = tf.nn.embedding_lookup(self.words, self.input_x)
      # print("embedded chars size: ", self.embedded_chars.get_shape().as_list())
      self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  
      print("embedded_chars_expanded chars size: ", self.embedded_chars_expanded.get_shape().as_list())

      
    # convolution and maxpool later per filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        #Convolution layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter")
        # print("Filter shape, sanity check: ", filter.get_shape().as_list())
        bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")
        
        conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            filter,
            strides=[1,1,1,1],
            padding="VALID",
            name="conv")
        
        # Add bias and take the relu
        h = tf.nn.relu(tf.nn.bias_add(conv,bias), name="relu")
        
        # max pooling layer over the outputs
        pooled = tf.nn.max_pool(
              h,
              ksize=[1, sequence_length - filter_size + 1,1,1], # this size is because we want our window to be over the entire length of word embeddings??? I believe
              strides=[1,1,1,1], # all windows are used
              padding="VALID", # this means no padding is added
              name="pool")
        
        pooled_outputs.append(pooled)
    
    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    self.h_pooled = tf.concat(pooled_outputs, 3) # concatenate along the third dimension [None,1,1,128]
    self.h_pooled_flat = tf.reshape(self.h_pooled, [-1, num_filters_total]) # -1 makes it a 1-D array
    
    # add dropout
    with tf.name_scope("dropout"):
      self.h_drop = tf.nn.dropout(self.h_pooled_flat, self.dropout_keep_prob)
    
    # final (unnormalized) scores and predictions
    with tf.name_scope("output"):
      filter = tf.get_variable(
            "filter",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer()) # initializes values based on reasonable guesses, depending on data
      
      bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bias")
      l2_loss += tf.nn.l2_loss(filter)
      l2_loss += tf.nn.l2_loss(bias)
      
      # below function does matmul(x,weights) + biases, the crux of NNs :)
      self.scores = tf.nn.xw_plus_b(self.h_drop, filter, bias, name="scores")
      self.predictions = tf.argmax(self.scores, 1, name="predictions")
    
    # calculate mean cross-entropy loss
    with tf.name_scope("loss"):
      print("input into logits1: ", self.scores)
      print("input into logits2: ", self.input_y)
      losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
      self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
      
      
    # Accuracy calculation
    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
      
  def get_inputs(self):
    return self.input_x, self.input_y
  def get_outputs(self):
    return self.predictions
    
    
        