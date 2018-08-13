#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for DNN-based text classification with DBpedia data."""

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

import lxml.etree as etree
import re
import math
from nltk import word_tokenize

MAX_DOCUMENT_LENGTH = 50
EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.

def fast_iter(context, func, *args, **kwargs):
	"""
	http://www.ibm.com/developerworks/xml/library/x-hiperfparse/ (Liza Daly)
	See also http://effbot.org/zone/element-iterparse.htm
	"""
	_, root = next(context)
	start_tag = None
	for event, elem in context:
		if(start_tag is None and event == "start"):
			start_tag = elem.tag
			continue
		# we are going to see if we can pull the entire pubmed article entry each time but then dump it after
		if(elem.tag == start_tag and event == "end"):
			func(elem, *args, **kwargs)
			# Instead of calling clear on the element and all its children, we will just call it on the root.
			start_tag = None
			root.clear()
	del context


def process_element(elem, output_list):
	global empty_abs_counter
	output_text = elem.find(".//AbstractText")
	medline_cit_tag = elem.find(".//MedlineCitation")
	if(output_text is not None):
		output_list.append(
		{"text": etree.tostring(output_text, method="text", with_tail=False, encoding='unicode'),
		 "target":medline_cit_tag.get("Status")
		 })
	else:
		empty_abstract = etree.Element("AbstractText")
		empty_abstract.text = ""
		output_list.append({"text": empty_abstract, "target":medline_cit_tag.get("Status")})
		empty_abs_counter += 1

def get_text_list(dictList):
	output_list = []
	for text in dictList:
		output_list.append(str(text['text']))
	return output_list



def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
  """A bag-of-words model. Note it disregards the word order in the text."""
  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
  bow = tf.feature_column.input_layer(
      features, feature_columns=[bow_embedding_column])
  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def main(unused_argv):
  global n_words
  tf.logging.set_verbosity(tf.logging.INFO)

  xml_file = "cits.xml"
  # xml_file = "small_data.xml"
  text_list = []
  global empty_abs_counter
  empty_abs_counter = 0
  # we are timing the abstract text data pull
  import time
  start_time = time.time()

  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, process_element, text_list)
  end_time = time.time()
  print("Total set size: " , len(text_list))
  print("Number of Cits with Empty Abstract: ", empty_abs_counter)
  print("Total execution time parsing: {}".format(end_time - start_time))

  # we want to shuffle the data first, so we have a good mix of positive and negative targets
  np.random.shuffle(text_list)
  train_test_divide = math.floor(0.9 * len(text_list))
  print("Training set size: ", train_test_divide)
  print("Testing set size: ", len(text_list) - train_test_divide)
  text_train_set = text_list[:train_test_divide]
  text_test_set = text_list[train_test_divide:]

  # count_vect = CountVectorizer(stop_words="english")
  # X_train_counts = count_vect.fit_transform(get_text_list(text_train_set))
  # print("X_train_counts.shape: " , X_train_counts.shape)
  # print("vocab length ", len(count_vect.vocabulary_))
  
  # we need to pull out our testing data 
  X_test_docs = get_text_list(text_test_set)
  test_target_list = []
  # here we are mapping the target categories to numerical values for speed
  for text in text_test_set:
    # print(text['target'])
    if text['target'] == "MEDLINE":
      test_target_list.append(1)
    elif text['target'] == "PubMed-not-MEDLINE":
      test_target_list.append(0)

  # now the same for the training data
  X_train_docs = get_text_list(text_train_set)
  train_target_list = []
  for text in text_train_set:
    # print(text['target'])
    if text['target'] == "MEDLINE":
      train_target_list.append(1)
    elif text['target'] == "PubMed-not-MEDLINE":
      train_target_list.append(0)

  word_tokenize_lengths = []
  for text in X_train_docs:
    word_tokenize_lengths.append(len(word_tokenize(text)))
  
  print("tokenized: ", word_tokenize_lengths)
  print("max length: ", max(word_tokenize_lengths))
  # Process vocabulary
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(X_train_docs)
  x_transform_test = vocab_processor.transform(X_test_docs)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)

  # setting up the categories for train and test
  y_train = pandas.Series(train_target_list)
  y_test = pandas.Series(test_target_list)  
  
  # Build model

  # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
  # ids start from 1 and 0 means 'no word'. But
  # categorical_column_with_identity assumes 0-based count and uses -1 for
  # missing word.
  x_train -= 1
  x_test -= 1
  model_fn = bag_of_words_model
  classifier = tf.estimator.Estimator(model_fn=model_fn)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_train},
      y=y_train,
      batch_size=len(x_train),
      num_epochs=None,
      shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=100)

  # Predict.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # now let's get some metrics
  print(metrics.classification_report(y_test, y_predicted, 
      target_names=["PubMed-not-MEDLINE", "MEDLINE"]))
  print(metrics.confusion_matrix(y_test, y_predicted))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run(main=main)