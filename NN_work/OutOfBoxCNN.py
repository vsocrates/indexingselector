import lxml.etree as etree
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.learn import preprocessing
import numpy as np
import os
import pandas as pd
import re
import math
import seaborn as sns
import nltk 
from sklearn.feature_extraction.text import CountVectorizer

def multilayer_perceptron(input_tensor, weights, biases):
	layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
	layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
	layer_1_activation = tf.nn.relu(layer_1_addition)
	# Hidden layer with RELU activation
	layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
	layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
	layer_2_activation = tf.nn.relu(layer_2_addition)
	# Output layer with linear activation
	out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
	out_layer_addition = out_layer_multiplication + biases['out']
	
	return out_layer_addition

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


if __name__ == '__main__':
	# xml_file = "cits.xml"
	xml_file = "small_data.xml"
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
	
	count_vect = CountVectorizer(stop_words="english")
	X_train_counts = count_vect.fit_transform(get_text_list(text_train_set))
	print("X_train_counts.shape: " , X_train_counts.shape)
	print("vocab length ", len(count_vect.vocabulary_))
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
			
			
	# Process vocabulary
	vocab_processor = preprocessing.VocabularyProcessor(
			len(X_train_docs))
	x_train_tensor = np.array(list(vocab_processor.fit_transform(X_train_docs)))
	print(x_train_tensor.shape)
	x_test_tensor = np.array(list(vocab_processor.transform(X_test_docs)))
	n_words = len(vocab_processor.vocabulary_)
	print('Total words: %d' % n_words)


	# Network Parameters
	n_hidden_1 = 10        # 1st layer number of features
	n_hidden_2 = 5         # 2nd layer number of features
	n_input = n_words  # Words in vocab
	n_classes = 2          # Categories: MEDLINE, PubMed-not-MEDLINE

	weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	
	biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
	}
	
	input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
	output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output")

	# Construct model
	prediction = multilayer_perceptron(input_tensor, weights, biases)
	# Define loss
	entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
	loss = tf.reduce_mean(entropy_loss)

	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	