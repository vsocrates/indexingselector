# Code heavily influenced by https://github.com/dennybritz/cnn-text-classification-tf

import lxml.etree as etree
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.learn import preprocessing
import numpy as np
import os
 
import re
import math
import seaborn as sns
import nltk 
from sklearn.feature_extraction.text import CountVectorizer

TRAIN_SET_PERCENTAGE = 0.9

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
    output_list.append({"text": etree.tostring(empty_abstract), "target":medline_cit_tag.get("Status")})
    empty_abs_counter += 1

def get_text_list(dictList):
  output_list = []
  for text in dictList:
    output_list.append(str(text['text']))
  return output_list

  # this returns 1s and 0s to save space and time, but don't call it a lot because of computation
def get_target_list(dictList):
  target_list = []
  output_list = []
  for text in dictList:
    target_list.append(str(text['target']))

  for text in target_list:
    # print(text['target'])
    if text == "MEDLINE":
      output_list.append(1)
    elif text == "PubMed-not-MEDLINE":
      output_list.append(0)
      
  print("output", output_list)
  return output_list

  
def data_load():
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
  
  # then we will build the vocabulary
  max_document_length = max([len(str(x['text']).split(" ")) for x in text_list])
  count_vect = preprocessing.VocabularyProcessor(max_document_length)
  X_vocab_vectors = np.array(list(count_vect.fit_transform(get_text_list(text_list))))
  Y_targets = np.array(get_target_list(text_list))

  # let's shuffle it some more, before we do the split, on the entire list
  np.random.seed(15)
  shuffle_indices = np.random.permutation(np.arange(len(Y_targets)))
  X_vocab_vectors_shuffled = X_vocab_vectors[shuffle_indices]
  Y_targets_shuffled = Y_targets[shuffle_indices]
  
  # now we'll split train/test set
  # TODO: will eventually have to replace this with cross-validation
  train_test_divide = math.floor(TRAIN_SET_PERCENTAGE * len(text_list))
  X_vocab_vect_train, X_vocab_vect_test = X_vocab_vectors_shuffled[:train_test_divide], X_vocab_vectors_shuffled[train_test_divide:]
  Y_target_train, Y_target_test = Y_targets_shuffled[:train_test_divide], Y_targets_shuffled[train_test_divide:]
  
  del X_vocab_vectors, Y_targets, X_vocab_vectors_shuffled, Y_targets_shuffled
  
  print("Vocabulary Size: {:d}".format(len(count_vect.vocabulary_)))
  print("Train/Dev split: {:d}/{:d}".format(len(Y_target_train), len(Y_target_test)))
  return X_vocab_vect_train, Y_target_train, count_vect, X_vocab_vect_test, Y_target_test

def train_CNN():
  
  


if __name__ == '__main__':
  # xml_file = "cits.xml"
  xml_file = "small_data.xml"
  text_list = []
  global empty_abs_counter
  empty_abs_counter = 0

  X_vocab_vect_train, Y_target_train, vocab_processor, X_vocab_vect_test, Y_target_test = data_load()    
  train()
  