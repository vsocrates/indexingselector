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
      
      
  