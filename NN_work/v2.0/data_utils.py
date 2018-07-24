import time
import datetime
import sys
import math
import collections 

from profilehooks import profile
import gensim 

import lxml.etree as etree
import numpy as np

from nltk import word_tokenize

import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.platform import gfile

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from vocab import VocabProcessor
from conditional_decorator import conditional_decorator
import globals

Datasets = collections.namedtuple('Datasets',['abs_text_train_dataset',
                                              'abs_text_test_dataset',
                                              "jrnl_title_train_dataset",
                                              "jrnl_title_test_dataset",
                                              "art_title_train_dataset",
                                              "art_title_test_dataset",
                                              "affl_train_dataset",
                                              "affl_test_dataset",
                                              "keyword_train_dataset",
                                              "keyword_test_dataset",
                                             ])
Dataset_Max_Lengths = collections.namedtuple('Dataset_Max_Lengths', ["abs_text_max_length",
                                                                     "jrnl_title_max_length",
                                                                     "art_title_max_length",
                                                                     "affl_max_length",
                                                                     "keyword_max_length"
                                                                    ])

DO_TIMING_ANALYSIS = False 
                                                               
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
    # we have to set these labels like this, otherwise softmax complains
    if text == "MEDLINE":
      output_list.append([1])
    elif text == "PubMed-not-MEDLINE":
      output_list.append([0])
  return output_list


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# ------------------------- XML data loading methods/conversion to Dataset methods  ------------------------- 

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

  # Deprecated, not used
def get_abstract_text_with_targets(elem, output_list):
  cit_dict = {}
  
  output_text = elem.find(".//AbstractText")
  medline_cit_tag = elem.find(".//MedlineCitation")
  
  if(output_text is not None):
    cit_dict["text"] = etree.tostring(output_text, method="text", with_tail=False, encoding='unicode')
  else:
    empty_abstract = etree.Element("AbstractText")
    empty_abstract.text = ""    
    cit_dict['text'] = etree.tostring(empty_abstract, method="text", with_tail=False, encoding='unicode')
  
  cit_dict["target"] = medline_cit_tag.get("Status")

  output_list.append(cit_dict)
    
def get_abstract_text_with_targets_and_metadata(elem, output_list):
  global NUM_POS
  global NUM_NEG
  cit_dict = {}
  output_text = elem.find(".//AbstractText")
  medline_cit_tag = elem.find(".//MedlineCitation")
  
  journal_title_tag = elem.find(".//Title")
  article_title_tag = elem.find(".//ArticleTitle")
  
  authors = elem.find(".//AuthorList")
  keywords = elem.find(".//KeywordList")
  
  dcom = medline_cit_tag.find(".//DateCompleted")
  dcom_year = etree.tostring(dcom.find("Year"), method="text", with_tail=False, encoding='unicode')
  dcom_month = etree.tostring(dcom.find("Month"), method="text", with_tail=False, encoding='unicode')
  dcom_day = etree.tostring(dcom.find("Day"), method="text", with_tail=False, encoding='unicode')
  dcom_date = datetime.date(int(dcom_year), int(dcom_month), int(dcom_day))
  
  if(output_text is not None):
    cit_dict["text"] = etree.tostring(output_text, method="text", with_tail=False, encoding='unicode')
  else:
    empty_abstract = etree.Element("AbstractText")
    empty_abstract.text = ""    
    cit_dict['text'] = etree.tostring(empty_abstract, method="text", with_tail=False, encoding='unicode')
  
  
  if authors is not None:
    affiliations = authors.findall(".//Affiliation")
  else:
    affiliations = []
  
  if keywords is not None:
    words = keywords.findall("Keyword")
  else:
    words = []

  cit_dict["target"] = medline_cit_tag.get("Status")
  
  if cit_dict["target"] == "MEDLINE":
    NUM_POS += 1
  elif cit_dict['target'] == "PubMed-not-MEDLINE":
    NUM_NEG += 1

  cit_dict["journal_title"] = etree.tostring(journal_title_tag, method="text", with_tail=False, encoding='unicode')
  cit_dict["article_title"] = etree.tostring(article_title_tag, method="text", with_tail=False, encoding='unicode')

  cit_dict["affiliations"] = [etree.tostring(aff, method="text", with_tail=False, encoding='unicode') for aff in affiliations]
  cit_dict["keywords"] = [etree.tostring(word, method="text", with_tail=False, encoding='unicode') for word in words]
  cit_dict['dcom'] = dcom_date
  
  # print("cit_dict: ", cit_dict['affiliations'])
  # print('citation: ', cit_dict)
  output_list.append(cit_dict)

@conditional_decorator(profile, DO_TIMING_ANALYSIS)
def data_load(xml_file, text_list, batch_size, remove_stop_words, should_stem, limit_vocab_size, max_vocab_length, train_size=0.0, with_aux_info=False, pos_text_list=[], test_date=None):
  global NUM_POS
  global NUM_NEG
  NUM_POS = 0
  NUM_NEG = 0
  
  # we are timing the abstract text data pull
  start_time = time.time()
  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_abstract_text_with_targets_and_metadata, text_list)
    
  end_time = time.time()
  # we need to save a local copy, since the NUM_POS and NUM_NEG globals will continue to increased
  local_num_pos = NUM_POS
  local_num_neg = NUM_NEG

  print("Num of pos articles in text_list: ", NUM_POS)
  print("Num of neg articles in text_list: ", NUM_NEG)
  print("Num of total articles in text_list: ", len(text_list))
  print("Data size (bytes): ", get_size(text_list))
  print("Parsing took: --- %s seconds ---" % (end_time - start_time))
  
  if globals.POS_XML_FILE:
    start_time = time.time()
    with open(globals.POS_XML_FILE, "rb") as xmlf:
      context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
      fast_iter(context, get_abstract_text_with_targets_and_metadata, pos_text_list)
      
    end_time = time.time()
    print("Num of pos ex articles: ", len(pos_text_list))
    print("Data size of Pos Ex Articles (bytes): ", get_size(pos_text_list))
    print("Parsing for Pos Ex articles took: --- %s seconds ---" % (end_time - start_time))
  
  # add positive examples to entire dataset
  np.random.shuffle(pos_text_list)
  if local_num_neg > local_num_pos:
    diff = local_num_neg - local_num_pos
    text_list = text_list + pos_text_list[0:diff]

  print("Num of articles after pos ex addition: ",len(text_list))
  
  # we use nltk to word tokenize
  vocab_proc_dict = {}
  if with_aux_info:
    # because there are 5 things we want (including raw abstract text)
    for name in ["text", "journal_title", "article_title", "affiliations", "keywords"]:
      vocab_proc_dict[name] = VocabProcessor(word_tokenize, batch_size, remove_stop_words, should_stem, limit_vocab_size, max_vocab_length)
      
    
    datasets, max_doc_length = prepare_data_text_with_aux(vocab_proc_dict, text_list, test_date, train_size)
  else:
    count_vect = VocabProcessor(word_tokenize, batch_size, remove_stop_words, should_stem, limit_vocab_size, max_vocab_length)
    # this function creates the datasets using the vocab.py file
    vocab_proc_dict = {"text":count_vect}
    datasets, max_doc_length = prepare_data_text_only(vocab_proc_dict, text_list, test_date, train_size)
        
  print("Vocabulary Size: ", len(vocab_proc_dict['text'].vocab))
  # print("vocab", vocab_proc_dict['text'].token_counter)
    
  return datasets, vocab_proc_dict, max_doc_length, len(text_list)

"""
  I don't know if this is the right move, but take in data of the form of a list of dict:
  With all features and raw text that we want. 
  i.e. [{
      "text":"raw text hello world",
      "label":1,
      "author_affiliation":["NLM", "DBMI"],
      ...
      },{...}]
      
  Returns:
    vocab: the vocabulary
    sequence_ex_list: the Dataset taken from from_tensor_slices containing all data per training example
"""
@conditional_decorator(profile, DO_TIMING_ANALYSIS)
def prepare_data_text_only(vocab_proc_dict, doc_data_list, test_date, train_size, save_records=False):
  vocab_processor = vocab_proc_dict['text']
  vocab_processor.reset_processor()
  # first we want to split up the text in all the docs and make the vocab
  all_word_id_list = []
  labels = []
  max_doc_length = 0 
  
  # these methods do that using defaultdict, so that we can automatically add newly found tokens
  for idx, doc in enumerate(doc_data_list):
    tokens = vocab_processor.tokenize(str(doc['text']))
    word_id_list = vocab_processor.tokens_to_id_list(tokens)      
    if len(word_id_list) > max_doc_length:
      max_doc_length = len(word_id_list)
    all_word_id_list.append(word_id_list)
    
    # add numeric labels
    if doc['target'] == "MEDLINE":
      labels.append([1])
    elif doc['target'] == "PubMed-not-MEDLINE":
      labels.append([0])
  
  # we are adding start and end tags
  for doc in all_word_id_list:
    doc.insert(0, 1)
    doc.append(2)
    
  # We have to do this here, because we just added two elements to each list, max increased by two
  max_doc_length += 2
  
  
  # now we'll split train/test set
  # TODO: will eventually have to replace this with cross-validation

  # we'll randomize the data and create train and test datasets using scikit here: 
  X_train, X_test, Y_train, Y_test = train_test_split(all_word_id_list, labels, test_size=round(1.0-train_size, 2), random_state=42, shuffle=True)

  train_tuple = zip(X_train, Y_train)
  test_tuple = zip(X_test, Y_test)

  # these are the generators used to create the datasets
  # Since we regulate the length of epochs, this generator needs to cycle through the data infinitely
  def train_generator():
    train_tuple = zip(X_train, Y_train)
    data_iter = iter(train_tuple)
    for x, y in data_iter:
      yield x, y

  # Since we regulate the length of epochs, this generator needs to cycle through the data infinitely
  def test_generator():
    test_tuple = zip(X_test, Y_test)
    data_iter = iter(test_tuple)
    for x, y in data_iter:
      yield x, y
  
  train_dataset = tf.data.Dataset.from_generator(train_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
                                         
  test_dataset = tf.data.Dataset.from_generator(test_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  
  # We are deciding to make them all the same length, as opposed to pad based on batch. 
  # TODO: look into if this is the right thing to do for CNN    
  batched_train_dataset = train_dataset.padded_batch(vocab_processor.batch_size, padded_shapes=([max_doc_length], [1])).repeat()
  batched_test_dataset = test_dataset.padded_batch(vocab_processor.batch_size, padded_shapes=([max_doc_length],[1])).repeat()

  return_datasets = Datasets(batched_train_dataset, batched_test_dataset,
                             None, None, None, None, None, None, None, None)
                             
  all_max_lengths = Dataset_Max_Lengths(max_doc_length, None, None, None, None) 
                              
  # TODO: this is for if we want to map backwards, which we can do later.
  # this.update_reverse_vocab()
  return return_datasets, all_max_lengths

@conditional_decorator(profile, DO_TIMING_ANALYSIS)
def prepare_data_text_with_aux(vocab_proc_dict, doc_data_list, test_date, train_size, save_records=False):
  # shouldn't actually need to do this, but just in case.
  for name, processor in vocab_proc_dict.items():
    processor.reset_processor()

  # get all the vocab processors
  text_vocab_proc = vocab_proc_dict['text']
  jrnl_title_vocab_proc = vocab_proc_dict['journal_title']
  art_title_vocab_proc = vocab_proc_dict['article_title']
  affil_vocab_proc = vocab_proc_dict['affiliations']
  keyword_vocab_proc = vocab_proc_dict['keywords']

  # first we want to split up the text in all the docs and make the vocab
  abs_text_word_ids = []
  labels = []
  
  jrnl_title_ids = [] 
  art_title_ids = []
  affiliation_ids = []
  keyword_ids = []
  
  train_test_tracker = []
  
  max_doc_length = 0 
  max_jrnl_title_length = 0
  max_art_title_length = 0
  max_affl_length = 0
  max_keyword_length = 0
  
  idx_hold = 0

  # these methods do that using defaultdict, so that we can automatically add newly found tokens
  for idx, doc in enumerate(doc_data_list):
    # first we do abstract text
    tokens = text_vocab_proc.tokenize(str(doc['text']))
    word_id_list = text_vocab_proc.tokens_to_id_list(tokens)      
    if len(word_id_list) > max_doc_length:
      max_doc_length = len(word_id_list)
    abs_text_word_ids.append(word_id_list)
    
    tokens = jrnl_title_vocab_proc.tokenize(str(doc['journal_title']))
    word_id_list = jrnl_title_vocab_proc.tokens_to_id_list(tokens)      
    if len(word_id_list) > max_jrnl_title_length:
      max_jrnl_title_length = len(word_id_list)
    jrnl_title_ids.append(word_id_list)

    tokens = art_title_vocab_proc.tokenize(str(doc['article_title']))
    # print("article title tokesN: ", tokens)
    word_id_list = art_title_vocab_proc.tokens_to_id_list(tokens)      
    if len(word_id_list) > max_art_title_length:
      max_art_title_length = len(word_id_list)
    art_title_ids.append(word_id_list)

    tokens = affil_vocab_proc.tokenize(str(doc['affiliations']))
    # print("affiliations tokesN: ", tokens)
    word_id_list = affil_vocab_proc.tokens_to_id_list(tokens)      
    if len(word_id_list) > max_affl_length:
      max_affl_length = len(word_id_list)
      idx_hold = tokens
    affiliation_ids.append(word_id_list)
    
    tokens = keyword_vocab_proc.tokenize(str(doc['keywords']))
    word_id_list = keyword_vocab_proc.tokens_to_id_list(tokens)      
    if len(word_id_list) > max_keyword_length:
      max_keyword_length = len(word_id_list)
    keyword_ids.append(word_id_list)
    
    # add numeric labels
    if doc['target'] == "MEDLINE":
      labels.append([1])
    elif doc['target'] == "PubMed-not-MEDLINE":
      labels.append([0])
  
    # Keep track of train or test, by date
    # if doc['dcom'] > test_date:
      # train_test_tracker.append(1)
    # else:
      # train_test_tracker.append(0)
  
  # print("train_test_tracker test", train_test_tracker)
  # print("affiliation test: ", idx_hold)
  # we are adding start and end tags
  # for doc in abs_text_word_ids:
    # doc.insert(0, 1)
    # doc.append(2)
    
  # We have to do this here, because we just added two elements to each list, max increased by two
  max_doc_length += 2
  
  
  # now we'll split train/test set
  # TODO: will eventually have to replace this with cross-validation

  # this is gross.
  # we'll randomize the data and create train and test datasets using scikit here: 
  if not globals.SPLIT_WITH_DATE:
    abs_text_train, abs_text_test, \
    jrnl_title_train, jrnl_title_test, \
    art_title_train, art_title_test, \
    affl_train, affl_test, \
    keyword_train, keyword_test, \
    labels_train, labels_test = train_test_split(abs_text_word_ids,
                                                jrnl_title_ids,
                                                art_title_ids,
                                                affiliation_ids,
                                                keyword_ids,
                                                labels,
                                                test_size=round(1.0-train_size, 2), random_state=42, shuffle=True)
  else:
    abs_text_train, abs_text_test, \
    jrnl_title_train, jrnl_title_test, \
    art_title_train, art_title_test, \
    affl_train, affl_test, \
    keyword_train, keyword_test, \
    labels_train, labels_test = train_test_split_with_date(abs_text_word_ids,
                                                jrnl_title_ids,
                                                art_title_ids,
                                                affiliation_ids,
                                                keyword_ids,
                                                labels,
                                                
                                                date=test_date, 
                                                shuffle=True)
    
  # these are the generators used to create the datasets
  # we can't make just one method yet since the generators need to be callables with no params right now. 
  # This is an open issue on stackoverflow: https://github.com/tensorflow/tensorflow/issues/13101
  # Since we regulate the length of epochs, this generator needs to cycle through the data infinitely
  
  # abstract text
  def abs_text_train_generator():
    train_tuple = zip(abs_text_train, labels_train)
    data_iter = iter(train_tuple)
    for x, y in data_iter:
      yield x, y
      
  def abs_text_test_generator():
    test_tuple = zip(abs_text_test, labels_test)
    data_iter = iter(test_tuple)
    for x, y in data_iter:
      yield x, y

  # journal titles      
  def jrnl_title_train_generator():
    train_tuple = zip(jrnl_title_train, labels_train)
    data_iter = iter(train_tuple)
    for x, y in data_iter:
      yield x, y
      
  def jrnl_title_test_generator():
    test_tuple = zip(jrnl_title_test, labels_test)
    data_iter = iter(test_tuple)
    for x, y in data_iter:
      yield x, y

  # article titles
  def art_title_train_generator():
    train_tuple = zip(art_title_train, labels_train)
    data_iter = iter(train_tuple)
    for x, y in data_iter:
      yield x, y
      
  def art_title_test_generator():
    test_tuple = zip(art_title_test, labels_test)
    data_iter = iter(test_tuple)
    for x, y in data_iter:
      yield x, y
  
  # affiliations
  def affl_train_generator():
    train_tuple = zip(affl_train, labels_train)
    data_iter = iter(train_tuple)
    for x, y in data_iter:
      yield x, y
      
  def affl_test_generator():
    test_tuple = zip(affl_test, labels_test)
    data_iter = iter(test_tuple)
    for x, y in data_iter:
      yield x, y

  # keywords
  def keyword_train_generator():
    train_tuple = zip(keyword_train, labels_train)
    data_iter = iter(train_tuple)
    for x, y in data_iter:
      yield x, y
      
  # Since we regulate the length of epochs, this generator needs to cycle through the data infinitely
  def keyword_test_generator():
    test_tuple = zip(keyword_test, labels_test)
    data_iter = iter(test_tuple)
    for x, y in data_iter:
      yield x, y
      
  
  abs_text_train_dataset = tf.data.Dataset.from_generator(abs_text_train_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  abs_text_test_dataset = tf.data.Dataset.from_generator(abs_text_test_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  
  jrnl_title_train_dataset = tf.data.Dataset.from_generator(jrnl_title_train_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  jrnl_title_test_dataset = tf.data.Dataset.from_generator(jrnl_title_test_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  
  art_title_train_dataset = tf.data.Dataset.from_generator(art_title_train_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  art_title_test_dataset = tf.data.Dataset.from_generator(art_title_test_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  
  affl_train_dataset = tf.data.Dataset.from_generator(affl_train_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  affl_test_dataset = tf.data.Dataset.from_generator(affl_test_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  
  keyword_train_dataset = tf.data.Dataset.from_generator(keyword_train_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  keyword_test_dataset = tf.data.Dataset.from_generator(keyword_test_generator,
                                         output_types= (tf.int32, tf.int32),
                                         output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))
  
  # We are deciding to make them all the same length, as opposed to pad based on batch. 
  # TODO: look into if this is the right thing to do for CNN and LSTM   
  abs_text_train_dataset = abs_text_train_dataset.padded_batch(text_vocab_proc.batch_size, padded_shapes=([max_doc_length], [1]))
  abs_text_test_dataset = abs_text_test_dataset.padded_batch(text_vocab_proc.batch_size, padded_shapes=([max_doc_length],[1]))

  jrnl_title_train_dataset = jrnl_title_train_dataset.padded_batch(jrnl_title_vocab_proc.batch_size, padded_shapes=([max_jrnl_title_length], [1]))
  jrnl_title_test_dataset = jrnl_title_test_dataset.padded_batch(jrnl_title_vocab_proc.batch_size, padded_shapes=([max_jrnl_title_length],[1]))

  art_title_train_dataset = art_title_train_dataset.padded_batch(art_title_vocab_proc.batch_size, padded_shapes=([max_art_title_length], [1]))
  art_title_test_dataset = art_title_test_dataset.padded_batch(art_title_vocab_proc.batch_size, padded_shapes=([max_art_title_length],[1]))

  affl_train_dataset = affl_train_dataset.padded_batch(affil_vocab_proc.batch_size, padded_shapes=([max_affl_length], [1]))
  affl_test_dataset = affl_test_dataset.padded_batch(affil_vocab_proc.batch_size, padded_shapes=([max_affl_length],[1]))

  keyword_train_dataset = keyword_train_dataset.padded_batch(keyword_vocab_proc.batch_size, padded_shapes=([max_keyword_length], [1]))
  keyword_test_dataset = keyword_test_dataset.padded_batch(keyword_vocab_proc.batch_size, padded_shapes=([max_keyword_length],[1]))

  return_datasets = Datasets(abs_text_train_dataset=abs_text_train_dataset,
                                abs_text_test_dataset=abs_text_test_dataset,
                                jrnl_title_train_dataset=jrnl_title_train_dataset,
                                jrnl_title_test_dataset=jrnl_title_test_dataset,
                                art_title_train_dataset=art_title_train_dataset,
                                art_title_test_dataset=art_title_test_dataset,
                                affl_train_dataset=affl_train_dataset,
                                affl_test_dataset=affl_test_dataset,
                                keyword_train_dataset=keyword_train_dataset,
                                keyword_test_dataset=keyword_test_dataset)
  all_max_lengths = Dataset_Max_Lengths(abs_text_max_length=max_doc_length,
                                        jrnl_title_max_length=max_jrnl_title_length,
                                        art_title_max_length=max_art_title_length,
                                        affl_max_length=max_affl_length,
                                        keyword_max_length=max_keyword_length
                                       ) 
  # TODO: this is for if we want to map backwards, which we can do later.
  # this.update_reverse_vocab()
  return return_datasets, all_max_lengths
  
def train_test_split_with_date(*arrays, date, shuffle=True):
  pass
  
# ------------------------- W2V Gensim Loading Method  ------------------------- 

@conditional_decorator(profile, DO_TIMING_ANALYSIS)            
def get_word_to_vec_model(model_path, matrix_size, vocab_proc, vocab_proc_tag):
  vocab = vocab_proc[vocab_proc_tag].vocab
  
  if model_path.find(".bin") > 0:
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=matrix_size)
  elif model_path.find(".txt") > 0:
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, limit=matrix_size)
  else:
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, limit=matrix_size)
    
  print("Embedding Dims: ", model.vector_size)
  print("Number of Tokens in W2V Model: ", len(model.index2word))
  # store the embeddings in a numpy array
  
  # set the global for other places
  globals.EMBEDDING_DIM = model.vector_size
  
  embedding_matrix = np.zeros((len(vocab), model.vector_size))
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

  
