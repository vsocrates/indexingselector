import time
import math
import lxml.etree as etree
import numpy as np

from nltk import word_tokenize

from vocab import VocabProcessor

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
  global dataset_size
  cit_dict = {}
  dataset_size += 1
  output_text = elem.find(".//AbstractText")
  medline_cit_tag = elem.find(".//MedlineCitation")
  
  journal_title_tag = elem.find(".//Title")
  article_title_tag = elem.find(".//ArticleTitle")
  
  authors = elem.find(".//AuthorList")
  keywords = elem.find(".//KeywordList")
  
  if(output_text is not None):
    cit_dict["text"] = etree.tostring(output_text, method="text", with_tail=False, encoding='unicode')
  else:
    empty_abstract = etree.Element("AbstractText")
    empty_abstract.text = ""    
    cit_dict['text'] = etree.tostring(empty_abstract, method="text", with_tail=False, encoding='unicode')
  
  
  if authors is not None:
    # print("not none:? ", authors)
    affiliations = authors.findall(".//Affiliation")
  else:
    affiliations = []
  
  if keywords is not None:
    words = keywords.findall("Keyword")
  else:
    words = []

  cit_dict["target"] = medline_cit_tag.get("Status")
  cit_dict["journal_title"] = etree.tostring(journal_title_tag, method="text", with_tail=False, encoding='unicode')
  cit_dict["article_title"] = etree.tostring(article_title_tag, method="text", with_tail=False, encoding='unicode')

  cit_dict["affiliations"] = [etree.tostring(aff, method="text", with_tail=False, encoding='unicode') for aff in affiliations]
  cit_dict["keywords"] = [etree.tostring(word, method="text", with_tail=False, encoding='unicode') for word in words]
  
  output_list.append(cit_dict)
    
    
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

import sys

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
  
  
def data_load(xml_file, text_list, batch_size, train_size, premade_vocab_processor=None):
  global dataset_size
  dataset_size = 0
  # we are timing the abstract text data pull
  start_time = time.time()
  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_abstract_text_with_targets_and_metadata, text_list)
    
  end_time = time.time()
  
  print(get_size(text_list))
  print("Parsing took: --- %s seconds ---" % (end_time - start_time))
  
  np.random.shuffle(text_list)
  
  count_vect = None
  if premade_vocab_processor is not None:
    count_vect = premade_vocab_processor
  
  # we use nltk to word tokenize
  count_vect = VocabProcessor(word_tokenize, batch_size, train_size)
  # this function creates the datasets using the vocab.py file
  train_dataset, test_dataset, max_doc_length = count_vect.prepare_data_text_only(text_list)
    
  print("Vocabulary Size: {:d}".format(len(count_vect.vocab)))
  
  return train_dataset, test_dataset, count_vect, max_doc_length, dataset_size
  
  
def get_batch(data, batch_size, num_epochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch cuz why not
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
    else:
      shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]
  
  
