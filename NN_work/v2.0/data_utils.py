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
    output_list.append({"text": etree.tostring(empty_abstract, method="text", with_tail=False, encoding='unicode'), "target":medline_cit_tag.get("Status")})

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
      output_list.append([1,0])
    elif text == "PubMed-not-MEDLINE":
      output_list.append([0,1])
  return output_list

def data_load(xml_file, text_list, premade_vocab_processor=None):
  # we are timing the abstract text data pull
  start_time = time.time()

  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_abstract_text_with_targets, text_list)
    
  end_time = time.time()
  print("Total set size: " , len(text_list))
  print("Total execution time parsing: {}".format(end_time - start_time))
  
  # we want to shuffle the data first, so we have a good mix of positive and negative targets
  np.random.shuffle(text_list)
  
  # then we will build the vocabulary
  max_document_length = max([len(str(x['text']).split(" ")) for x in text_list])
  count_vect = None
  if premade_vocab_processor is not None:
    count_vect = premade_vocab_processor
  
  count_vect = VocabProcessor(word_tokenize)
  # X_vocab_vectors = np.array(list(count_vect.fit_transform(get_text_list(text_list))))
  dataset = count_vect.prepare_data(text_list)
  print("dataset no way!!!: ", dataset)
  
  vocab_dict = count_vect.vocabulary_._mapping
  sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
  vocabulary = list(list(zip(*sorted_vocab))[0])
  print("vocab1: ", vocab_dict)

  print("vocab1: ", vocabulary)
  print("xvectors: ", X_vocab_vectors[0:2])
  Y_targets = np.array(get_target_list(text_list))

  # let's shuffle it some more, before we do the split, on the entire list
  np.random.seed(15)
  shuffle_indices = np.random.permutation(np.arange(len(Y_targets)))
  X_vocab_vectors_shuffled = X_vocab_vectors[shuffle_indices]
  Y_targets_shuffled = Y_targets[shuffle_indices]
  print("Vocabulary Size: {:d}".format(len(count_vect.vocabulary_)))
  
  del X_vocab_vectors, Y_targets 

  return X_vocab_vectors_shuffled, Y_targets_shuffled, count_vect
  
  
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
  
  
