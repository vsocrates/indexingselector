import argparse
import os
import sys
import lxml.etree as etree
import re
import math
import csv
from nltk import word_tokenize

from scipy.stats import pearsonr

sys.path.append(r"C:\Users\socratesv2\Documents\indexingselector\NN_work\v2.0")

from vocab import VocabProcessor
import globals 

import gensim
import pandas as pd
#import smart_open
import random
from smart_open import smart_open

# TODO: fix this script so that it automatically finds the root element and places all pubmedarticles under that.
# Right now, you have to fix this manually, which isn't that bad, but can be improved.
# Issue is that it seems that you have to parse the entire file before you can access root. Maybe not true??
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
      category_name = func(elem, *args, **kwargs)
      
      # Instead of calling clear on the element and all its children, we will just call it on the root.
      start_tag = None
      root.clear()

  del context


def get_text_and_metadata(elem, output_list):

  cit_dict = {}
  
  output_text = elem.find(".//AbstractText")
  medline_cit_tag = elem.find(".//MedlineCitation")
  
  if globals.SPLIT_WITH_DATE:
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
  
  cit_dict["target"] = medline_cit_tag.get("Status")
  if globals.SPLIT_WITH_DATE:
    cit_dict['dcom'] = dcom_date
  
  output_list.append(cit_dict)

  
def read_corpus(documents):
    for i, plot in enumerate(documents):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(plot, max_len=30), [i])
  
  
def main():

  globals.SPLIT_WITH_DATE = False
  globals.VOCAB_LOWERCASE = True
  
  global NUM_POS
  global NUM_NEG

  parser = argparse.ArgumentParser()

  # Data loading params
  parser.add_argument("-f", "--data-file", help="location of data file", required=True)
  arguments = parser.parse_args()
  globals.XML_FILE = arguments.data_file

  xml_file = globals.XML_FILE
    
  print("All arguments: ", arguments)

  text_list = []
  with open(xml_file, "rb") as xmlf:      
    journal_context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(journal_context, get_text_and_metadata, text_list)
  
  should_remove_stop_words = True
  should_stem = False
  
  pos_vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
  neg_vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
  
  text_only = [str(text['text']) for text in text_list]

  train_corpus = list(read_corpus(text_only))

  # for idx, doc in enumerate(text_list):
  print(train_corpus[:2])

  model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=110)
  model.build_vocab(train_corpus)
  model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
  
  output_file = os.path.splitext(os.path.basename(globals.XML_FILE))[0] + "_doc2vec_50dim.w2v"
  
  model.save_word2vec_format(output_file, doctag_vec=True, word_vec=False)


  
if __name__ == '__main__':
  main()