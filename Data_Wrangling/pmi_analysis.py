import argparse
import sys
import lxml.etree as etree
import re
import math
import csv
from nltk import word_tokenize
import os 

from scipy.stats import pearsonr

sys.path.append(r"C:\Users\socratesv2\Documents\indexingselector\NN_work\v2.0")

from vocab import VocabProcessor
import globals 

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


def make_lang_model(elem, output_list):

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


def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

        
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
  output_file = os.path.splitext(os.path.basename(globals.XML_FILE))[0] + "_word_freqs.txt"
  print("All arguments: ", arguments)

  text_list = []
  with open(xml_file, "rb") as xmlf:      
    journal_context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(journal_context, make_lang_model, text_list)
  
  should_remove_stop_words = True
  should_stem = False
  
  pos_vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
  neg_vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
  for idx, doc in enumerate(text_list):
    if doc['target'] == "MEDLINE":
      tokens = pos_vocab_proc.tokenize(str(doc['text']))
      word_id_list = pos_vocab_proc.tokens_to_id_list(tokens)      
    elif doc['target'] == "PubMed-not-MEDLINE":
      tokens = neg_vocab_proc.tokenize(str(doc['text']))
      word_id_list = neg_vocab_proc.tokens_to_id_list(tokens)      
    
  with open(output_file, "w+") as out:
    
  # print("Positive articles: ", pos_vocab_proc.token_counter.most_common(50))
  
if __name__ == '__main__':
  main()