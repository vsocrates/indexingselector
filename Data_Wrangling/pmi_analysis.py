import argparse
from operator import itemgetter
import math
import itertools
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
  global NUM_POS
  global NUM_NEG
  

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

  if cit_dict["target"] == "MEDLINE":
    NUM_POS += 1
    # print([etree.tostring(aff, method="text", with_tail=False, encoding='unicode') for aff in affiliations])
    # print("\n")
  elif cit_dict['target'] == "PubMed-not-MEDLINE":
    NUM_NEG += 1
    
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
  NUM_POS = 0
  NUM_NEG = 0

  parser = argparse.ArgumentParser()

  # Data loading params
  parser.add_argument("-f", "--data-file", help="location of data file", required=True)
  parser.add_argument("-w", "--word-file", help="location of word freq file")
  arguments = parser.parse_args()
  globals.XML_FILE = arguments.data_file
  globals.WORD_FREQ = arguments.word_file

  xml_file = globals.XML_FILE
  output_file = os.path.splitext(os.path.basename(globals.XML_FILE))[0] + "_word_freqs.txt"
  print("All arguments: ", arguments)

  text_list = []
  with open(xml_file, "rb") as xmlf:      
    journal_context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(journal_context, make_lang_model, text_list)
  
  NUM_TOTAL = len(text_list)
  should_remove_stop_words = True
  should_stem = False
  
  if not globals.WORD_FREQ:
    # pos_vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
    # neg_vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
    vocab_proc = VocabProcessor(word_tokenize, 16, should_remove_stop_words, should_stem)
    pos_tokens_list = []
    neg_tokens_list = []
    # the main thing keeping track of all of our counts, for pos and neg.
    token_count_dict = {}
    for idx, doc in enumerate(text_list):
      tokens = vocab_proc.tokenize(str(doc['text']))
      if doc['target'] == "MEDLINE":
        pos_tokens_list.append(tokens)
      elif doc['target'] == "PubMed-not-MEDLINE":
        neg_tokens_list.append(tokens)
      word_id_list = vocab_proc.tokens_to_id_list(tokens)
      
    print("Done making tokens!")
    
    for rank, pair in enumerate(vocab_proc.token_counter.most_common()):  
      token_count_dict[pair[0]] = [0,0,0]
      # check pos first
      for token_set in pos_tokens_list:
        if pair[0] in token_set:
          # add to total first
          token_count_dict[pair[0]][0] += 1
          # add to pos
          token_count_dict[pair[0]][1] += 1
      # now neg
      for token_set in neg_tokens_list:    
        if pair[0] in token_set:
          # add to total first
          token_count_dict[pair[0]][0] += 1
          # add to neg
          token_count_dict[pair[0]][2] += 1
    print(list(itertools.islice(token_count_dict.items(), 5, 15)))

          
    with open(output_file, "wb") as out:
      for key, value in token_count_dict.items():
        out_string = key + " " + " ".join(map(str,value)) + "\n"
        out.write(out_string.encode('utf8'))

  else:
    token_count_dict = {}
    linecount = 0 
    with open(globals.WORD_FREQ, "rb") as xmlf:
      line = xmlf.readline()
      while line:
        line_list = line.decode("utf8").split()
        token_count_dict[line_list[0]] = [int(line_list[1]),int(line_list[2]),int(line_list[3])]
        # linecount += 1
        # if linecount > 100:
          # break
        line = xmlf.readline()
        
    pos_pmi_vals = {}
    neg_pmi_vals = {}
    for key, value in token_count_dict.items():
      # pos first
      if value[1] <= 0:
        pos_pmi_vals[key] = float("-Inf")
      else:
        pos_pmi_vals[key] = math.log( (value[1] * NUM_TOTAL)/(value[0] * NUM_POS), 2 ) / (-1 * math.log(value[1] / NUM_TOTAL, 2) )
      # neg next
      if value[2] <= 0:
        neg_pmi_vals[key] = float('-Inf')
      else:
        neg_pmi_vals[key] = math.log( (value[2] * NUM_TOTAL)/(value[0] * NUM_NEG), 2 ) / (-1 * math.log(value[2] / NUM_TOTAL, 2) )
    
    counter = 0
    pos_pmi_fname = os.path.splitext(os.path.basename(globals.XML_FILE))[0] + "_pos_pmi.txt"
    neg_pmi_fname = os.path.splitext(os.path.basename(globals.XML_FILE))[0] + "_neg_pmi.txt"
    with open(pos_pmi_fname, "wb") as out:
      for key, value in sorted(pos_pmi_vals.items(), key = itemgetter(1), reverse = True):
        out_string = key + " " + str(value) + "\n"
        out.write(out_string.encode("utf8"))
        counter += 1
        if counter < 50:
          print("%s: %s" % (key, value))
          
    counter = 0
    print("\n")

    with open(neg_pmi_fname, "wb") as out:
      for key, value in sorted(neg_pmi_vals.items(), key = itemgetter(1), reverse = True):
        out_string = key + " " + str(value) + "\n"
        out.write(out_string.encode("utf8"))
        counter += 1
        if counter < 50:
          print("%s: %s" % (key, value))
          
if __name__ == '__main__':
  main()