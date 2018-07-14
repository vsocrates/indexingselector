import lxml.etree as etree
import re
import math
import csv

# This file gets the articles from a source file based on the JID (journal ID)

# TODO: fix this script so that it automatically finds the root element and places all pubmedarticles under that.
# Right now, you have to fix this manually, which isn't that bad, but can be improved.
# Issue is that it seems that you have to parse the entire file before you can access root. Maybe not true??

def fast_iter(context, func, output_file_list, *args, **kwargs):
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
      is_listed_journal = func(elem, *args, **kwargs)
      
      if is_listed_journal:
        output_file_list.write(etree.tostring(elem))
			# Instead of calling clear on the element and all its children, we will just call it on the root.
      start_tag = None
      root.clear()
      
  del context


def get_indv_journal_articles(elem, jid_to_match):
  global counter
  medline_jrnl_info_elem = elem.find(".//MedlineJournalInfo")
  nlm_uid = str(medline_jrnl_info_elem.find(".//NlmUniqueID").text)
  # print(jid_to_match)
  # print(nlm_uid)
  if nlm_uid == jid_to_match:
    counter += 1
    return True
  else:
    # print("We have no MedlineJournalInfo for some reason???")
    return False

def main():
  global counter
  counter = 0
  xml_file = "../data/pubmed_result_2012_2018.xml"
  jid_to_match = "101563288"
  
  pattern = re.compile(r"[^\/]*$")
  outxml_path = pattern.search(xml_file).group(0).split(".")[0]

  
  out_file = jid_to_match + "_" + outxml_path + ".xml"
  print("Starting Article Extraction for JID %s on %s" % (jid_to_match, xml_file))
  
  with open(xml_file, "rb") as xmlf, open(out_file, "wb") as out:
  
    out.write(bytearray("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n", encoding="utf8"))
    result_set_elem = etree.Element("PubmedArticleSet")
    out.write(etree.tostring(result_set_elem))

  
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_indv_journal_articles, out, jid_to_match)
  
  print("number of journals: ", counter)
  
  
if __name__ == '__main__':
  main()
