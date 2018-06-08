import lxml.etree as etree
import re
import math
import csv

# TODO: fix this script so that it automatically finds the root element and places all pubmedarticles under that.
# Right now, you have to fix this manually, which isn't that bad, but can be improved.
# Issue is that it seems that you have to parse the entire file before you can access root. Maybe not true??
def fast_iter_new_file_write(context, func, output_file_list, *args, **kwargs):
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
      
      if category_name is "medhistory":
        output_file_list[0].write(etree.tostring(elem))
      elif category_name is "chem":
        output_file_list[1].write(etree.tostring(elem))
      elif category_name is "phys":
        output_file_list[2].write(etree.tostring(elem))
      elif category_name is "nuclear":
        output_file_list[3].write(etree.tostring(elem))
      else:
        output_file_list[4].write(etree.tostring(elem))
      
      # Instead of calling clear on the element and all its children, we will just call it on the root.
      start_tag = None
      root.clear()

  del context


def get_journal_category(elem, jids_by_category):
  medline_jrnl_info_elem = elem.find(".//MedlineJournalInfo")
  nlm_uid = str(medline_jrnl_info_elem.find(".//NlmUniqueID").text)
  
  for category, jid_list in jids_by_category.items():
    if nlm_uid in jid_list:
      return category
      
  return "none"

def main():
  xml_file = "pubmed_result.xml"
  # xml_file = "small_data.xml"
  med_history_fname = "pubmed_result_med_history.xml"
  chem_fname = "pubmed_result_chemistry.xml"
  physics_fname = "pubmed_result_physics.xml"
  nuclear_fname = "pubmed_result_nuclear.xml"
  other_journal_fname = "pubmed_result_other_journal.xml"
  
  journal_id_categorized_fname = "journal_ids_by_category_parseable.txt"
  ids_by_category_cleaned = {}

  with open(journal_id_categorized_fname, "r") as jids_file:  
    file_contents = jids_file.read()
    ids_by_category = [s.strip() for s in file_contents.split("\n\n")]
    # need an array accessible by dict
    ids_by_category_cleaned['medhistory'] = ids_by_category[0].split("\n")
    ids_by_category_cleaned['chem'] = ids_by_category[1].split("\n")
    ids_by_category_cleaned['phys'] = ids_by_category[2].split("\n")
    ids_by_category_cleaned['nuclear'] = ids_by_category[3].split("\n")
       
  journal_metadata = []
  with open(xml_file, "rb") as xmlf, \
        open(med_history_fname, "wb") as med_history_file, \
        open(chem_fname, "wb") as chem_file, \
        open(physics_fname, "wb") as phys_file, \
        open(nuclear_fname, "wb") as nuclear_file, \
        open(other_journal_fname, "wb") as other_file:
    
    output_files = [med_history_file, chem_file, phys_file, nuclear_file, other_file]
    # initialize the files
    for file in output_files:
      print("file", file)
      file.write(bytearray("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n", encoding="utf8"))
      result_set_elem = etree.Element("PubmedArticleSet")
      file.write(etree.tostring(result_set_elem))
      
    journal_context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter_new_file_write(journal_context, get_journal_category, output_files, ids_by_category_cleaned)

if __name__ == '__main__':
  main()