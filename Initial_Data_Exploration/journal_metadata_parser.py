import lxml.etree as etree
import re
import math
import csv

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


def get_journal_metadata(elem, output_list):
  currently_indexed_flag = elem.find(".//CurrentlyIndexedYN")
  mesh_heading_list_elem = elem.find(".//MeshHeadingList")
  broad_heading_list_elem = elem.find(".//BroadJournalHeadingList")
  mesh_headings = []
  broad_headings = []
  if mesh_heading_list_elem:
    for child in mesh_heading_list_elem:
      mesh_headings.append(child.text)

  if :
    for child in mesh_heading_list_elem:
      mesh_headings.append(child.text)
            
      
  output_list.append({"nlmID": elem.find(".//NlmUniqueID").text,
                      "title":elem.find(".//Title").text,
                      "medlineTA":elem.find(".//MedlineTA").text,
                      "publisher": elem.find(".//Publisher").text
                      
                      })


def main():
  xml_file = "cits.xml"
  # xml_file = "small_data.xml"
  output_fname = "pos_neg_by_journal_2014-2018.csv"
  journal_data_fname = "journal_metadata.csv"
  journal_list = []
  
  journal_metadata = []
  with open(journal_data_fname, "rb") as journal_data:
    journal_context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(journal_context, get_journal_metadata, journal_list)
