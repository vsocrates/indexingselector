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


def add_to_counter(elem):
  global counter
  journal_title_element = elem.find(".//Title")
  
  if journal_title_element is not None:
    counter += 1
  else:
    print("We have no Title for some reason???")
  

def main():
  global counter
  counter = 0
  # this one is the larger one
  # xml_file = "pubmed_result.xml"
  xml_file = "cits.xml"
  # xml_file = "small_data.xml"

  
  print("Starting Journal count on %s" % xml_file)
  
  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, add_to_counter)
  
  print("number of journals: ", counter)
  
  
if __name__ == '__main__':
  main()
