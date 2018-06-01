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


def get_journals(elem, output_list):
  journal_title_element = elem.find(".//Title")
  medline_cit_tag = elem.find(".//MedlineCitation")
  # print("journal", journal_title_element.text)
  # print("cit", medline_cit_tag.get("Status"))
  if journal_title_element is not None:
    journal_title = str(journal_title_element.text)
    status = medline_cit_tag.get("Status")
    if journal_title in output_list:
      pos_neg_list = output_list[journal_title]
      if status == "MEDLINE":
        pos_neg_list[0] += 1
      elif status == "PubMed-not-MEDLINE":      
        pos_neg_list[1] += 1
    else:
      output_list[journal_title] = [0,0]
      pos_neg_list = output_list[journal_title]
      if status == "MEDLINE":
        pos_neg_list[0] += 1
      elif status == "PubMed-not-MEDLINE":      
        pos_neg_list[1] += 1
  else:
    print("We have no title for some reason?")

def get_text_list(dictList):
	output_list = []
	for text in dictList:
		output_list.append(str(text['text']))
	return output_list


def main():
  xml_file = "cits.xml"
  # xml_file = "small_data.xml"
  journal_list = {}

  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_journals, journal_list)

  # print("final list", journal_list)
  
  with open("pos_neg_by_journal.csv", "w", newline="") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')

    for key, value in journal_list.items(): 
      spamwriter.writerow([key, value[0], value[1]])

  
if __name__ == '__main__':
  main()
