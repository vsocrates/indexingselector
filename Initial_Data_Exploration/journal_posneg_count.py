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


def get_journals(elem, output_list, nlm_ids, journal_metadata):
  journal_title_element = elem.find(".//Title")
  medline_cit_tag = elem.find(".//MedlineCitation")
  
  medline_jrnl_info_elem = elem.find(".//MedlineJournalInfo")
  nlm_uid = str(medline_jrnl_info_elem.find(".//NlmUniqueID").text)
  
  if journal_title_element is not None:
    status = medline_cit_tag.get("Status")
    if nlm_uid in output_list:
      pos_neg_list = output_list[nlm_uid]['posnegs']
      if status == "MEDLINE":
        pos_neg_list[0] += 1
      elif status == "PubMed-not-MEDLINE":      
        pos_neg_list[1] += 1
    else:
      output_list[nlm_uid] = {"posnegs":[0,0]}
      pos_neg_list = output_list[nlm_uid]['posnegs']
      if status == "MEDLINE":
        pos_neg_list[0] += 1
      elif status == "PubMed-not-MEDLINE":      
        pos_neg_list[1] += 1
    # comment this section out if we don't want to populate with the BroadJournalHeadingList and MeshHeadingList 
    if 'broad_list' not in output_list[nlm_uid]:
      if nlm_uid in nlm_ids:
        output_list[nlm_uid]['broad_list'] = journal_metadata[nlm_uid]['broad_list']
      else:
        # print("We can't get any metadata, not in the given file")
        output_list[nlm_uid]['broad_list'] = "UNK"
    if "name" not in output_list[nlm_uid]:
      output_list[nlm_uid]['name'] = journal_title_element.text
      
  else:
    print("We have no Title for some reason???")
  
def get_text_list(dictList):
	output_list = []
	for text in dictList:
		output_list.append(str(text['text']))
	return output_list


def main():
    
  # this one is the larger one
  xml_file = "pubmed_result.xml"
  # xml_file = "cits.xml"
  # xml_file = "small_data.xml"

  output_fname = "pos_neg_by_journal_2014-2018.csv"

  # this one is my theoretical one
  # journal_data_fname = "journal_metadata.csv"
  # this one is Jim's
  journal_data_fname = "JournalInfo_fromPubMed.csv"
 
  journal_list = {}
  journal_metadata = {}
  journal_metadata_nlmids = []
  # from Jim's file. comment this out if you use the journal_metadata_parser.py to create a file (unfinished)
  with open(journal_data_fname, "rt") as jdata_xml:
    jrnl_reader = csv.reader(jdata_xml, delimiter=",")
    # row[0] = nlmid
    # row[1] = short name
    # row[2] = long name
    # row[3] = publication country
    # row[4] = language
    # row[5] = BroadJournalHeadingList (semicolon separated)
    for row in jrnl_reader:
      journal_metadata_nlmids.append(row[0])
      journal_metadata[str(row[0])] = {"short_name":row[1],
                                       "long_name":row[2],
                                       "pub_country": row[3],
                                       "language":row[4],
                                       "broad_list":row[5]
                                      }
  
  print("Starting Journal count on %s" % xml_file)

  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_journals, journal_list, journal_metadata_nlmids, journal_metadata)
  
  with open(output_fname, "w", newline="") as csvfile:
    print("Writing to file: %s" % output_fname)  
    spamwriter = csv.writer(csvfile, delimiter=',')

    for key, value in journal_list.items(): 
      spamwriter.writerow([key, value['name'], value['posnegs'][0],
                           value['posnegs'][1],
                           value['broad_list']
                          ])

  print("Done!")
  
if __name__ == '__main__':
  main()
