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
  global journal_count
  start_tag = None

  for event, elem in context:

    if(start_tag is None and event == "start"):
      start_tag = elem.tag
      continue
    # we are going to see if we can pull the entire pubmed article entry each time but then dump it after
    if(elem.tag == start_tag and event == "end"):
      journal_count += 1 
      func(elem, *args, **kwargs)
      # Instead of calling clear on the element and all its children, we will just call it on the root.
      start_tag = None
      root.clear()
  del context


def get_journals_list(elem, output_list):
  global journal_count
  journal_count += 1
  currently_indexed_flag = elem.find(".//CurrentlyIndexedYN")

  if currently_indexed_flag.text == "Y":
    indexingHistoryList = elem.find(".//IndexingHistoryList")
    newest_entry_elem = indexingHistoryList[0]
    newest_date = newest_entry_elem[0]
    for child in indexingHistoryList:
      child_date = child[0] # assuming there is only ever child and it is the DateOfAction elem
      if( int(child_date.find("Year").text) > int(newest_date.find("Year").text) and
        int(child_date.find("Month").text) > int(newest_date.find("Month").text) and
        int(child_date.find("Day").text) > int(newest_date.find("Day").text) ):
        newest_entry_elem = child
        newest_date = child_date
    if newest_entry_elem.get('IndexingTreatment') == "Selective" and newest_entry_elem.get('IndexingStatus') == "Currently-indexed":
      output_list.append({"title":elem.find(".//Title").text,
                          "medlineTA":elem.find(".//MedlineTA").text,
                          "nlmID": elem.find(".//NlmUniqueID").text,
                          "indexingStatus": newest_entry_elem.get('IndexingStatus'),
                          "indexingTreatment": newest_entry_elem.get('IndexingTreatment'),
                          "indexingDateYear": newest_date.find("Year").text,
                          "indexingDateMonth": newest_date.find("Month").text,
                          "indexingDateDay": newest_date.find("Day").text,
                          })
  elif currently_indexed_flag.text == "N":
    pass
  else:
    print(elem.find(".//NlmUniqueID").text)
    print("There is no CurrentlyIndexedYN tag")
    
    
def get_text_list(dictList):
  output_list = []
  for text in dictList:
    output_list.append(str(text['text']))
  return output_list


def main():
  global journal_count
  journal_count = 0
  xml_file = "lsi2018.xml"
  journal_list = []

  with open(xml_file, "rb") as xmlf:
    context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
    fast_iter(context, get_journals_list, journal_list)
  
  with open("journalIDs.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ['title', 'medlineTA', "nlmID", "indexingStatus", 
        "indexingTreatment", "indexingDateYear", "indexingDateMonth", "indexingDateDay"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for journal_entry in journal_list:
      writer.writerow(journal_entry)

  print("count: ", journal_count)
  
if __name__ == '__main__':
  main()
