# import sys
# import codecs
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)
import lxml.etree as etree
import pprint
import sys
import codecs
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import math 

from sklearn import metrics

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


def process_element(elem, output_list):
	global empty_abs_counter
	output_text = elem.find(".//AbstractText")
	medline_cit_tag = elem.find(".//MedlineCitation")
	if(output_text is not None):
		output_list.append(
		{"text": etree.tostring(output_text, method="text", with_tail=False, encoding='unicode'),
		 "target":medline_cit_tag.get("Status")
		 })
	else:
		empty_abstract = etree.Element("AbstractText")
		empty_abstract.text = ""
		output_list.append({"text": empty_abstract, "target":medline_cit_tag.get("Status")})
		empty_abs_counter += 1

def get_top_n_words(count_vectorizer, corpus, n=None):
	"""
	List the top n words in a vocabulary according to occurrence in a text corpus.
	"""
	vec = count_vectorizer.fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0) 
	words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
	return words_freq[:n]

def get_text_list(dictList):
	output_list = []
	for text in dictList:
		output_list.append(str(text['text']))
	return output_list
	
if __name__ == '__main__':
	xml_file = "cits.xml"
	# xml_file = "small_data.xml"
	text_list = []
	global empty_abs_counter
	empty_abs_counter = 0
	# we are timing the abstract text data pull
	import time
	start_time = time.time()

	with open(xml_file, "rb") as xmlf:
		context = etree.iterparse(xmlf, events=('start', 'end', ), encoding='utf-8')
		fast_iter(context, process_element, text_list)
	end_time = time.time()
	print("Total set size: " , len(text_list))
	print("Number of Cits with Empty Abstract: ", empty_abs_counter)
	print("Total execution time parsing: {}".format(end_time - start_time))
	
	# we want to shuffle the data first, so we have a good mix of positive and negative targets
	np.random.shuffle(text_list)
	train_test_divide = math.floor(0.9 * len(text_list))
	print("Training set size: ", train_test_divide)
	print("Testing set size: ", len(text_list) - train_test_divide)
	text_train_set = text_list[:train_test_divide]
	text_test_set = text_list[train_test_divide:]
	
	# we need to pull out our testing data 
	X_test_docs = [str(x.get("text")) for x in text_test_set]
	test_target_list = []
	# here we are mapping the target categories to numerical values for speed
	for text in text_test_set:
		# print(text['target'])
		if text['target'] == "MEDLINE":
			test_target_list.append(1)
		elif text['target'] == "PubMed-not-MEDLINE":
			test_target_list.append(0)
	
	# now the same for the training data
	X_train_docs = [str(x.get("text")) for x in text_train_set]
	train_target_list = []
	for text in text_train_set:
		# print(text['target'])
		if text['target'] == "MEDLINE":
			train_target_list.append(1)
		elif text['target'] == "PubMed-not-MEDLINE":
			train_target_list.append(0)

			
	text_clf = Pipeline([('vect', CountVectorizer(stop_words="english")),
						('tfidf', TfidfTransformer()),
						('clf', SGDClassifier(loss='hinge', penalty='l2',
											alpha=1e-3, random_state=42,
											max_iter=5, tol=None)),
						])

	
	# we are timing the count vectorization of the text corpus (all in memory read)
	start_time = time.time()
	# apparently, in python 3 all strings are unicode. who knew? certainly not me.
	# X_train_counts = count_vect.fit_transform(X_train_docs)
	# print("X_train_counts.shape: " , X_train_counts.shape)

	text_clf.fit(X_train_docs, train_target_list)
	end_time = time.time()
	print("Total execution time training: {}".format(end_time - start_time))

	# now let's see how well we did
	predicted = text_clf.predict(X_test_docs)
	# print("test_target_list", test_target_list)
	print("Mean: ", np.mean(predicted == test_target_list))

	# now let's get some metrics
	print(metrics.classification_report(test_target_list, predicted, 
			target_names=["PubMed-not-MEDLINE", "MEDLINE"]))
	print(metrics.confusion_matrix(test_target_list, predicted))
	

	# Further analysis time! 
	# Let's see what the most frequent words in both types of docs are.
	cv = text_clf.named_steps['vect']
		
		
	medline_text_target = []
	nonmedline_text_target = []
	for data in text_list:
		if data['target'] == "MEDLINE":
			medline_text_target.append(data)
		elif data['target'] == "PubMed-not-MEDLINE":
			nonmedline_text_target.append(data)
	
	medline_text = get_text_list(medline_text_target)
	non_medline_text = get_text_list(nonmedline_text_target)
	medline_top_words = get_top_n_words(cv, medline_text, n=10)
	nonmedline_top_words = get_top_n_words(cv, non_medline_text, n=10)
	print(medline_top_words)
	print(nonmedline_top_words)
	
	
	