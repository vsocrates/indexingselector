from collections import defaultdict
from collections import Counter 

import numpy as np

import tensorflow as tf

PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"

class VocabProcessor:
  
  def __init__(self, tokenizer_fn):
    self.vocab = defaultdict(self.next_value)  # map tokens to ids. Automatically gets next id when needed
    self.token_counter = Counter()  # Counts the token frequency
    self.vocab[PAD] = 0
    self.vocab[START] = 1
    self.vocab[EOS] = 2
    self.next = 2  # After 2 comes 3
    self.tokenizer = tokenizer_fn
    self.reverse_vocab = {}

  def next_value(self):
      self.next += 1
      return self.next
  
  """
    I don't know if this is the right move, but take in data of the form of a list of dict:
    With all features and raw text that we want. 
    i.e. [{
        "text":"raw text hello world",
        "label":1,
        "author_affiliation":["NLM", "DBMI"],
        ...
        },{...}]
        
    Returns:
      vocab: the vocabulary
      sequence_ex_list: the Dataset taken from from_tensor_slices containing all data per training example
  """
  def prepare_data(self, doc_data_list, save_records=False):
    # first we want to split up the text in all the docs and make the vocab
    all_word_id_list = []
    labels = []
    for idx, doc in enumerate(doc_data_list):
      print("Preparing doc number %d" % idx)
      tokens = self.tokenize(str(doc['text']))
      word_id_list = self.tokens_to_id_list(tokens)
      all_word_id_list.append(word_id_list)
      labels.append(doc['target'])
    
    all_word_id_list = np.array(all_word_id_list)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((all_word_id_list, labels))
    # print(self.vocab)
    return dataset



  """
    Expects this passed as a list of documents in memory. If not, we need to come up with a different way to read in all these docs
  """
  def fit():
    pass
  """
    This one is going to convert the documents into word-id matrices (which are each just a list)
  """
  def transform():
    pass

  def sequence_to_tf_example(self, sequence):
      '''
      Gets a sequence (a text like "hello how are you") and returns a a SequenceExample
      :param sequence: Some text
      :return: A A sequence exmaple
      '''
      # Convert the text to a list of ids
      id_list = self.sentence_to_id_list(sequence)
      ex = tf.train.SequenceExample()
      # A non-sequential feature of our example
      sequence_length = len(id_list) + 2  # For start and end
      # Add the context feature, here we just need length
      ex.context.feature["length"].int64_list.value.append(sequence_length)
      # Feature lists for the two sequential features of our example
      # Add the tokens. This is the core sequence.
      # You can add another sequence in the feature_list dictionary, for translation for instance
      fl_tokens = ex.feature_lists.feature_list["tokens"]
      # Prepend with start token
      fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
      for token in id_list:
          # Add those tokens one by one
          fl_tokens.feature.add().int64_list.value.append(token)
      # apend  with end token
      fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
      return ex

  def ids_to_string(self, tokens, length=None):
      string = ''.join([self.reverse_vocab[x] for x in tokens[:length]])
      return string

  def convert_token_to_id(self, token):
      '''
      Gets a token, looks it up in the vocabulary. If it doesn't exist in the vocab, it gets added to id with an id
      Then we return the id
      :param token:
      :return: the token id in the vocab
      '''
      self.token_counter[token] += 1
      return self.vocab[token]

  def tokenize(self, text):
      return self.tokenizer(text)

  def tokens_to_id_list(self, tokens):
      return list(map(self.convert_token_to_id, tokens))

  # def sentence_to_id_list(self, sent):
      # tokens = self.sentence_to_tokens(sent)
      # id_list = self.tokens_to_id_list(tokens)
      # return id_list

  # def sentence_to_numpy_array(self, sent):
      # id_list = self.sentence_to_id_list(sent)
      # return np.array(id_list)

  def update_reverse_vocab(self):
      self.reverse_vocab = {id_: token for token, id_ in self.vocab.items()}

  def id_list_to_text(self, id_list):
      tokens = ''.join(map(lambda x: self.reverse_vocab[x], id_list))
      return tokens  
  
  # we are not going to convert to tfrecord right now. if we want to save files after, implement this above.
  
  # def to_tfrecord(df, file_path):
    # def _int64_feature(value):
        # return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # def _float_feature(value):
        # return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # def _bytes_feature(value):
        # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))

    # def _int64_feature_list(values):
        # return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

    # def _bytes_feature_list(values):
        # return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

    # with tf.python_io.TFRecordWriter(file_path) as writer:
        # for row in df.itertuples():
            # example = tf.train.SequenceExample(
                # context=tf.train.Features(
                    # feature={
                        # "id": _bytes_feature(getattr(row, "id")),
                        # "group": _bytes_feature(getattr(row, "group")),
                        # "label": _bytes_feature(getattr(row, "label")),
                        # "sentiment": _float_feature(getattr(row, "sentiment")),
                        # "rating": _float_feature(getattr(row, "rating")),
                        # "text": _bytes_feature(getattr(row, "text")),
                        # "cleaned_text": _bytes_feature(getattr(row, "cleaned_text")),
                        # "token_count": _int64_feature(getattr(row, "token_count")),
                    # }),
                # feature_lists=tf.train.FeatureLists(
                    # feature_list={
                        # "tokens": _bytes_feature_list(getattr(row, "tokens")),
                        # "token_ids": _int64_feature_list(getattr(row, "token_ids")),
                    # }))
            # writer.write(example.SerializeToString())
  
