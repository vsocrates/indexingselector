from collections import defaultdict
from collections import Counter 
import pickle
import itertools 

try:
  # pylint: disable=g-import-not-at-top
  import cPickle as pickle
except ImportError:
  # pylint: disable=g-import-not-at-top
  import pickle
  
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.platform import gfile

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"
BATCH_SIZE = 64 # default 64

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
    max_doc_length = 0 
    for idx, doc in enumerate(doc_data_list):
      # print("Preparing doc number %d" % idx)
      tokens = self.tokenize(str(doc['text']))
      word_id_list = self.tokens_to_id_list(tokens)      
      if len(word_id_list) > max_doc_length:
        max_doc_length = len(word_id_list)
      all_word_id_list.append(word_id_list)
      
      # add to labels
      if doc['target'] == "MEDLINE":
        labels.append([0,1])
      elif doc['target'] == "PubMed-not-MEDLINE":
        labels.append([1,0])
      
    # we are adding start and end tags
    for doc in all_word_id_list:
      doc.insert(0, 1)
      doc.append(2)
      
    # We have to do this here, because we just added two elements to each list, max increased by two
    max_doc_length += 2
    
    # # we'll randomize the data and create train and test datasets here: 
    # all_word_id_list, labels = shuffle(all_word_id_list, labels)
    
    # here we're padding the words to match the longest abstract text
    # output_arr = sequence.pad_sequences(all_word_id_list, padding="post", value=0)
    
    # now we'll split train/test set
    # TODO: will eventually have to replace this with cross-validation
    X_train, X_test, Y_train, Y_test = train_test_split(all_word_id_list, labels, test_size=0.10, random_state=42, shuffle=True)

    train_tuple = zip(X_train, Y_train)
    test_tuple = zip(X_test, Y_test)
    # print("First shape test: ", X_train[0])
    # print("Second shape test: ", Y_train[0])
    def train_generator():
      while True:
        # print("RESTART!!!?")
        train_tuple = zip(X_train, Y_train)
        data_iter = iter(train_tuple)
        for x, y in data_iter:
          # print("Y DATA: ", y)
          yield x, y
        # print("OUT OF DATA!!! ")
        

    def test_generator():
      # while True:
      for x, y in test_tuple:
        yield x, y
          
    # print("X_train: ", X_train)
    # print("Type: ", X_test[0])
      
    # We are deciding to make them all the same length, as opposed to based on batch. 
    # TODO: look into if this is the right thing to do for CNN
    train_dataset = tf.data.Dataset.from_generator(train_generator,
                                           output_types= (tf.int32, tf.int32),
                                           output_shapes=( tf.TensorShape([None]),tf.TensorShape([2]) ))
                                           
    test_dataset = tf.data.Dataset.from_generator(test_generator,
                                           output_types= (tf.int32, tf.int32),
                                           output_shapes=( tf.TensorShape([None]),tf.TensorShape([2]) ))
    
    print("before batching: ", train_dataset)
    
    batched_train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([max_doc_length], [2])).                          repeat()
    batched_test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=([max_doc_length],[2])).                          repeat()

    # print("after batching: ", batched_train_dataset)

    # train_dataset = tf.data.Dataset.from_generator( lambda: (X_train, Y_train), (tf.int32, tf.int32), (tf.TensorShape([None, None]), tf.TensorShape([None,1])) )
    # test_dataset = tf.data.Dataset.from_generator(lambda: (X_test, Y_test), (tf.int32, tf.int32), 
        # (tf.TensorShape([None, None]), tf.TensorShape([None,1])) )
    print(train_dataset)
    print(test_dataset)
  
    # TODO: this is for if we want to map backwards, which we can do later.
    # this.update_reverse_vocab()

    return batched_train_dataset, batched_test_dataset, max_doc_length



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
  
  def save(self, filename):
    """Saves vocabulary processor into given file.
    Args:
      filename: Path to output file.
    """
    with gfile.Open(filename, 'wb') as f:
      f.write(pickle.dumps(self))

  @classmethod
  def restore(cls, filename):
    """Restores vocabulary processor from given file.
    Args:
      filename: Path to file to load from.
    Returns:
      VocabularyProcessor object.
    """
    with gfile.Open(filename, 'rb') as f:
      return pickle.loads(f.read())
  
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
  
