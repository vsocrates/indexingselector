# Heavily influenced by https://github.com/LightTag/BibSample/blob/master/preppy.py

from collections import defaultdict
from collections import Counter 
from string import punctuation
import itertools 

try:
  # pylint: disable=g-import-not-at-top
  import cPickle as pickle
except ImportError:
  # pylint: disable=g-import-not-at-top
  import pickle
  
from nltk.corpus import stopwords 
from nltk.stem.snowball import EnglishStemmer
  
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.platform import gfile

import globals

UNK = "<UNK>"
PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"

class VocabProcessor:
  
  def __init__(self, tokenizer_fn, batch_size, remove_stop_words, should_stem, limit_vocab=True, max_vocab_size=80000):
    self.vocab = defaultdict(self.next_value)  # map tokens to ids. Automatically gets next id when needed
    self.token_counter = Counter()  # Counts the token frequency
    self.vocab[PAD] = globals.PAD_ID
    self.vocab[UNK] = globals.UNK_ID
    self.vocab[START] = globals.START_ID
    self.vocab[EOS] = globals.END_ID
    self.next = globals.END_ID  # After 3 comes 4
    self.tokenizer = tokenizer_fn
    self.reverse_vocab = {}

    self.batch_size = batch_size
    self.remove_stop_words = remove_stop_words
    self.should_stem = should_stem
    self.limit_vocab = limit_vocab
    self.max_vocab_size = max_vocab_size
    
    if remove_stop_words:
      remove_words = stopwords.words("english") + list(punctuation)
      # not included in punctuation for some reason
      remove_words.append("+")
      remove_words.append("``")
      remove_words.append("''")
      remove_words.append("'s")
      self.remove_word_set = set(remove_words)
      # adding punctuation from the same flag
    if self.should_stem:
      self.stemmer = EnglishStemmer()
  
  """Gets the next index for defaultdict"""
  def next_value(self):
    self.next += 1
    return self.next

  def reset_processor(self):
    self.vocab = defaultdict(self.next_value)  # map tokens to ids. Automatically gets next id when needed
    self.token_counter = Counter()  # Counts the token frequency
    self.vocab[PAD] = globals.PAD_ID
    self.vocab[UNK] = globals.UNK_ID
    self.vocab[START] = globals.START_ID
    self.vocab[EOS] = globals.END_ID
    self.next = globals.END_ID  # After 3 comes 4
    self.reverse_vocab = {}
    
  """Convert the ids back to strings."""
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
      if self.limit_vocab:        
        if token in self.vocab:
          self.token_counter[token] += 1
          return self.vocab[token]
        else:
          if self.next < self.max_vocab_size:
            self.token_counter[token] += 1
            return self.vocab[token]
          else:
            self.token_counter[UNK] += 1
            return self.vocab[UNK]
      else:
          self.token_counter[token] += 1
          return self.vocab[token]

  # does more than just tokenization
  def tokenize(self, text):
      words = self.tokenizer(text)
    
      # 7-17-18 1:12 PM Testing with lowercase all words, works, switched to this!
      if self.remove_stop_words:
        if globals.VOCAB_LOWERCASE:
          words = [word.lower() for word in words if word.lower() not in self.remove_word_set]
        else:
          words = [word for word in words if word.lower() not in self.remove_word_set]
      if self.should_stem:
        words = [self.stemmer.stem(word) for word in words]
      return words

  
  def tokens_to_id_list(self, tokens):
      return list(map(self.convert_token_to_id, tokens))

  # def sentence_to_id_list(self, sent):
      # tokens = self.sentence_to_tokens(sent)
      # id_list = self.tokens_to_id_list(tokens)
      # return id_list

  # def sentence_to_numpy_array(self, sent):
      # id_list = self.sentence_to_id_list(sent)
      # return np.array(id_list)

  # All used to map back to vocab, not tested fully or used

  # def update_reverse_vocab(self):
      # self.reverse_vocab = {id_: token for token, id_ in self.vocab.items()}

  # def id_list_to_text(self, id_list):
      # tokens = ''.join(map(lambda x: self.reverse_vocab[x], id_list))
      # return tokens  
  
  # Untested
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
  
  
