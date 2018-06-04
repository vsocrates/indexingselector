import tensorflow as tf
import numpy as np

class IndexClassCNN(object):
  """
  A CNN for document classification of medline citations using abstract text and metadata.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  """
  def __init__(self, ):
    