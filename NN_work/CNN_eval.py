import os
import csv
import numpy as np

import tensorflow as tf
from tensorflow.contrib.learn import preprocessing

from data_utils import get_batch
from data_utils import data_load

# Evaluation Parameters
BATCH_SIZE = 64
MODEL_DIR = ""
EVAL_TRAIN = True

# Misc Parameters
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

def evaluate_CNN(X_test, Y_test, X_raw):
  
  graph = tf.Graph()
  with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=ALLOW_SOFT_PLACEMENT,
        log_device_placement=LOG_DEVICE_PLACEMENT
        )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      # Load the saved mta graph and restore variables
      tf.saved_model.loader.load(sess, [tag_constants.SERVING], MODEL_DIR)
      
      # Get the placeholders from the graph by name
      input_x = graph.get_operation_by_name("input_x").outputs[0]
      input_y = graph.get_operation_by_name("input_y").outputs[0]
      dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
      
      # Tensors we want to evaluate
      predictions = graph.get_operation_by_name("output/predictions").outputs[0]
      
      # Generate batches for one epoch
      batches = get_batch(list(X_test), BATCH_SIZE, 1, shuffle=False)
      
      # Collect the predictions here
      all_predictions = []
      
      for x_test_batch in batches:
        batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        
  # Print the accuracy if _test is defined
  if Y_test is not None:
    correct_predicions = float(sum(all_predictions == Y_test))
    print("total number of test examples: {}".format(len(Y_test)))
    print("Accuracy: {:g}".format(correct_predicions/float(len(Y_test))))
  
  predictions_csv = np.column_stack((np.array(X_raw), all_predictions))
  out_put = os.path.join(MODEL_DIR, "..", "prediction.csv")
  print("Saving evaluation to {0}".format(out_path))
  with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
    
def main(argv=None):
  # xml_file = "pubmed_result.xml"
  xml_file = "small_data.xml"
  text_list = []

  # Map data into vocabulary
  vocab_path = os.path.join(MODEL_DIR, "..", "vocab")
  vocab_processor = preprocessing.VocabularyProcessor.restore(vocab_path)

  
  if EVAL_TRAIN:
    X_vocab_vectors_shuffled, Y_targets_shuffled, count_vect = data_load(xml_file, text_list, vocab_processor)
    # This line gets the second value of each pair of Y targets (i.e. [0,1] => 1) as the prediction
    Y_targets_shuffled = np.argmax(Y_targets_shuffled, axis=1)
  else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]  

  print("\nEvaluating...\n")
  evaluate_CNN(X_vocab_vectors_shuffled, Y_targets_shuffled, text_list)
      
if __name__ == '__main__':
  tf.app.run(main=main)