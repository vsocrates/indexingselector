# Data loading Parameters
global XML_FILE#  = "pubmed_result.xml"
global POS_XML_FILE
global PRETRAINED_W2V_PATH# = "PubMed-and-PMC-w2v.bin"
global WITH_AUX_INFO
global MATRIX_SIZE#  = 9000
global VOCAB_LOWERCASE
global SPLIT_WITH_DATE

# Common Model Hyperparameters
global AUX_TRAINABLE
global REMOVE_STOP_WORDS
global SHOULD_STEM
global TRAIN_SET_PERCENTAGE#  = 0.9
global EMBEDDING_TRAINABLE
global LEARNING_RATE

global EMBEDDING_DIM # default 128, pretrained => 200 # not currently set
global BATCH_SIZE 
global NUM_EPOCHS 
global MAIN_DROPOUT_KEEP_PROB
global AUX_DROPOUT_KEEP_PROB

# LSTM Hyperparameters
global MAIN_LSTM_SIZE
global AUX_LSTM_SIZE
global COMBI_DENSE_LAYER_DIM

# CNN Hyperparameters
global HIDDEN_DIMS
global FILTER_SIZES
global NUM_FILTERS # this is per filter size; default = 128
# L2_REG_LAMBDA=0.0 # L2 regularization lambda

# Stdout params
global DEBUG
global SAVE_MODEL

# These are TF flags, the first of which doesn't seem to do anything in keras??? and second is rarely used
global ALLOW_SOFT_PLACEMENT
ALLOW_SOFT_PLACEMENT=False
global LOG_DEVICE_PLACEMENT
LOG_DEVICE_PLACEMENT=False

# Non-user-specified globals for use internally
global NUM_POS
global NUM_NEG
     
