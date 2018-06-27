import numpy as np

n_row = 1000
x1 = np.random.randn(n_row)
x2 = np.random.randn(n_row)
x3 = np.random.randn(n_row)
y_classifier = np.array([1 if (x1[i] + x2[i] + (x3[i])/3 + np.random.randn(1) > 1) else 0 for i in range(n_row)])
y_cts = x1 + x2 + x3/3 + np.random.randn(n_row)
dat = np.array([x1, x2, x3]).transpose()

# Generate indexes of test and train 
idx_list = np.linspace(0,999,num=1000)
idx_test = np.random.choice(n_row, size = 200, replace=False)
idx_train = np.delete(idx_list, idx_test).astype('int')
 
# Split data into test and train
dat_train = dat[idx_train,:]
dat_test = dat[idx_test,:]
y_classifier_train = y_classifier[idx_train]
y_classifier_test = y_classifier[idx_test]
y_cts_train = y_cts[idx_train]
y_cts_test = y_cts[idx_test]

# setup
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.models import Model

metadata_1 = y_classifier + np.random.gumbel(scale = 0.6, size = n_row)
metadata_2 = y_classifier - np.random.laplace(scale = 0.5, size = n_row)
metadata = np.array([metadata_1,metadata_2]).T

# Create training and test set
metadata_train = metadata[idx_train,:]
metadata_test = metadata[idx_test,:]

from tensorflow.python.keras.layers import concatenate

input_dat = Input(shape=(3,)) # for the three columns of dat_train
n_net_layer = Dense(50, activation='relu') # first dense layer
x1 = n_net_layer(input_dat)
x1 = Dropout(0.5)(x1)

input_metadata = Input(shape=(2,))
x2 = Dense(25, activation= 'relu')(input_metadata)
x2 = Dropout(0.3)(x2)

con = concatenate(inputs = [x1,x2] ) # merge in metadata
x3 = Dense(50)(con)
x3 = Dropout(0.3)(x3)
output = Dense(1, activation='sigmoid')(x3)
meta_n_net = Model(inputs=[input_dat, input_metadata], outputs=output)

meta_n_net.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

meta_n_net.fit(x=[dat_train, metadata_train], y=y_classifier_train, epochs=50, verbose=2,
validation_data=([dat_test, metadata_test], y_classifier_test))
meta_n_net.fit(x=[dat_train, metadata_train], y=y_classifier_train, epochs=1, verbose=2,
validation_data=([dat_test, metadata_test], y_classifier_test))

