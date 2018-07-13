# Import `tensorflow`
import numpy as np
import tensorflow as tf

# # Initialize two constants
# x1 = tf.constant([1,2,3,4])
# x2 = tf.constant([5,6,7,8])

# # Multiply
# result = tf.multiply(x1, x2)

#--------------------------------------

# create a random vector of shape (100,2)
x = np.random.sample((100,2))
# make a dataset from a numpy array
features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
iter = dataset.make_one_shot_iterator()
el = iter.get_next()

#-----------------------------------------

# from generator
sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])
print(sequence)
train_dataset = (np.random.sample((10,2)), np.random.sample((10,1)))
train_part1 = np.random.sample((5,2)).tolist()
train_part2 = np.random.sample((5,3)).tolist()
test_out = train_part1 + train_part2
zip_train_dataset = zip(test_out, train_dataset[1].tolist())
print("test1: ", test_out)
print("test2: ", train_dataset[1].tolist())
# def generator():
    # for el in sequence:
        # yield el

def generator():
    # print('featuresa:  ', features)
    # print('labels:  ', labels)
    for x, y in zip_train_dataset:
      # print("X: ", x)
      # print("Y: ", y)
      
      yield x, y

dataset = tf.data.Dataset().batch(1).from_generator(generator,
                                           output_types= (tf.float32, tf.float32),
                                           output_shapes=( tf.TensorShape([None]),tf.TensorShape([1]) ))

batched_train_dataset = dataset.padded_batch(2, padded_shapes=([None], [1]))
iter = batched_train_dataset.make_initializable_iterator()
el = iter.get_next()


# Initialize Session and run `result`
with tf.Session() as sess:
  # print(sess.run(el))
  # print(sess.run(el))
  # print(sess.run(el))
  # print(sess.run(el))
  
# -----------------------------------------------------
  sess.run(iter.initializer)
  print("eyyyyy: 1", sess.run(el))
  print("eyyyyy: 2", sess.run(el))
  print("eyyyyy: 3", sess.run(el))
  print("eyyyyy: 4", sess.run(el))
  print("eyyyyy: 5", sess.run(el))
  # print("eyyyyy: 6", sess.run(el))
  # print("eyyyyy: 7", sess.run(el))
  # print("eyyyyy: 8", sess.run(el))
  # print("eyyyyy: 9", sess.run(el))
  # print("eyyyyy: 10", sess.run(el))
  