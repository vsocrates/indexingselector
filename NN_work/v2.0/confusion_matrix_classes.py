import numpy as np

import keras
from keras import metrics
from keras import backend as K

class BinaryTruePositives(keras.layers.Layer):
    """Stateful Metric to count the total true positives over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
        true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives,
                                     true_pos),
                        inputs=[y_true, y_pred])
        return current_true_pos + true_pos
        
class BinaryTrueNegatives(keras.layers.Layer):
    """Stateful Metric to count the total true negatives over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='true_negatives', **kwargs):
        super(BinaryTrueNegatives, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_negatives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_negatives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true negatives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        sum_true_pred = y_true + y_pred# K.cast(K.equal(y_pred, y_true), 'int32')
        true_neg = K.cast(K.sum(K.cast(K.less(sum_true_pred, 1), 'int32')), 'int32')
        current_true_neg = self.true_negatives * 1
        self.add_update(K.update_add(self.true_negatives,
                                     true_neg),
                        inputs=[y_true, y_pred])
        return current_true_neg + true_neg

class BinaryFalsePositives(keras.layers.Layer):
    """Stateful Metric to count the total false positives over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='false_positives', **kwargs):
        super(BinaryFalsePositives, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.false_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.false_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        false_pos = K.cast(K.sum(K.cast(K.greater(y_true, y_pred) ,'int32')), 'int32')
        current_false_pos = self.false_positives * 1
        self.add_update(K.update_add(self.false_positives,
                                     false_pos),
                        inputs=[y_true, y_pred])
        return current_false_pos + false_pos

class BinaryFalseNegatives(keras.layers.Layer):
    """Stateful Metric to count the total true positives over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='false_negatives', **kwargs):
        super(BinaryFalseNegatives, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.false_negatives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.false_negatives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        false_neg = K.cast(K.sum(K.cast(K.greater(y_pred, y_true), 'int32')), 'int32')
        current_false_neg = self.false_negatives * 1
        self.add_update(K.update_add(self.false_negatives,
                                     false_neg),
                        inputs=[y_true, y_pred])
        return current_false_neg + false_neg        

 
        
class Recall(keras.layers.Layer):
    """Stateful Metric to count the total recall over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.stateful = True
        
        self.recall = K.variable(value=0.0, dtype='float32')
        self.true_positives = K.variable(value=0, dtype='int32')
        self.false_negatives = K.variable(value=0, dtype='int32')
    def reset_states(self):
        K.set_value(self.recall, 0.0)
        K.set_value(self.true_positives, 0)
        K.set_value(self.false_negatives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')

        # False negative calculations
        false_neg = K.cast(K.sum(K.cast(K.greater(y_pred, y_true), 'int32')), 'int32')
        current_false_neg = self.false_negatives * 1
        self.add_update(K.update_add(self.false_negatives,
                                     false_neg),
                        inputs=[y_true, y_pred])
        # True positive  calculations
        correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
        true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives,
                                     true_pos),
                        inputs=[y_true, y_pred])
        # Combine
        recall = (K.cast(self.true_positives, 'float32') / (K.cast(self.true_positives, 'float32') + K.cast(self.false_negatives, 'float32') + K.cast(K.epsilon(), 'float32')))
        self.add_update(K.update(self.recall,
                                     recall),
                        inputs=[y_true, y_pred])
        
        return recall        

class Precision(keras.layers.Layer):
    """Stateful Metric to count the total precision over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.stateful = True
        
        self.precision = K.variable(value=0.0, dtype='float32')
        self.true_positives = K.variable(value=0, dtype='int32')
        self.false_positives = K.variable(value=0, dtype='int32')
    def reset_states(self):
        K.set_value(self.precision, 0.0)
        K.set_value(self.true_positives, 0)
        K.set_value(self.false_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')

        # False positives calculations
        false_pos = K.cast(K.sum(K.cast(K.greater(y_true, y_pred) ,'int32')), 'int32')
        current_false_pos = self.false_positives * 1
        self.add_update(K.update_add(self.false_positives,
                                     false_pos),
                        inputs=[y_true, y_pred])

        # True positive  calculations
        correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
        true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives,
                                     true_pos),
                        inputs=[y_true, y_pred])
        # Combine
        precision = (K.cast(self.true_positives, 'float32') / (K.cast(self.true_positives, 'float32') + K.cast(self.false_positives, 'float32') + K.cast(K.epsilon(), 'float32')))
        self.add_update(K.update(self.precision,
                                     precision),
                        inputs=[y_true, y_pred])
        
        return precision        
        
class F1Score(keras.layers.Layer):
    """Stateful Metric to calculate the F1Score over all batches.

    Assumes predictions and targets of shape `(samples, 1)`.

    # Arguments
        name: String, name for the metric.
    """

    def __init__(self, name='F1Score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.stateful = True
        
        self.F1Score = K.variable(value=0.0, dtype='float32')
        self.true_positives = K.variable(value=0, dtype='int32')
        self.false_positives = K.variable(value=0, dtype='int32')
        self.false_negatives = K.variable(value=0, dtype='int32')
        
    def reset_states(self):
        K.set_value(self.F1Score, 0.0)
        K.set_value(self.true_positives, 0)
        K.set_value(self.false_positives, 0)
        K.set_value(self.false_negatives, 0)
        
    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.

        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions

        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')

        # False negative calculations
        false_neg = K.cast(K.sum(K.cast(K.greater(y_pred, y_true), 'int32')), 'int32')
        current_false_neg = self.false_negatives * 1
        self.add_update(K.update_add(self.false_negatives,
                                     false_neg),
                        inputs=[y_true, y_pred])
        
        # False positives calculations
        false_pos = K.cast(K.sum(K.cast(K.greater(y_true, y_pred) ,'int32')), 'int32')
        current_false_pos = self.false_positives * 1
        self.add_update(K.update_add(self.false_positives,
                                     false_pos),
                        inputs=[y_true, y_pred])

        # True positive  calculations
        correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
        true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives,
                                     true_pos),
                        inputs=[y_true, y_pred])
        # Combine
        F1Score = (( 2.0 * K.cast(self.true_positives, 'float32')) / (2.0 * K.cast(self.true_positives, 'float32') + K.cast(self.false_positives, 'float32') + K.cast(self.false_negatives, 'float32') + K.cast(K.epsilon(), 'float32')))
        self.add_update(K.update(self.F1Score,
                                     F1Score),
                        inputs=[y_true, y_pred])
        
        return F1Score        

