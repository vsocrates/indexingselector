from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier

import types


class KerasBatchClassifier(KerasClassifier):
    # def __init__(self, *args, **kwargs):
        # super().__init__(*args, kwargs)
        # self.train_steps = train_steps
        # self.test_steps = test_steps
        # self.epochs = epochs
        # self.callbacks = callbacks
        
    def fit(self, X, y, **kwargs):
    
        self.train_steps = kwargs['train_steps']
        self.test_steps = kwargs['test_steps']
        self.epochs = kwargs['epochs']
        self.callbacks = kwargs['callbacks']

        def make_iterator(dataset, batch_num):
            while True:
              iterator = dataset.make_one_shot_iterator()
              next_val = iterator.get_next()
              for i in range(batch_num):
                try:
                  *inputs, labels = sess.run(next_val)
                  yield inputs, labels  
                except tf.errors.OutOfRangeError:
                  if DEBUG:
                    print("OutOfRangeError Exception Thrown")          
                  break
                except Exception as e: 
                  if DEBUG:
                    print(e)
                    print("Unknown Exception Thrown")
                  break
        itr_train = make_iterator(X, self.train_steps)
        itr_validate = make_iterator(y, self.test_steps)
   

        print("X: ", X)
        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ###################################################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__

        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        # ###############################################################################################################
        
        if 'X_val' in kwargs and 'y_val' in kwargs:
            val_gen = kwargs['y_val']
            
        else:
            val_gen = None
            test_steps = None

        epochs = self.sk_params['epochs'] if 'epochs' in self.sk_params else 100

        self.__history = self.model.fit_generator(X,
            steps_per_epoch=self.train_steps,
            validation_data=val_gen, 
            validation_steps=self.test_steps, 
            epochs=self.epochs,
            callbacks=self.callbacks,
            workers=0,
            verbose=1,
        )

        
        return self.__history

    def score(self, X, y, **kwargs):
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)
        outputs = self.model.evaluate(X, y, **kwargs)
        if type(outputs) is not list:
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise Exception('The model is not configured to compute accuracy. '
                        'You should pass `metrics=["accuracy"]` to '
                        'the `model.compile()` method.')

    @property
    def history(self):
        return self.__history
