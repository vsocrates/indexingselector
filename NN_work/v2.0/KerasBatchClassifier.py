from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier

import types


class KerasBatchClassifier(KerasClassifier):
    def __init__(self, train_itr, test_itr, train_steps, test_steps, epochs, callbacks, *args, **kwargs):
        super().__init__()

    def fit(self, X, y, **kwargs):

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


        datagen = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        if 'X_val' in kwargs and 'y_val' in kwargs:
            X_val = kwargs['X_val']
            y_val = kwargs['y_val']

            val_gen = ImageDataGenerator(
                horizontal_flip=True
            )
            val_flow = val_gen.flow(X_val, y_val, batch_size=32)
            val_steps = len(X_val) / 32

            early_stopping = EarlyStopping( patience=5, verbose=5, mode="auto")
            model_checkpoint = ModelCheckpoint("results/best_weights.{epoch:02d}-{loss:.5f}.hdf5", verbose=5, save_best_only=True, mode="auto")
        else:
            val_flow = None
            val_steps = None
            early_stopping = EarlyStopping(monitor="acc", patience=3, verbose=5, mode="auto")
            model_checkpoint = ModelCheckpoint("results/best_weights.{epoch:02d}-{loss:.5f}.hdf5", monitor="acc", verbose=5, save_best_only=True, mode="auto")

        callbacks = [early_stopping, model_checkpoint]

        epochs = self.sk_params['epochs'] if 'epochs' in self.sk_params else 100

        self.__history = self.model.fit_generator(
            datagen.flow(X, y, batch_size=32),  
            steps_per_epoch=len(X) / 32,
            validation_data=val_flow, 
            validation_steps=val_steps, 
            epochs=epochs,
            callbacks=callbacks
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
