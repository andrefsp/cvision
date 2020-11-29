import tensorflow as tf


class Network(object):

    @classmethod
    def normalize(cls, x):
        return x / 255

    @classmethod
    def flatten(cls, x):
        return x.reshape(x.shape[0], 28*28)

    def __init__(self):
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.n_classes = len(set(y))

        self.x_train = self.normalize(self.flatten(x))
        #self.y_train = tf.keras.backend.one_hot(y, self.n_classes)
        self.y_train = y

        # dataset has and input between 1 and 255. We must normalize it
        # so that it is between 0 and 1.
        self.x_test = self.normalize(self.flatten(x_test))
        #self.y_test = tf.keras.backend.one_hot(y_test, self.n_classes)
        self.y_test = y_test

    @property
    def model(self):
        if hasattr(self, '_model'):
            return self._model

        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.layers.Flatten()) # input has been flattened.
        self._model.add(tf.keras.layers.Dense(128, activation='relu'))
        self._model.add(
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        )

        self._model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self._model

    def train_model(self):

        #callbacks = [tf.keras.callbacks.TensorBoard('./keras')]

        self.model.fit(
            self.x_train, self.y_train,
            epochs=5, verbose=1,
            validation_data=(self.x_test, self.y_test),
            #callbacks=callbacks
        )

    def predict_with_model(self, x):
        return self.model.predict(x)

    def estimator_train_input(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        )
        train_dataset = train_dataset.batch(10).repeat()
        return train_dataset

    @property
    def estimator(self):
        if hasattr(self, '_estimator'):
            return self._estimator

        self._estimator = tf.keras.estimator.model_to_estimator(
            self.model, model_dir='./estimator_dir')
        return self._estimator

    def train_estimator(self):
        self.estimator.train(
            self.estimator_train_input, steps=len(self.x_train)/10
        )

    def predict_with_estimator(self, x):
        return self.estimator.predict(x)
