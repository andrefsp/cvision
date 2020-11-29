import tensorflow as tf


class Network(object):

    @classmethod
    def normalize(cls, x):
        return x / 255

    def reshape(cls, x):
        return x.reshape(x.shape[0], 28, 28, 1)

    def __init__(self):
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.n_classes = len(set(y))

        self.x_train = self.normalize(self.reshape(x))
        self.y_train = tf.keras.backend.one_hot(y, self.n_classes)

        # dataset has and input between 1 and 255. We must normalize it
        # so that it is between 0 and 1.
        self.x_test = self.normalize(self.reshape(x_test))
        self.y_test = tf.keras.backend.one_hot(y_test, self.n_classes)

    @property
    def model(self):
        if hasattr(self, '_model'):
            return self._model

        self._model = tf.keras.Sequential()
        # first conv
        self._model.add(
            tf.keras.layers.Conv2D(
                6, kernel_size=(5, 5), padding='same', activation='relu',
                input_shape=(28, 28, 1), kernel_regularizer='l2',
            )
        )
        self._model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # second conv
        self._model.add(
            tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu')
        )
        self._model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # FF
        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(120, activation='relu'))
        self._model.add(tf.keras.layers.Dense(84, activation='relu'))

        # output
        self._model.add(
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        )

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        return self._model

    def train_model(self):
        self.model.fit(
            self.x_train, self.y_train,
            epochs=5, verbose=1,
            validation_data=(self.x_test, self.y_test),
        )

    def predict_with_model(self, x):
        return self.model.predict(x)
