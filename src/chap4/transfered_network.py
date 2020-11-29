import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt


class NmistSequence(Sequence):
    def show_img(self, img):
        plt.imshow(img)
        plt.show()

    def convert_img(self, mnist_img):
        new_img = Image.fromarray(
            np.zeros(28 * 28 * 3).reshape((28, 28, 3)), 'RGB'
        )
        h, w = mnist_img.shape
        for hi in range(h):
            for wi in range(w):
                val = mnist_img[hi][wi]
                new_img.putpixel((wi, hi), (val, val, val))

        return preprocess_input(
            tf.keras.preprocessing.image.img_to_array(
                new_img.resize((224, 224))
            )
        )

    def __len__(self):
        return int(len(self.x_raw)/self.batch_size)

    def __getitem__(self, idx):
        lower_idx = idx * self.batch_size
        top_idx = (idx + 1) * self.batch_size
        y = np.array(self.y_raw[lower_idx:top_idx])
        x = np.array([
            self.convert_img(img)
            for img in self.x_raw[lower_idx:top_idx]
        ])
        return (x, y)

    def __init__(self, x, y, n_classes, batch_size=32):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.x_raw = x
        self.y_raw = tf.keras.backend.one_hot(y, self.n_classes)


class TransferredNetwork(object):
    """
        Used VGG16 at the initial block for transfer learning
    """

    def __init__(self, batch_size=32):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.n_classes = len(set(y_train))
        self.batch_size = batch_size

        self.train_sequence = NmistSequence(
            x_train, y_train, self.n_classes, batch_size=batch_size
        )
        self.test_sequence = NmistSequence(
            x_test, y_test, self.n_classes, batch_size=batch_size
        )[0]

    @property
    def model(self):
        if hasattr(self, '_model'):
            return self._model

        self.input = tf.keras.layers.Input((224, 224, 3))

        self._model = tf.keras.applications.VGG16(
            include_top=False, weights='imagenet', input_tensor=self.input,
            pooling='max'
        )

        # self._model = inception(self._model.output).layer

        self._model = tf.keras.layers.Flatten()(self._model.output)
        self._model = tf.keras.layers.Dense(120, activation='relu')(self._model)
        self._model = tf.keras.layers.Dense(84, activation='relu')(self._model)

        self._model = tf.keras.layers.Dense(
            self.n_classes, activation='softmax'
        )(self._model)

        self._model = tf.keras.models.Model(
            inputs=self.input, outputs=self._model
        )

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            # run_eagerly=True
        )
        return self._model

    def preprocess(self, obj):
        if isinstance(obj, str):
            img = Image.open(obj).resize((224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)
        return preprocess_input(img)

    def predict(self, objs):
        if isinstance(objs, str):
            objs = [objs]
        imgs = np.array([self.preprocess(obj) for obj in objs])
        return self.model(imgs)

    def train_model(self):
        self.model.fit(
            x=self.train_sequence,
            epochs=1, verbose=1,
            batch_size=32,
            validation_data=(self.test_sequence[0], self.test_sequence[1]),
        )
