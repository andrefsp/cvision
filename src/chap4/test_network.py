import os
import unittest
from .transfered_network import TransferredNetwork
from .network import Network


CURRENT_PATH = os.path.dirname(__file__)


class TransferredNetworkTestCase(unittest.TestCase):

    def test_build_network(self):
        network = TransferredNetwork()
        prediction = network.predict(os.path.join(CURRENT_PATH, "flower.jpg"))
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.shape, (1, 10))

    def test_build_network_mnist_3d(self):
        network = TransferredNetwork()
        img, target = network.train_sequence[0]
        self.assertEqual(img.shape, (32, 224, 224, 3))
        self.assertEqual(target.shape, (32, 10))

    def test_build_network_train_model(self):
        network = TransferredNetwork()
        network.train_model()
        prediction = network.predict(os.path.join(CURRENT_PATH, "flower.jpg"))
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.shape, (1, 10))


class NetworkTestCase(unittest.TestCase):

    def test_load_data(self):
        net = Network()
        self.assertIsNotNone(net.x_train)
        self.assertIsNotNone(net.y_train)
        self.assertIsNotNone(net.x_test)
        self.assertIsNotNone(net.y_test)

    def test_model_predict(self):
        net = Network()
        y = net.predict_with_model(net.x_train[0:1])
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (1, 10))

    def test_model_train(self):
        net = Network()
        net.train_model()
        y = net.predict_with_model(net.x_train[0:1])
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (1, 10))
