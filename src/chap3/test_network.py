import unittest
from .network import Network


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
