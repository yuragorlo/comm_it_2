import unittest
import numpy as np
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from digit_classifier import DigitClassifier
from models.cnn_model import CNNModel
from models.random_forest_model import RandomForestModel
from models.random_model import RandomModel

class TestDigitClassifier(unittest.TestCase):

    def setUp(self):
        self.test_image = np.random.rand(28, 28, 1)

    def test_cnn_classifier_initialization(self):
        classifier = DigitClassifier('cnn')
        self.assertIsInstance(classifier.model, CNNModel)

    def test_rf_classifier_initialization(self):
        classifier = DigitClassifier('rf')
        self.assertIsInstance(classifier.model, RandomForestModel)

    def test_random_classifier_initialization(self):
        classifier = DigitClassifier('rand')
        self.assertIsInstance(classifier.model, RandomModel)

    def test_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            DigitClassifier('invalid_algorithm')

    def test_cnn_predict(self):
        classifier = DigitClassifier('cnn')
        classifier.model.is_trained = True  # Mock training
        prediction = classifier.predict(self.test_image)
        self.assertIsInstance(prediction, int)
        self.assertTrue(0 <= prediction <= 9)

    @patch('models.random_forest_model.RandomForestClassifier')
    def test_rf_predict(self, mock_rf):
        # Create a mock RandomForestClassifier
        mock_rf_instance = mock_rf.return_value
        mock_rf_instance.predict.return_value = np.array([5])  # Mock prediction

        # Create the DigitClassifier with the mocked RandomForestModel
        classifier = DigitClassifier('rf')
        classifier.model.is_trained = True
        classifier.model.model = mock_rf_instance  # Replace the actual model with the mock

        prediction = classifier.predict(self.test_image)
        self.assertIsInstance(prediction, int)
        self.assertTrue(0 <= prediction <= 9)

        # Verify that the predict method was called with the correct shape
        mock_rf_instance.predict.assert_called_once()
        call_arg = mock_rf_instance.predict.call_args[0][0]
        self.assertEqual(call_arg.shape, (1, 784))  # 28*28 = 784

    def test_random_predict(self):
        classifier = DigitClassifier('rand')
        prediction = classifier.predict(self.test_image)
        self.assertIsInstance(prediction, int)
        self.assertTrue(0 <= prediction <= 9)

    def test_train_not_implemented(self):
        classifiers = [DigitClassifier('cnn'), DigitClassifier('rf'), DigitClassifier('rand')]
        for classifier in classifiers:
            with self.assertRaises(NotImplementedError):
                classifier.train()

    def test_predict_without_training(self):
        classifiers = [DigitClassifier('cnn'), DigitClassifier('rf')]
        for classifier in classifiers:
            with self.assertRaises(RuntimeError):
                classifier.predict(self.test_image)


if __name__ == '__main__':
    unittest.main()