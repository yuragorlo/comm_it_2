import numpy as np
from models.digit_classification_interface import DigitClassificationInterface
from models.cnn_model import CNNModel
from models.random_forest_model import RandomForestModel
from models.random_model import RandomModel



class DigitClassifier:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm.lower()
        self.model: DigitClassificationInterface

        if self.algorithm == 'cnn':
            self.model = CNNModel()
        elif self.algorithm == 'rf':
            self.model = RandomForestModel()
        elif self.algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("Invalid algorithm. Choose 'cnn', 'rf', or 'rand'.")

    def predict(self, image: np.ndarray) -> int:
        return self.model.predict(image)

    def train(self):
        raise NotImplementedError("Training is not implemented for this classifier.")
