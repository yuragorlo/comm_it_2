import numpy as np
from digit_classification_interface import DigitClassificationInterface

class RandomModel(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError("Input image must have shape (28, 28, 1)")

        center_crop = image[9:19, 9:19, 0]
        return np.random.randint(0, 10)
