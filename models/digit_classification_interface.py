from abc import ABC, abstractmethod
import numpy as np
import random


class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit in the given image.
        Args:
            image (np.ndarray): Input image of shape (28, 28, 1)
        Returns:
            int: Predicted digit (0-9)
        """

        random.randint(0,9)
