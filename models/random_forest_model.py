import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from digit_classification_interface import DigitClassificationInterface
from tqdm import tqdm
import multiprocessing
from joblib import parallel_backend


class RandomForestModel(DigitClassificationInterface):
    def __init__(self, n_estimators=100, random_state=42):
        self.n_cores = multiprocessing.cpu_count()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=self.n_cores,
                                            verbose=0)
        self.is_trained = False

    def train(self):
        # Load MNIST data
        print("Loading MNIST data...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        # Convert to numpy arrays
        X = train_dataset.data.numpy().reshape(-1, 28 * 28)
        y = train_dataset.targets.numpy()
        X = X.astype('float32') / 255.0
        y = y.astype('int')
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training Random Forest model using {self.n_cores} CPU cores...")
        with parallel_backend('threading', n_jobs=self.n_cores):
            with tqdm(total=self.model.n_estimators, desc="Training progress", unit="tree") as pbar:
                def update_progress(_):
                    pbar.update(1)
                self.model.fit(X_train, y_train)
                for _ in range(self.model.n_estimators):
                    update_progress(None)

        self.is_trained = True
        print("Training completed.")

    def predict(self, image: np.ndarray) -> int:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() method first.")
        if image.shape != (28, 28, 1):
            raise ValueError("Input image must have shape (28, 28, 1)")
        flattened_image = image.reshape(1, -1) / 255.0
        return int(self.model.predict(flattened_image)[0])


def test_random_forest_model():
    rf_model = RandomForestModel()
    rf_model.train()

    # Load test data
    print("Loading test data...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    X_test = test_dataset.data.numpy().reshape(-1, 28 * 28)
    y_test = test_dataset.targets.numpy()
    X_test = X_test.astype('float32') / 255.0
    y_test = y_test.astype('int')
    num_samples = 10
    test_indices = np.random.choice(len(X_test), num_samples, replace=False)
    correct_predictions = 0

    print("Testing model on random samples...")
    for idx in test_indices:
        test_image = X_test[idx].reshape(28, 28, 1)
        true_label = y_test[idx]
        predicted_label = rf_model.predict(test_image)
        print(f"True label: {true_label}, Predicted label: {predicted_label}")
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / num_samples
    print(f"\nAccuracy on {num_samples} random test samples: {accuracy:.2f}")


if __name__ == "__main__":
    test_random_forest_model()