import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from digit_classification_interface import DigitClassificationInterface


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel(DigitClassificationInterface):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CNN().to(device)
        self.is_trained = False

    def train(self):
        # Load MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        # Train the model
        self.model.train()
        for epoch in range(5):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        self.is_trained = True

    def predict(self, image: np.ndarray) -> int:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() method first.")

        if image.shape != (28, 28, 1):
            raise ValueError("Input image must have shape (28, 28, 1)")

        # Preprocess the image
        image = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        image = image.permute(0, 3, 1, 2)
        image = (image - 0.1307) / 0.3081

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        return int(predicted.item())


def test_cnn_model():
    # Create and train the model
    cnn_model = CNNModel()
    cnn_model.train()

    # Load test data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(cnn_model.device), target.to(cnn_model.device)
            outputs = cnn_model.model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy:.4f}')

    # Test on a few individual samples
    num_samples = 10
    for i in range(num_samples):
        image, label = test_dataset[i]
        image_np = image.squeeze().numpy()
        image_np = np.expand_dims(image_np, axis=2)  # Add channel dimension
        predicted = cnn_model.predict(image_np)
        print(f'Sample {i+1}: True label: {label}, Predicted: {predicted}')


if __name__ == "__main__":
    test_cnn_model()