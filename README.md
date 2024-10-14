# Digit Classifier

This project implements a digit classification system using various machine learning algorithms, including Convolutional Neural Networks (CNN), Random Forest (RF), and a random classifier for comparison.

## Project Structure

comm_it_2/\
├── models/\
│ ├── init.py\
│ ├── digit_classification_interface.py\
│ ├── cnn_model.py\
│ ├── random_forest_model.py\
│ └── random_model.py\
├── tests/\
│ └── test_digit_classifier.py\
├── digit_classifier.py\
├── README.md\
└── requirements.txt


## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yuragorlo/comm_it_2.git
   cd comm_it_2
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The main class `DigitClassifier` in `digit_classifier.py` provides an interface to use different classification algorithms:

## Running Tests

To run the unit tests, use the following command from the project root directory:
```
python -m unittest tests/test_digit_classifier.py
```

## Models

- **CNN Model**: Implements a Convolutional Neural Network for digit classification.
- **Random Forest Model**: Uses a Random Forest Classifier for digit recognition.
- **Random Model**: A baseline model that makes random predictions.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request