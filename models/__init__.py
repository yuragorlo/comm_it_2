import unittest
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')

    runner = unittest.TextTestRunner()
    runner.run(test_suite)