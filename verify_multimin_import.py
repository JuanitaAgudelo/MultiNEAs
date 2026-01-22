import sys
import os

# Add src to python path to emulate package installed or editable install
sys.path.insert(0, os.path.abspath('src'))

try:
    import multineas.multimin
    print("Successfully imported multineas.multimin")
    print(f"Module file: {multineas.multimin.__file__}")
except ImportError as e:
    print(f"Failed to import multineas.multimin: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
