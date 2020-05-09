import argparse
from nn_operations import from_model, perform_experiments
import os

parser = argparse.ArgumentParser(description="Neural Network experiments")

parser.add_argument("--load", "-l", dest="load_path", help="Load the trained model from a serialized file. It implies the -t flag.")
parser.add_argument("--save", "-s", dest="save_path", help="Store trained models to the specified directory", default="models")
parser.add_argument("--test", "-t", dest="test", action="store_true", help="Output test data files")

args = parser.parse_args()

test_dir = os.path.join(os.getcwd(), "output")
if args.load_path:
    from_model(os.path.join(os.getcwd(), args.load_path), test_dir=test_dir)
else:
    perform_experiments(store_dir=os.path.join(os.getcwd(), args.save_path), test_model=args.test, test_dir=test_dir)