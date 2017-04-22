import numpy as np

from argparse import ArgumentParser
from lem2_classifier import LEM2Classifier

PATH = "example_datasets/"

def run_example():
    parser = ArgumentParser(description="Run example")
    parser.add_argument("dataset", help="example dataset")
    parser.add_argument("-a", "--accuracy", type=int, default=0,
        help="Minimum accuracy of displayed induced rules")
    parser.add_argument("-c", "--coverage", type=int, default=0,
        help="Minimum coverage of displayed induced rules")
    args = parser.parse_args()

    data = np.loadtxt(PATH + args.dataset + ".data" , dtype='str', delimiter=',')
    name = np.loadtxt(PATH + args.dataset + ".names", dtype='str', delimiter=',')
    X, y = data[:,0:-1], data[:,len(data[0])-1]
    A, d = name[0:-1], name[len(name)-1]

    lem2 = LEM2Classifier()
    lem2.fit(X, y)
    lem2.print_rules(attr_names=A, class_name=d, min_acc=args.accuracy, min_cov=args.coverage)

if __name__ == "__main__":
    run_example()
