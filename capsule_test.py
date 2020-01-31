import argparse

from classifiers.CapsuleForensics import classify as clf

parser = argparse.ArgumentParser()

batch_size_default = 8

root_checkpoint = 'checkpoints'

parser.add_argument('--classifier', '-c', required=True, help="""Can only be BINARY_FFPP (for now)""")
parser.add_argument('--dataset',    '-d', required=True, help='path to root dataset')
parser.add_argument("--version",    '-v', required=True, type=int, help="Version of the weights to load (has to be > 0)")

parser.add_argument("--batch_size", '-b', required=False, default=batch_size_default, type=int, help="Number of images in each batch")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    name_classifier = args["classifier"]
    dir_database = args["dataset"]
    batch_size = args["batch_size"]
    version_weights = args["version"]
    evals_test = clf.test_from_dir(
        method_classifier=name_classifier,
        dir_dataset=dir_database,
        version_weights=version_weights,
        root_checkpoint=root_checkpoint,
        batch_size=batch_size)
    evals_test.print()
