import argparse
import extract as ext

from classifiers.MesoNet.classify import learn_from_dir, test_from_dir

classifier_name_default = 'MESO4_DF'

parser = argparse.ArgumentParser(description="Extract faces and warp according to facial landmarks.")
parser.add_argument("--classifier", '-c', required=True, type=str,
                    help="""Can be 'MESO4_DF', 'MESO4_F2F', 'MESOCEPTION_DF', 'MESOCEPTION_F2F'""")

parser.add_argument("--database", '-d', required=True, type=str, help="directory in which to look for input images")
parser.add_argument("--epochs", '-e', required=True, type=str, help="number of epochs")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    name_classifier = args["classifier"]
    dir_database = args["database"]
    number_epochs = int(args["epochs"])
    learn_from_dir(
        name_classifier=name_classifier,
        path_database=dir_database,
        number_epochs=number_epochs
    )