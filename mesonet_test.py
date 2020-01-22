import argparse

from classifiers.MesoNet import classify

classifier_name_default = 'MESO4_DF'

parser = argparse.ArgumentParser(description="Extract faces and warp according to facial landmarks.")
parser.add_argument("--classifier", '-c', required=True, type=str, help="""Can be 'MESO4_DF', 'MESO4_F2F', 'MESOCEPTION_DF', 'MESOCEPTION_F2F'""")
parser.add_argument("--input",      '-i', required=True, type=str, help="Directory in which to look for images to test.")
parser.add_argument("--batchsize",  '-b', required=True, type=int, help="Size of each batch use for testing.")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    name_classifier = args["classifier"]
    dir_input = args["input"]
    batch_size = args["batchsize"]
    classify.test_from_dir(
        name_classifier=name_classifier,
        dir_input=dir_input,
        batch_size=batch_size
    )