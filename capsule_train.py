import argparse

from classifiers.CapsuleForensics import train_binary_ffpp

parser = argparse.ArgumentParser()

batch_size_default = 8

root_checkpoint = 'checkpoints'

parser.add_argument('--classifier', '-c', required=True, help="""Can only be BINARY_FFPP (for now)""")
parser.add_argument('--dataset',    '-d', required=True, help='path to root dataset')
parser.add_argument("--epochs",     '-e', required=True, type=int, help="Number of epochs")
parser.add_argument("--resume",     '-r', required=True, type=int, help="Which epoch to resume (starting over if 0).")

parser.add_argument("--batch_size", '-b', required=False, type=int, help="Number of images in each batch")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    name_classifier = args["classifier"]
    dir_database = args["dataset"]
    batch_size = args["batch_size"]
    number_epochs = args["epochs"]
    iteration_resume = args["resume"]
    train_binary_ffpp.learn_from_dir(
        method_classifier=name_classifier,
        dir_dataset=dir_database,
        iteration_resume=iteration_resume,
        root_checkpoint=root_checkpoint,
        batch_size=batch_size,
        number_epochs=number_epochs,
    )
