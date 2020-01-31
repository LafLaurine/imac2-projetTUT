import argparse

from classifiers.CapsuleForensics import classify as  clf

parser = argparse.ArgumentParser()

root_checkpoint = 'checkpoints'
step_save_checkpoint_default = 5

parser.add_argument('--classifier', '-c', required=True, type=str, help="""Can only be BINARY_FFPP (for now)""")
parser.add_argument('--dataset',    '-d', required=True, type=str, help='path to root dataset')
parser.add_argument("--batch_size", '-b', required=True, type=int, help="Number of images in each batch")
parser.add_argument("--epochs",     '-e', required=True, type=int, help="Number of epochs")
parser.add_argument("--resume",     '-r', required=True, type=int, help="Which epoch to resume (starting over if 0)")

parser.add_argument("--step",       '-s', required=False, default = step_save_checkpoint_default, type=int, help="Step at which to save temporary weights.")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    name_classifier = args["classifier"]
    dir_database = args["dataset"]
    batch_size = args["batch_size"]
    number_epochs = args["epochs"]
    iteration_resume = args["resume"]
    step_save_checkpoint = args["step"]
    evals_learning = clf.learn_from_dir(
        method_classifier=name_classifier,
        dir_dataset=dir_database,
        iteration_resume=iteration_resume,
        root_checkpoint=root_checkpoint,
        batch_size=batch_size,
        number_epochs=number_epochs,
        step_save_checkpoint=step_save_checkpoint)
    evals_learning.print()
