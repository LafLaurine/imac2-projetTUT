import argparse

from classifiers.MesoNet import classify

classifier_name_default = 'MESO4_DF'
step_save_weights_temp_default = 5

parser = argparse.ArgumentParser(description="Train MesoNet from directory.")
parser.add_argument("--classifier", '-c', required=True, type=str, help="""Can be any of 'MESO4_DF', 'MESO4_F2F', 'MESOCEPTION_DF', 'MESOCEPTION_F2F'""")
parser.add_argument("--dataset",    '-d', required=True, type=str, help="Path to the dataset directory")
parser.add_argument("--batch_size", '-b', required=True, type=int, help="Number of images in each batch")
parser.add_argument("--epochs",     '-e', required=True, type=int, help="Number of epochs")
parser.add_argument("--step",       '-s', required=False, default = step_save_weights_temp_default, type=int, help="Step at which to save temporary weights.")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    name_classifier = args["classifier"]
    dir_dataset = args["dataset"]
    batch_size = args["batch_size"]
    number_epochs = args["epochs"]
    step_save_weights_temp = args["step"]
    evals_learning = classify.learn_from_dir(
        name_classifier=name_classifier,
        dir_dataset=dir_dataset,
        batch_size=batch_size,
        number_epochs=number_epochs,
        step_save_weights_temp=step_save_weights_temp)
    evals_learning.print()