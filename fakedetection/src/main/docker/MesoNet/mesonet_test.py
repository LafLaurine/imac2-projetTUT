import os
from classifiers.MesoNet import classify

if __name__ == "__main__":
    name_classifier = os.getenv("mesonet_classifier")
    dir_dataset_test = os.getenv("path_to_dataset")
    batch_size = int(os.getenv("batch_size"))
    number_epochs = int(os.getenv("number_epochs"))
    evals_test = classify.test_from_dir(
        name_classifier=name_classifier,
        dir_dataset_test=dir_dataset_test,
        batch_size=batch_size,
        number_epochs=number_epochs)
    evals_test.print()