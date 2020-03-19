import os
from classifiers.MesoNet import classify

batch_size_default = 4

if __name__ == "__main__":
    name_classifier = os.getenv("mesonet_classifier")
    dir_input = os.getenv("path_to_dataset")
    batch_size = os.getenv("batch_size")
    prediction = classify.analyse_from_dir(
        name_classifier=name_classifier,
        dir_input=dir_input,
        batch_size=batch_size
        )
    prediction.print()