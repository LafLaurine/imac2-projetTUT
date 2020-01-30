import os

from classifiers.MesoNet import classify

batch_size_default = 4

name_classifier = os.getenv("mesonet_classifier")
dir_input = os.getenv("path_to_dataset")
prediction = classify.analyse_from_dir(
    name_classifier=name_classifier,
    dir_input=dir_input
)
prediction.print()