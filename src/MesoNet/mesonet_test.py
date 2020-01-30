import os
import argparse

from classifiers.MesoNet import classify

classifier_name_default = 'MESO4_DF'

name_classifier = os.getenv("mesonet_classifier")
dir_dataset_test = os.getenv("path_to_dataset")
batch_size = int(os.getenv("batch_size"))

evals_test = classify.test_from_dir(
    name_classifier=name_classifier,
    dir_dataset_test=dir_dataset_test,
    batch_size=batch_size)
evals_test.print()
