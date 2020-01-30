import os

from classifiers.CapsuleForensics import classify as clf

root_checkpoint = 'checkpoints'

name_classifier = os.getenv("capsule_forensics_classifier")
dir_input = os.getenv("path_to_dataset")
batch_size = int(os.getenv("batch_size"))
version_weights = int(os.getenv("version_weights"))

prediction = clf.analyse_from_dir(
    method_classifier=name_classifier,
    dir_input=dir_input,
    version_weights=version_weights,
    root_checkpoint=root_checkpoint,
    batch_size=batch_size)
prediction.print()

