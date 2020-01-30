import os
import argparse

from classifiers.CapsuleForensics import classify as clf

root_checkpoint = 'checkpoints'
step_save_checkpoint_default = 5

name_classifier = os.getenv("capsule_forensics_classifier")
dir_database = os.getenv("capsule_forensics_train_dataset")
batch_size = int(os.getenv("batch_size"))
number_epochs = int(os.getenv("number_epochs"))
iteration_resume = int(os.getenv("capsule_forensics_epoch_resume"))
step_save_checkpoint = int(os.getenv("step_save_checkpoint"))

evals_learning = clf.learn_from_dir(
    method_classifier=name_classifier,
    dir_dataset=dir_database,
    iteration_resume=iteration_resume,
    root_checkpoint=root_checkpoint,
    batch_size=batch_size,
    number_epochs=number_epochs,
    step_save_checkpoint=step_save_checkpoint)
evals_learning.print()