import os

from classifiers.MesoNet import classify

classifier_name_default = 'MESO4_DF'
step_save_weights_temp_default = 5

name_classifier = os.getenv("mesonet_classifier")
dir_dataset = os.getenv("path_to_dataset")
batch_size = int(os.getenv("batch_size"))
number_epochs = int(os.getenv("number_epochs"))
step_save_weights_temp = int(os.getenv("step_save_weights_temp"))

evals_learning = classify.learn_from_dir(
    name_classifier=name_classifier,
    dir_dataset=dir_dataset,
    batch_size=batch_size,
    number_epochs=number_epochs,
    step_save_weights_temp=step_save_weights_temp)
evals_learning.print()