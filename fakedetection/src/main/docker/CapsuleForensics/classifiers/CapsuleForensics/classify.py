"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import os

from ..common_config import DICT_LABELS_DF, \
                            DICT_LABELS_F2F, \
                            DICT_LABELS_FACESWAP, \
                            DICT_LABELS_MULTICLASS, \
                            LIST_LABELS_DF, \
                            LIST_LABELS_F2F, \
                            LIST_LABELS_FACESWAP, \
                            LIST_LABELS_MULTICLASS

from .module import df_learning as lrn, df_test as tst, model_big as mb


gpu_id_default = -1

number_workers_default = 1

NUM_CLASS = 2


beta1_default = 0.9
beta2_default = 0.999
betas_default = (beta1_default, beta2_default)

root_checkpoint = 'checkpoints'
name_model_state_default = 'capsule'
name_optimizer_state_default = 'optim'
dir_checkpoint_binary_df_default = "binary_df"
dir_checkpoint_binary_f2f_default = "binary_f2f"
dir_checkpoint_binary_fs_default = "binary_faceswap"
dir_checkpoint_multiclass_default = "multiclass"

step_save_checkpoint_default = 5

learning_rate_default = 0.0001
number_epochs_default = 10
batch_size_default = 20
size_image_default = 512
is_random_default = True  # what exactly IS random?
perc_dropout_default = 0.05
prop_training_default = 0.90  # in ]0, 1[ : proportion of images to be used in training, the rest in validation
# Share dataset between training and test subsets

log_enabled_default = True


class ClassifierLoader:
    BINARY_DF  = "BINARY_DF"
    BINARY_F2F = "BINARY_F2F"
    BINARY_FS = "BINARY_FACESWAP"
    MULTICLASS = "MULTICLASS"
    @staticmethod
    def get_classifier(method_classifier,
                       version_weights,
                       learning_rate,
                       betas,
                       gpu_id,
                       root_checkpoint=root_checkpoint,
                       name_model_state=name_model_state_default,
                       name_optimizer_state=name_optimizer_state_default,
                       dir_checkpoint=None
                       ):
        switch = {
            ClassifierLoader.BINARY_DF: (DICT_LABELS_DF,
                                         LIST_LABELS_DF,
                                         dir_checkpoint_binary_df_default),
            ClassifierLoader.BINARY_F2F: (DICT_LABELS_F2F,
                                          LIST_LABELS_F2F,
                                          dir_checkpoint_binary_f2f_default),
            ClassifierLoader.BINARY_FS: (DICT_LABELS_FACESWAP,
                                         LIST_LABELS_FACESWAP,
                                         dir_checkpoint_binary_fs_default),
            ClassifierLoader.MULTICLASS: (DICT_LABELS_MULTICLASS,
                                          LIST_LABELS_MULTICLASS,
                                          dir_checkpoint_multiclass_default)
        }

        tuple_classifier = switch.get(method_classifier, None)
        if tuple_classifier is None:
            raise ValueError("No classifier called: " + method_classifier)
        dict_labels, list_labels, dir_checkpoint_default = tuple_classifier
        path_dir = os.path.dirname(os.path.realpath(__file__))
        if dir_checkpoint is None:
            path_dir_checkpoint = os.path.join(path_dir, root_checkpoint, dir_checkpoint_default)
        else:
            path_dir_checkpoint = os.path.join(path_dir, root_checkpoint, dir_checkpoint)
        path_model_state = os.path.join(path_dir_checkpoint, name_model_state)
        path_optimizer_state = os.path.join(path_dir_checkpoint, name_optimizer_state)
        classifier = mb.CapsuleNet(list_labels=list_labels,
                                   num_class=len(list_labels),
                                   path_model_state=path_model_state,
                                   path_optimizer_state=path_optimizer_state,
                                   dict_labels=dict_labels,
                                   learning_rate=learning_rate,
                                   betas=betas,
                                   gpu_id=gpu_id)
        load_model_checkpoint(classifier, version_weights)
        return classifier



def load_model_checkpoint(classifier, version_weights):
    if version_weights < 0:
        # WE'VE GOT TO HAVE WEIGHTS TO START WITH
        # OTHERWISE IT DOES FUNKY THINGS COME TEST AND ANALYSIS
        # We can't just not load weights and expect
        raise ValueError("Weights version incorrect.")
    # we load from a previously trained model
    elif version_weights > 0:
        classifier.load_states(version_weights)
    return classifier


def learn_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   iteration_resume,
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   batch_size=batch_size_default,
                   step_save_checkpoint=step_save_checkpoint_default,
                   size_image=size_image_default,
                   is_random=is_random_default, 
                   perc_dropout=perc_dropout_default,
                   betas=betas_default,
                   gpu_id=gpu_id_default,
                   prop_training=prop_training_default,
                   number_workers=number_workers_default,
):
    classifier = ClassifierLoader.get_classifier(method_classifier=method_classifier,
                                             root_checkpoint=root_checkpoint,
                                             version_weights=iteration_resume,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             gpu_id=gpu_id)
    extractor_vgg = mb.VggExtractor()
    loss_capnet = mb.CapsuleLoss(gpu_id)

    dataloader_training, dataloader_validation = lrn.load_dataloaders_learning(classifier=classifier,
                                                                               path_dataset=dir_dataset,
                                                                               prop_training=prop_training,
                                                                               size_image=size_image,
                                                                               batch_size=batch_size,
                                                                               number_workers=number_workers)
    evals_learning = lrn.learn_from_dataloaders(classifier=classifier,
                                                loss_classifier=loss_capnet,
                                                extractor_vgg=extractor_vgg,
                                                dataloader_training=dataloader_training,
                                                dataloader_validation=dataloader_validation,
                                                epoch_start=iteration_resume,
                                                number_epochs=number_epochs,
                                                step_save_checkpoint=step_save_checkpoint,
                                                is_random=is_random,
                                                perc_dropout=perc_dropout)

    return evals_learning

def test_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   version_weights, # cannot be 0 this time
                   batch_size=batch_size_default,
                   number_epochs=number_epochs_default,
                   size_image=size_image_default,
                   is_random=is_random_default,  # what exactly IS random?
                   betas=betas_default,
                   gpu_id=gpu_id_default,
                   number_workers=number_workers_default,
                   ):
    capnet = ClassifierLoader.get_classifier(method_classifier=method_classifier,
                                             root_checkpoint=root_checkpoint,
                                             version_weights=version_weights,
                                             learning_rate=0, # don't need no education
                                             betas=betas,
                                             gpu_id=gpu_id)

    extractor_vgg = mb.VggExtractor()
    # no loss monitoring here

    dataloader_test = tst.load_dataloader_test(classifier=capnet,
                                                path_dataset_test=dir_dataset,
                                                size_image=size_image,
                                                batch_size=batch_size,
                                                number_workers=number_workers)

    evals_test = tst.test_from_dataloader(classifier=capnet,
                                          extractor_vgg=extractor_vgg,
                                          dataloader_test=dataloader_test,
                                          number_epochs=number_epochs,
                                          is_random=is_random)
    return evals_test

def analyse_from_dir(method_classifier,
                   dir_input,
                   root_checkpoint,
                   version_weights, # cannot be 0 this time
                   batch_size=batch_size_default,
                   size_image=size_image_default,
                   is_random=is_random_default, 
                   betas=betas_default,
                   gpu_id=gpu_id_default,
                   number_workers=number_workers_default,
                   ):
    capnet = ClassifierLoader.get_classifier(method_classifier=method_classifier,
                                             root_checkpoint=root_checkpoint,
                                             version_weights=version_weights,
                                             learning_rate=0,
                                             betas=betas,
                                             gpu_id=gpu_id)

    extractor_vgg = mb.VggExtractor()
    dataloader_analysis, nb_images = tst.load_dataloader_analysis(classifier=capnet,
                                                                  path_dir_input=dir_input,
                                                                  size_image=size_image,
                                                                  batch_size=batch_size,
                                                                  number_workers=number_workers)
    number_epochs = nb_images // batch_size + 1
    prediction = tst.analyse_from_dataloader(classifier=capnet,
                                             extractor_vgg=extractor_vgg,
                                             dataloader_analysis=dataloader_analysis,
                                             number_epochs=number_epochs,
                                             is_random=is_random)
    return prediction