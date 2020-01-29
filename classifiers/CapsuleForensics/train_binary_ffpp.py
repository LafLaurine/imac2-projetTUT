"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
# sys.setrecursionlimit(15000)
import os
import torch.backends.cudnn as cudnn
from . import model_big

from ..common_labels import DICT_LABELS_DF, DICT_LABELS_F2F

from .module import df_learning as lrn

gpu_id_default = -1

number_workers_default = 1


beta1_default = 0.9
beta2_default = 0.999
betas_default = (beta1_default, beta2_default)

root_checkpoint = 'checkpoints'
dir_checkpoint_binary_ffpp_default = "binary_faceforensicspp"
name_model_state_default = 'capsule'
name_optimizer_state_default = 'optim'

learning_rate_default = 0.0001
number_epochs_default = 10
batch_size_default = 20
size_image_default = 256
is_random_default = True,  # what exactly IS random?
perc_dropout_default = 0.05
prop_training_default = 0.9  # in ]0, 1[ : proportion of images to be used in training, the rest in validation
# Share dataset between training and test subsets

dir_checkpoint_binaray_ffpp_default = "binary_faceforensicspp"

log_enabled_default = True

class ClassifierLoader:
    binary_ffpp = "BINARY_FFPP"

    @staticmethod
    def get_classifier(method_classifier,
                       iteration_resume,
                       learning_rate,
                       betas,
                       gpu_id,
                       root_checkpoint=root_checkpoint,
                       name_model_state=name_model_state_default,
                       name_optimizer_state=name_optimizer_state_default,
                       dir_checkpoint=None
                       ):
        switch = {
            ClassifierLoader.binary_ffpp: (load_model_checkpoint,
                                           DICT_LABELS_DF,
                                           dir_checkpoint_binary_ffpp_default)
        }

        tuple_classifier = switch.get(method_classifier, None)
        if tuple_classifier is None:
            raise ValueError("No classifier called: " + method_classifier)
        functor_classifier, dict_labels, dir_checkpoint_default = tuple_classifier
        path_dir = os.path.dirname(os.path.realpath(__file__))
        if dir_checkpoint is None:
            path_dir_checkpoint = os.path.join(path_dir, root_checkpoint, dir_checkpoint_default)
        else:
            path_dir_checkpoint = os.path.join(path_dir, root_checkpoint, dir_checkpoint)
        path_model_state = os.path.join(path_dir_checkpoint, name_model_state)
        path_optimizer_state = os.path.join(path_dir_checkpoint, name_optimizer_state)
        classifier = functor_classifier(path_model_state,
                                        path_optimizer_state,
                                        dict_labels,
                                        iteration_resume,
                                        learning_rate,
                                        betas,
                                        gpu_id)
        return classifier



def load_model_checkpoint(path_model_state,
                          path_optimizer_state,
                          dict_labels,
                          iteration_resume,
                          learning_rate,
                          betas,
                          gpu_id
                          ):
    capnet = model_big.CapsuleNet(4, path_model_state, path_optimizer_state, dict_labels, learning_rate, betas, gpu_id)
    if iteration_resume > 0:
        capnet.load_states(iteration_resume)
        capnet.train(mode=True)
    return capnet


def learn_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   iteration_resume, # 0 to start from scratch
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   batch_size=batch_size_default,
                   size_image=size_image_default,
                   is_random=is_random_default,  # what exactly IS random?
                   perc_dropout=perc_dropout_default,
                   betas=betas_default,
                   gpu_id=gpu_id_default,
                   prop_training=prop_training_default,
                   number_workers=number_workers_default,
                   log_enabled=log_enabled_default
):
    capnet = ClassifierLoader.get_classifier(method_classifier=method_classifier,
                                             root_checkpoint=root_checkpoint,
                                             iteration_resume=iteration_resume,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             gpu_id=gpu_id)


    extractor_vgg = model_big.VggExtractor()
    loss_capnet = model_big.CapsuleLoss(gpu_id)

    dataloader_training, dataloader_validation = lrn.load_dataloaders_learning(classifier=capnet,
                                                                           path_dataset=dir_dataset,
                                                                           prop_training=prop_training,
                                                                           size_image=size_image,
                                                                           batch_size=batch_size,
                                                                           number_workers=number_workers)

    evals_learning = lrn.learn_from_dataloaders(classifier=capnet,
                           loss_classifier=loss_capnet,
                           extractor_vgg=extractor_vgg,
                           dataloader_training=dataloader_training,
                           dataloader_validation=dataloader_validation,
                           epoch_start=iteration_resume,
                           number_epochs=number_epochs,
                           is_random=is_random,
                           perc_dropout=perc_dropout
                           )

    if log_enabled:
        evals_learning.print()
    return
