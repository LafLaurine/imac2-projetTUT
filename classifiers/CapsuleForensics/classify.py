"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

# sys.setrecursionlimit(15000)
import os

from ..common_labels import DICT_LABELS_DF

from .module import df_learning as lrn, df_test as tst, model_big

gpu_id_default = -1

number_workers_default = 1


beta1_default = 0.9
beta2_default = 0.999
betas_default = (beta1_default, beta2_default)

root_checkpoint = 'checkpoints'
name_model_state_default = 'capsule'
name_optimizer_state_default = 'optim'
dir_checkpoint_binary_ffpp_default = "binary_faceforensicspp"

step_save_checkpoint_default = 5

learning_rate_default = 0.0001
number_epochs_default = 10
batch_size_default = 20
size_image_default = 256
is_random_default = True,  # what exactly IS random?
perc_dropout_default = 0.05
prop_training_default = 0.90  # in ]0, 1[ : proportion of images to be used in training, the rest in validation
# Share dataset between training and test subsets

log_enabled_default = True

class ClassifierLoader:
    binary_ffpp = "BINARY_FFPP"

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
        classifier = functor_classifier(path_model_state=path_model_state,
                                        path_optimizer_state=path_optimizer_state,
                                        dict_labels=dict_labels,
                                        version_weights=version_weights,
                                        learning_rate=learning_rate,
                                        betas=betas,
                                        gpu_id=gpu_id)
        return classifier



def load_model_checkpoint(path_model_state,
                          path_optimizer_state,
                          dict_labels,
                          version_weights,
                          learning_rate,
                          betas,
                          gpu_id
                          ):
    capnet = model_big.CapsuleNet(4, path_model_state, path_optimizer_state, dict_labels, learning_rate, betas, gpu_id)
    if version_weights < 0:
        # WE'VE GOT TO HAVE WEIGHTS TO START WITH
        # OTHERWISE IT DOES FUNKY THINGS COME TEST AND ANALYSIS
        # We can't just not load weights and expect
        raise ValueError("Weights version incorrect.")
    # we load from a previously trained model
    capnet.load_states(version_weights)
    return capnet


def learn_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   iteration_resume, # 0 to start from scratch
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   batch_size=batch_size_default,
                   step_save_checkpoint=step_save_checkpoint_default,
                   size_image=size_image_default,
                   is_random=is_random_default,  # what exactly IS random?
                   perc_dropout=perc_dropout_default,
                   betas=betas_default,
                   gpu_id=gpu_id_default,
                   prop_training=prop_training_default,
                   number_workers=number_workers_default,
):
    capnet = ClassifierLoader.get_classifier(method_classifier=method_classifier,
                                             root_checkpoint=root_checkpoint,
                                             version_weights=iteration_resume,
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
                                                step_save_checkpoint=step_save_checkpoint,
                                                is_random=is_random,
                                                perc_dropout=perc_dropout)

    return evals_learning

def test_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   version_weights, # cannot be 0 this time
                   batch_size=batch_size_default,
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

    extractor_vgg = model_big.VggExtractor()
    # no loss monitoring here

    dataloader_test =  tst.load_dataloader_test(classifier=capnet,
                                                path_dataset_test=dir_dataset,
                                                size_image=size_image,
                                                batch_size=batch_size,
                                                number_workers=number_workers)

    evals_test = tst.test_from_dataloader(classifier=capnet,
                                          extractor_vgg=extractor_vgg,
                                          dataloader_test=dataloader_test,
                                          is_random=is_random)
    return evals_test

def analyse_from_dir(method_classifier,
                   dir_input,
                   root_checkpoint,
                   version_weights, # cannot be 0 this time
                   batch_size=batch_size_default,
                   size_image=size_image_default,
                   is_random=is_random_default,  # what exactly IS random?
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

    extractor_vgg = model_big.VggExtractor()

    dataloader_analysis = tst.load_dataloader_analysis(path_dir_input=dir_input,
                                                       size_image=size_image,
                                                       batch_size=batch_size,
                                                       number_workers=number_workers)

    prediction = tst.analyse_from_dataloader(classifier=capnet,
                                             extractor_vgg=extractor_vgg,
                                             dataloader_analysis=dataloader_analysis,
                                             is_random=is_random)
    return prediction

