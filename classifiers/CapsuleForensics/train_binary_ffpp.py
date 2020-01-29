"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics
from . import model_big

from ..common_labels import match_labels_dict, DICT_LABELS_DF, DICT_LABELS_F2F

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
dropout_perc_default = 0.05
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

class ImageFolderCapsule(dset.ImageFolder):
    def __init__(self, classifier, *args, **kwargs):
        super(ImageFolderCapsule, self).__init__(*args, **kwargs)
        self.set_labels_idx(classifier)

    def set_labels_idx(self, classifier):
        match_labels_dict(self.class_to_idx, classifier.get_classes())


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

def get_random_generator_torch(seed_manual):
    print("Random Seed: ", seed_manual)
    torch.manual_seed(seed_manual)
    return torch.Generator()

def save_model_checkpoint(capnet, epoch):
    capnet.save_states(epoch)


def load_dataloaders_learning(classifier,
                              path_dataset,
                              prop_training,
                              size_image,
                              batch_size,
                              number_workers
                              ):
    #  =================================================

    transform_fwd = transforms.Compose([
        transforms.Resize((size_image, size_image)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # random number generator
    rng = np.random.default_rng()

    # Computing training and validation datasets from whole directory
    dataset_learning = ImageFolderCapsule(classifier=classifier, root=path_dataset, transform=transform_fwd)
    number_images = len(dataset_learning)
    number_images_training = int(number_images * prop_training)

    list_training = rng.choice(number_images, size=number_images_training, replace=False)
    list_validation = [x for x in range(number_images) if x not in list_training]
    subset_training = data.Subset(dataset_learning, list_training)
    subset_validation = data.Subset(dataset_learning, list_validation)

    dataloader_training = data.DataLoader(subset_training,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=number_workers)
    dataloader_validation = data.DataLoader(subset_validation,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=number_workers)
    return dataloader_training, dataloader_validation




def learn_from_dir(method_classifier,
                   dir_dataset,
                   root_checkpoint,
                   iteration_resume, # 0 to start from scratch
                   number_epochs=number_epochs_default,
                   learning_rate=learning_rate_default,
                   batch_size=batch_size_default,
                   size_image=size_image_default,
                   is_random=is_random_default,  # what exactly IS random?
                   dropout_perc=dropout_perc_default,
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


    vgg_ext = model_big.VggExtractor()
    capsule_loss = model_big.CapsuleLoss(gpu_id)

    dataloader_training, dataloader_validation = load_dataloaders_learning(classifier=capnet,
                                                                           path_dataset=dir_dataset,
                                                                           prop_training=prop_training,
                                                                           size_image=size_image,
                                                                           batch_size=batch_size,
                                                                           number_workers=number_workers)

    for epoch in range(iteration_resume+1, number_epochs+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for data_images, data_labels in tqdm(dataloader_training):
            data_labels[data_labels > 1] = 1
            labels_images = data_labels.numpy().astype(np.float)
            capnet.optimizer.zero_grad()

            input_v = Variable(data_images)
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=is_random, dropout=dropout_perc)

            loss_dis = capsule_loss(classes, Variable(data_labels, requires_grad=False))
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            capnet.optimizer.step()

            output_dis = class_.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i, 1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0
            tol_label = np.concatenate((tol_label, labels_images))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_train += loss_dis_data
            count += 1


        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        save_model_checkpoint(capnet, epoch)

        capnet.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for data_images, data_labels in dataloader_validation:

            data_labels[data_labels > 1] = 1
            labels_images = data_labels.numpy().astype(np.float)

            input_v = Variable(data_images)

            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=False)

            loss_dis = capsule_loss(classes, Variable(data_labels, requires_grad=False))
            loss_dis_data = loss_dis.item()
            output_dis = class_.data.cpu().numpy()

            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i, 1] >= output_dis[i, 0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, labels_images))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_test += loss_dis_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count
        capnet.train(mode=True)

        if log_enabled:
            print('[Epoch {0}] Train loss: {1}   acc: {2} | Test loss: {3}  acc: {4}'.format(epoch, loss_train, acc_train*100, loss_test, acc_test*100))

