
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from torchvision.datasets import ImageFolder
from .common_classifier import ImageFolderCapsule
from ...common_prediction import Prediction, EvaluationTest

def test_from_dataloader(classifier,
                         extractor_vgg,
                         dataloader_test,
                         number_epochs,
                         is_random):
    list_errors = []
    for epoch in tqdm(range(number_epochs)):
        eer = test_from_dataloader_epoch(classifier=classifier,
                                         extractor_vgg=extractor_vgg,
                                         dataloader_test=dataloader_test,
                                         is_random=is_random)
        list_errors.append(eer)
    # adding evaluation
    evals_test = EvaluationTest()
    evals_test.set_error(list_errors)
    return evals_test

def load_dataloader_test(classifier,
                         path_dataset_test,
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
    dataset_test = ImageFolderCapsule(classifier=classifier, root=path_dataset_test, transform=transform_fwd)

    dataloader_test = data.DataLoader(dataset_test,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=number_workers)
    return dataloader_test


def test_from_dataloader_epoch(classifier,
                               extractor_vgg,
                               dataloader_test,
                               is_random,
                               ):
    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)
    # evaluation mode
    classifier.eval()
    data_images, data_labels = next(iter(dataloader_test))
    data_labels[data_labels > 1] = 1
    labels_images = data_labels.numpy().astype(np.float)

    input_v = Variable(data_images)
    x = extractor_vgg(input_v)
    classes, class_ = classifier(x, random=is_random)


    output_dis = class_.data.cpu()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

    for i in range(output_dis.shape[0]):
        if output_dis[i, 1] >= output_dis[i, 0]:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0

    tol_label = np.concatenate((tol_label, labels_images))
    tol_pred = np.concatenate((tol_pred, output_pred))

    pred_prob = torch.softmax(output_dis, dim=1)
    tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:, 1].data.numpy()))
    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    #todo: CHANGE ERROR RETURN? USE SAME AS MESONET
    #todo: mean squared -> EER?
    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def load_dataloader_analysis(path_dir_input,
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

    # For analysis, we don't need to match our custom labels to input directory
    # So we use ImageFolder and not ImageFolderCapsule
    dataset_analysis = ImageFolder(root=path_dir_input, transform=transform_fwd)
    # This time, we infer the required number of epochs
    # To go through in order to analysis every image
    number_images = len(dataset_analysis)
    dataloader_analysis = data.DataLoader(dataset_analysis,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=number_workers)
    return dataloader_analysis

def analyse_from_dataloader(classifier,
                         extractor_vgg,
                         dataloader_analysis,
                         is_random):
    labels_predicted = np.empty_like([], dtype=np.float)
    epoch_labels_predicted = analyse_from_dataloader_epoch(classifier=classifier,
                                                               extractor_vgg=extractor_vgg,
                                                               dataloader_analysis=dataloader_analysis,
                                                               is_random=is_random)
    labels_predicted = np.concatenate((labels_predicted, epoch_labels_predicted))
    # adding evaluation
    prediction = Prediction(labels_predicted, classifier.get_classes())
    return prediction

def analyse_from_dataloader_epoch(classifier,
                                  extractor_vgg,
                                  dataloader_analysis,
                                  is_random,
                                  ):
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)
    # evaluation mode
    classifier.eval()
    # The labels inferred by the Dataloader are irrelevant,
    # We can safely ignore them
    data_images, _ = next(iter(dataloader_analysis))

    input_v = Variable(data_images)

    x = extractor_vgg(input_v)
    classes, class_ = classifier(x, random=is_random)

    output_dis = class_.data.cpu()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

    for i in range(output_dis.shape[0]):
        if output_dis[i, 1] >= output_dis[i, 0]:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0
    tol_pred = np.concatenate((tol_pred, output_pred))
    return tol_pred

