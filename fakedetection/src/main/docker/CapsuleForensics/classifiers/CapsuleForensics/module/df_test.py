import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from tqdm import tqdm
from sklearn import metrics

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from torchvision.datasets import ImageFolder
from .common_classifier import ImageFolderCapsule, get_transform
from ...common_prediction import Prediction, EvaluationTest



def test_from_dataloader(classifier,
                         extractor_vgg,
                         dataloader_test,
                         number_epochs,
                         is_random):
    list_errors = []
    # evaluation mode
    # classifier.eval()
    classifier.train() # TODO: ????
    try:
        for epoch in tqdm(range(number_epochs)):
            error = test_from_dataloader_epoch(classifier=classifier,
                                             extractor_vgg=extractor_vgg,
                                             dataloader_test=dataloader_test,
                                             is_random=is_random)
            list_errors.append(error)
    except KeyboardInterrupt:
        pass
    # adding evaluation
    evals_test = EvaluationTest()
    evals_test.set_error(list_errors)
    return evals_test



def test_from_dataloader_epoch(classifier,
                               extractor_vgg,
                               dataloader_test,
                               is_random,
                               ):
    # next batch
    data_images, data_labels = next(iter(dataloader_test))
    labels_images = data_labels.numpy()

    input_v = Variable(data_images)
    x = extractor_vgg(input_v)
    classes, class_ = classifier(x, random=is_random)

    output_dis = class_.data.cpu()
    output_pred = classifier.infer_pred(output_dis)

    print(output_pred)
    print(labels_images)

    acc_test = metrics.accuracy_score(labels_images, output_pred)
    #todo: CHANGE ERROR RETURN? USE SAME AS MESONET
    #todo: mean -> EER?
    # pred_prob = torch.softmax(output_dis, dim=1)
    # tol_pred_prob = pred_prob[:, 1].data.numpy()
    # fpr, tpr, thresholds = roc_curve(labels_images, tol_pred_prob, pos_label=1)
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    error = 1 - acc_test
    return error

def load_dataloader_test(classifier,
                         path_dataset_test,
                         size_image,
                         batch_size,
                         number_workers
                         ):
    transform_fwd = get_transform(size_image)
    dataset_test = ImageFolderCapsule(classifier=classifier, root=path_dataset_test, transform=transform_fwd)

    dataloader_test = data.DataLoader(dataset_test,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=number_workers)
    return dataloader_test

def load_dataloader_analysis(classifier,
                             path_dir_input,
                             size_image,
                             batch_size,
                             number_workers
                             ):
    transform_fwd = get_transform(size_image)
    # For analysis, we don't need to match our custom labels to input directory
    # So we use ImageFolder and not ImageFolderCapsule
    dataset_analysis = ImageFolder(root=path_dir_input, transform=transform_fwd)
    # This time, we infer the required number of epochs
    # in order to analysis every image
    nb_images = len(dataset_analysis)
    dataloader_analysis = data.DataLoader(dataset_analysis,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=number_workers)
    return dataloader_analysis, nb_images

def analyse_from_dataloader(classifier,
                            extractor_vgg,
                            dataloader_analysis,
                            number_epochs,
                            is_random):
    labels_predicted = np.empty_like([], dtype=np.float)
    # evaluation mode
    # classifier.eval()
    classifier.train() # TODO: ????
    for epoch in tqdm(range(number_epochs)):
        epoch_labels_predicted = analyse_from_dataloader_epoch(classifier=classifier,
                                                               extractor_vgg=extractor_vgg,
                                                               dataloader_analysis=dataloader_analysis,
                                                               is_random=is_random)
        labels_predicted = np.concatenate((labels_predicted, epoch_labels_predicted))
    # adding evaluation
    prediction = Prediction(labels_predicted, classifier.get_dict_classes(), classifier.get_list_classes())
    return prediction

def analyse_from_dataloader_epoch(classifier,
                                  extractor_vgg,
                                  dataloader_analysis,
                                  is_random,
                                  ):
    # The labels inferred by the Dataloader are irrelevant,
    # We can safely ignore them
    data_images, _ = next(iter(dataloader_analysis))
    input_v = Variable(data_images)
    x = extractor_vgg(input_v)
    classes, class_ = classifier(x, random=is_random)

    output_dis = class_.data.cpu()
    output_pred = classifier.infer_pred(output_dis)
    print(output_pred)

    return output_pred