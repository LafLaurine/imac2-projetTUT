
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics


from .common_classifier import ImageFolderCapsule
from ...common_prediction import EvaluationLearning

def save_model_checkpoint(capnet, epoch):
    capnet.save_states(epoch)


def get_random_generator_torch(seed_manual):
    print("Random Seed: ", seed_manual)
    torch.manual_seed(seed_manual)
    return torch.Generator()

def learn_from_dataloaders(classifier,
                           loss_classifier,
                           extractor_vgg,
                           dataloader_training,
                           dataloader_validation,
                           epoch_start,
                           number_epochs,
                           step_save_checkpoint,
                           is_random,
                           perc_dropout):
    evals_learning = EvaluationLearning()
    epoch_end = epoch_start + number_epochs
    i = 1
    # TRAINING MODE #
    classifier.train()
    try:
        for epoch in tqdm(range(epoch_start + 1, epoch_end + 1)):
            loss_training, acc_training = train_from_dataloader_epoch(classifier=classifier,
                                                                      loss_classifier=loss_classifier,
                                                                      extractor_vgg=extractor_vgg,
                                                                      dataloader_training=dataloader_training,
                                                                      is_random=is_random,
                                                                      perc_dropout=perc_dropout)

            if i % step_save_checkpoint == 0 or epoch == epoch_end:
                    save_model_checkpoint(classifier, epoch)
            i += 1
            loss_validation, acc_validation = validate_from_dataloader_epoch(classifier=classifier,
                                                                             loss_classifier=loss_classifier,
                                                                             extractor_vgg=extractor_vgg,
                                                                             dataloader_validation=dataloader_validation)
            # adding evaluation
            evals_learning.add_eval(epoch=epoch,
                                    loss_training=loss_training,
                                    acc_training=acc_training,
                                    loss_validation=loss_validation,
                                    acc_validation=acc_validation)
    except KeyboardInterrupt:
        pass
    return evals_learning

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
    number_images_tot = len(dataset_learning)
    number_images_training = int(number_images_tot * prop_training)
    list_learning = rng.choice(number_images_tot, size=number_images_tot, replace=False)
    list_training = list_learning[:number_images_training]
    list_validation = list_learning[number_images_training:]
    subset_training = data.Subset(dataset_learning, list_training)
    subset_validation = data.Subset(dataset_learning, list_validation)

    dataloader_training = data.DataLoader(subset_training,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=number_workers)
    dataloader_validation = data.DataLoader(subset_validation,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=number_workers)
    return dataloader_training, dataloader_validation


def train_from_dataloader_epoch(classifier,
                                loss_classifier,
                                extractor_vgg,
                                dataloader_training,
                                is_random,
                                perc_dropout
                                ):
    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    loss_training = 0
    # Training mode
    classifier.train()
    data_images, data_labels = next(iter(dataloader_training))
    ### Processing batch ###
    data_labels[data_labels > 1] = 1
    labels_images = data_labels.numpy().astype(np.float)

    classifier.optimizer.zero_grad()
    input_v = Variable(data_images)
    x = extractor_vgg(input_v)
    classes, class_ = classifier(x, random=is_random, dropout=perc_dropout)

    loss_dis = loss_classifier(classes, Variable(data_labels, requires_grad=False))
    loss_dis_data = loss_dis.item()
    loss_dis.backward()
    classifier.optimizer.step()

    output_dis = class_.data.cpu().numpy()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

    for i in range(output_dis.shape[0]):
        if output_dis[i, 1] >= output_dis[i, 0]:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0
    tol_label = np.concatenate((tol_label, labels_images))
    tol_pred = np.concatenate((tol_pred, output_pred))
    loss_training += loss_dis_data
    acc_training = metrics.accuracy_score(tol_label, tol_pred)
    print(tol_pred)
    print(tol_label)
    return loss_training, acc_training

def validate_from_dataloader_epoch(classifier,
                                   loss_classifier,
                                   extractor_vgg,
                                   dataloader_validation):
    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    loss_validation = 0

    # Evaluation mode
    data_images, data_labels = next(iter(dataloader_validation))
    ## BATCH ##
    data_labels[data_labels > 1] = 1
    labels_images = data_labels.numpy().astype(np.float)

    classifier.optimizer.zero_grad()
    input_v = Variable(data_images)
    x = extractor_vgg(input_v)
    classes, class_ = classifier(x, random=False)

    loss_dis = loss_classifier(classes, Variable(data_labels, requires_grad=False))
    loss_dis_data = loss_dis.item()
    loss_dis.backward()
    classifier.optimizer.step()

    output_dis = class_.data.cpu().numpy()
    output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

    for i in range(output_dis.shape[0]):
        if output_dis[i, 1] >= output_dis[i, 0]:
            output_pred[i] = 1.0
        else:
            output_pred[i] = 0.0

    tol_label = np.concatenate((tol_label, labels_images))
    tol_pred = np.concatenate((tol_pred, output_pred))

    loss_validation += loss_dis_data
    acc_validation = metrics.accuracy_score(tol_label, tol_pred)
    print(tol_pred)
    print(tol_label)
    return loss_validation, acc_validation
