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
import random
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
import model_big

# Share dataset between training and test subsets
class SamplerLearning(torch.utils.data.sampler):
    def __init__(self, mask):
        self.__mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.__mask))

    def __len___(self):
        return len(self.__mask)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to root dataset')
parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--val_set', default ='validation', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=4, help='batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False, help='disable randomness for routing matrix')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout percentage')
parser.add_argument('--seed_manual', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

opt.random = not opt.disable_random

gpu_id_default = -1

number_workers_default=1

beta1_default = 0.9
beta2_default = 0.999
betas_default = (beta1_default, beta2_default)

prop_training_default = 0.9  # in ]0, 1[ : proportion of images to be used in training, the rest in validation


def load_model_checkpoint(weights_path,
                          dir_checkpoint,
                          iteration_resume,
                          learning_rate,
                          betas,
                          gpu_id
                          ):
    capnet = model_big.CapsuleNet(4, gpu_id)
    optimizer = Adam(capnet.parameters(), lr=learning_rate, betas=betas)
    if iteration_resume > 0:
        capnet.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'capsule_' + str(iteration_resume) + '.pt')))
        capnet.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'optim_' + str(iteration_resume) + '.pt')))

    mode_weights = 'a' if iteration_resume > 0 else 'w'
    weights = open(weights_path, mode_weights)
    return capnet, weights, optimizer

def get_random_generator_torch(seed_manual):
    print("Random Seed: ", seed_manual)
    torch.manualSeed(seed_manual)
    return torch.Generator()


def load_dataloaders_learning(size_image,
                          path_dataset,
                          prop_training,
                          batch_size,
                          seed_manual,
                          number_workers=number_workers_default,
                          ):
    #  =================================================

    transform_fwd = transforms.Compose([
        transforms.Resize((size_image, size_image)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # random number generator
    generator = get_random_generator_torch(seed_manual)

    # Computing training and validation datasets from whole directory
    dataset_learning = dset.ImageFolder(path_dataset, transform=transform_fwd)
    number_images = len(dataset_learning)
    number_images_training = int(number_images * prop_training)

    sampler_training = SamplerLearning(torch.randint(0, number_images, size=(1, number_images_training)))
    sampler_validation = SamplerLearning(sampler_training - list(range(0, number_images)))


    transform_fwd = transforms.Compose([
        transforms.Resize((size_image, size_image)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataloader_training = torch.utils.data.DataLoader(dataset_learning,
                                                      batch_size=batch_size,
                                                      sampler = sampler_training,
                                                      shuffle=True,
                                                      num_workers=int(number_workers))

    dataloader_valuation = torch.utils.data.DataLoader(dataset_learning,
                                                       batch_size=batch_size,
                                                       sampler = sampler_validation,
                                                       shuffle=False,
                                                       num_workers=int(number_workers))
    return dataloader_training, dataloader_valuation


def learn_from_dir(dir_dataset,
                   dir_checkpoint,
                   iteration_resume,
                   learning_rate,
                   betas,
                   gpu_id,
                   size_image,
                   prop_training,
                   batch_size,
                   seed_manual,
                   number_workers=number_workers_default,
):
    if seed_manual is None:
        seed_manual = random.randint(1, 10000)
    print("Random Seed: ", opt.seed_manual)
    random.seed(opt.seed_manual)
    torch.seed_manual(opt.seed_manual)

    capnet, weights, optimizer = load_model_checkpoint(weights_path,
                                            dir_checkpoint,
                                            iteration_resume,
                                            learning_rate,
                                            betas,
                                            gpu_id
                                            )


    vgg_ext = model_big.VggExtractor()
    capsule_loss = model_big.CapsuleLoss(opt.gpu_id)

    dataloader_training, dataloader_validation = load_dataloaders_learning(size_image,
                                                                           )

    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float)
            optimizer.zero_grad()

            input_v = Variable(img_data)
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=opt.random, dropout=opt.dropout)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            output_dis = class_.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_train += loss_dis_data
            count += 1


        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        torch.save(capnet.state_dict(), os.path.join(opt.outf, 'capsule_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        capnet.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for img_data, labels_data in dataloader_val:

            labels_data[labels_data > 1] = 1
            img_label = labels_data.numpy().astype(np.float)

            input_v = Variable(img_data)

            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=False)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()
            output_dis = class_.data.cpu().numpy()

            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_test += loss_dis_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()
        capnet.train(mode=True)

    text_writer.close()
