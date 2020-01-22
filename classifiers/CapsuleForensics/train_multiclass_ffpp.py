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
import random
import torch
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

gpu_id_default = -1

number_workers_default=1

beta1_default = 0.9
beta2_default = 0.999

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',                             default ='databases/faceforensicspp',             help='path to root dataset')
parser.add_argument('--train_set',                           default ='train',                                 help='train set')
parser.add_argument('--val_set',                             default ='validation',                            help='validation set')
parser.add_argument('--workers',        type=int,            default = 1,                                      help='number of data loading workers')
parser.add_argument('--batch_size',     type=int,            default=32,                                       help='batch size')
parser.add_argument('--image_size',     type=int,            default=300,                                      help='the height / width of the input image to network')
parser.add_argument('--niter',          type=int,            default=25,                                       help='number of epochs to train for')
parser.add_argument('--lr',             type=float,          default=0.0005,                                   help='learning rate')
parser.add_argument('--beta1',          type=float,          default=beta1_default,                            help='beta1 for adam')
parser.add_argument('--resume',         type=int,            default=0,                                        help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--dir_out',                             default='checkpoints/multiclass_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False,                                    help='disable randomness for routing matrix')
parser.add_argument('--dropout',        type=float,          default=0.05,                                     help='dropout percentage')
parser.add_argument('--manual_seed',    type=int,            default=random.randint(1, 10000),                 help='manual seed')



def get_loaders_learning(manual_seed,
                          learning_rate,
                          dir_out,
                          size_image,
                          path_dataset,
                          set_training,
                          set_validation,
                          name_model='train.csv',
                          number_workers=number_workers_default,
                          gpu_id=gpu_id_default,
                          betas=(beta1, 0.999)
                          ):
    #  =================================================
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    capnet = model_big.CapsuleNet(4, gpu_id)

    optimiser = Adam(capnet.parameters(), lr=learning_rate, betas=betas)

    if resume > 0:
        capnet.load_state_dict(torch.load(os.path.join(dir_out, 'capsule_' + str(resume) + '.pt')))
        capnet.train(mode=True)
        optimiser.load_state_dict(torch.load(os.path.join(dir_out, 'optim_' + str(resume) + '.pt')))

    transform_fwd = transforms.Compose([
        transforms.Resize((size_image, size_image)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_training = dset.ImageFolder(root=os.path.join(path_dataset, set_training), transform=transform_fwd)
    assert dataset_training
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=True,
                                                   num_workers=int(number_workers))

    dataset_validation = dset.ImageFolder(root=os.path.join(path_dataset, set_validation), transform=transform_fwd)
    assert dataset_validation
    dataloader_valuation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False,
                                                 num_workers=int(number_workers))
    return dataloader_training, dataloader_valuation

opt.random = not opt.disable_random

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    #
    method_open = 'a' if (resume > 0) else 'w'
    text_writer = open(os.path.join(dir_out, name_model), method_open)
    vgg_ext = model_big.VggExtractor("""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import random
import torch
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

gpu_id_default = -1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',                             default ='databases/faceforensicspp',             help='path to root dataset')
parser.add_argument('--train_set',                           default ='train',                                 help='train set')
parser.add_argument('--val_set',                             default ='validation',                            help='validation set')
parser.add_argument('--workers',        type=int,            default = 1,                                      help='number of data loading workers')
parser.add_argument('--batch_size',     type=int,            default=32,                                       help='batch size')
parser.add_argument('--image_size',     type=int,            default=300,                                      help='the height / width of the input image to network')
parser.add_argument('--niter',          type=int,            default=25,                                       help='number of epochs to train for')
parser.add_argument('--lr',             type=float,          default=0.0005,                                   help='learning rate')
parser.add_argument('--beta1',          type=float,          default=0.9,                                      help='beta1 for adam')
parser.add_argument('--resume',         type=int,            default=0,                                        help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--dir_out',                             default='checkpoints/multiclass_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False,                                    help='disable randomness for routing matrix')
parser.add_argument('--dropout',        type=float,          default=0.05,                                     help='dropout percentage')
parser.add_argument('--manual_seed',    type=int,            default=random.randint(1, 10000),                 help='manual seed')




def load_(manual_seed,
                          learning_rate,
                          dir_out,
                          size_image,
                          path_dataset,
                          set_training,
                          set_validation,
                          name_model='train.csv',
                          gpu_id=gpu_id_default,
                          betas=(opt.beta1, 0.999)
                          ):
    #  =================================================
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    capnet = model_big.CapsuleNet(4, gpu_id)

    optimiser = Adam(capnet.parameters(), lr=learning_rate, betas=betas)

    if resume > 0:
        capnet.load_state_dict(torch.load(os.path.join(dir_out, 'capsule_' + str(resume) + '.pt')))
        capnet.train(mode=True)
        optimiser.load_state_dict(torch.load(os.path.join(dir_out, 'optim_' + str(resume) + '.pt')))

    transform_fwd = transforms.Compose([
        transforms.Resize((size_image, size_image)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_training = dset.ImageFolder(root=os.path.join(path_dataset, set_training), transform=transform_fwd)
    assert dataset_training
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=True,
                                                   num_workers=int(opt.workers))

    dataset_validation = dset.ImageFolder(root=os.path.join(opt.dataset, set_validation), transform=transform_fwd)
    assert dataset_validation
    dataloader_valuation = torch.utils.data.DataLoader(dataset_validation, batch_size=opt.batchSize, shuffle=False,
                                                 num_workers=int(opt.workers))
    return dataloader_training, dataloader_valuation

opt.random = not opt.disable_random

def train_from dir multiclass_:
    opt = parser.parse_args()
    print(opt)
    #
    train_multiclass_ffpp(manual_seed, )
    #
    method_open = 'a' if (resume > 0) else 'w'
    text_writer = open(os.path.join(dir_out, name_model), method_open)
    vgg_ext = model_big.VggExtractor()
    capsule_loss = model_big.CapsuleLoss(gpu_id)
    # =============================================================

    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):

            img_label = labels_data.numpy().astype(np.float)
            optimizer.zero_grad()

            input_v = Variable(img_data)
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=opt.random, dropout=opt.dropout)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            output_dis = class_.data.cpu()
            _, output_pred = output_dis.max(1)
            output_pred = output_pred.numpy()

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

            img_label = labels_data.numpy().astype(np.float)
            input_v = Variable(img_data)

            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=False)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()
            output_dis = class_.data.cpu().numpy()

            output_dis = class_.data.cpu()
            _, output_pred = output_dis.max(1)

            output_dis = output_dis.numpy()
            output_pred = output_pred.numpy()

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
)
    capsule_loss = model_big.CapsuleLoss(gpu_id)
    # =============================================================

    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):

            img_label = labels_data.numpy().astype(np.float)
            optimizer.zero_grad()

            input_v = Variable(img_data)
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=opt.random, dropout=opt.dropout)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            output_dis = class_.data.cpu()
            _, output_pred = output_dis.max(1)
            output_pred = output_pred.numpy()

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

            img_label = labels_data.numpy().astype(np.float)
            input_v = Variable(img_data)

            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=False)

            loss_dis = capsule_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()
            output_dis = class_.data.cpu().numpy()

            output_dis = class_.data.cpu()
            _, output_pred = output_dis.max(1)

            output_dis = output_dis.numpy()
            output_pred = output_pred.numpy()

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
