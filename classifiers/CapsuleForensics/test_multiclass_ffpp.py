"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model_big

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/multiclass_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=13, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(4, opt.gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    capnet.eval()

    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for img_data, labels_data in tqdm(dataloader_test):

        img_label = labels_data.numpy().astype(np.float)
        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt.random)

        output_dis = class_.data.cpu()
        _, output_pred = (output_dis.max(1))

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred.numpy()))
        
        pred_prob = torch.softmax(output_dis, dim=1)

        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count

    print('[Epoch %d] Test acc: %.2f' % (opt.id, acc_test*100))
    text_writer.write('%d,%.2f\n'% (opt.id, acc_test*100))

    text_writer.flush()
    text_writer.close()
