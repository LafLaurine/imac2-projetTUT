import time

<<<<<<< HEAD
"""
=======
>>>>>>> CapsuleForensics
import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)
<<<<<<< HEAD
"""
=======
>>>>>>> CapsuleForensics

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
parser.add_argument('--dataset', default=os.getenv('path_to_dataset'), help='path to dataset')
# parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to dataset')
# parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
<<<<<<< HEAD
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--image_size', type=int, default=300, help='the height / width of the input image to network')
=======
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
>>>>>>> CapsuleForensics
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')

<<<<<<< HEAD
opt, unknown = parser.parse_known_args()
print(opt)



=======
opt,unknown = parser.parse_known_args()
print(opt)

>>>>>>> CapsuleForensics
@app.route('/')

def hello():
    text_writer = open('CapsuleForensics/'+os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
<<<<<<< HEAD
        transforms.Resize((opt.image_size, opt.image_size)),
=======
        transforms.Resize((opt.imageSize, opt.imageSize)),
>>>>>>> CapsuleForensics
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    arr = os.listdir(os.getcwd()+"/extraction/output")
    bla = ''
    for el in arr:
        bla += el+"-"
    #return bla
    # dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    dataset_test = dset.ImageFolder(root=os.getcwd()+"/extraction/output", transform=transform_fwd)
    assert dataset_test
<<<<<<< HEAD
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.workers))
=======
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
>>>>>>> CapsuleForensics
    
    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(2, opt.gpu_id)
    
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt'), map_location=torch.device('cpu')))
    capnet.eval()
    
    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for img_data, labels_data in tqdm(dataloader_test):

        labels_data[labels_data > 1] = 1
        img_label = labels_data.numpy().astype(np.float)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt.random)

        output_dis = class_.data.cpu()
        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
<<<<<<< HEAD
            if output_dis[i, 1] >= output_dis[i,0]:
=======
            if output_dis[i,1] >= output_dis[i,0]:
>>>>>>> CapsuleForensics
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        pred_prob = torch.softmax(output_dis, dim=1)
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))

        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count

    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
<<<<<<< HEAD
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
=======
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
>>>>>>> CapsuleForensics

    # fnr = 1 - tpr
    # hter = (fpr + fnr)/2

    print('[Epoch %d] Test acc: %.2f   EER: %.2f' % (opt.id, acc_test*100, eer*100))
    text_writer.write('%d,%.2f,%.2f\n'% (opt.id, acc_test*100, eer*100))

    text_writer.flush()
    text_writer.close()

    return 'Hello from Docker'
