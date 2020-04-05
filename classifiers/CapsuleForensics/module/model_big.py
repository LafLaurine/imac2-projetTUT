"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for Capsule-Forensics-v2 model
"""

import os
import sys
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam
import numpy as np

from ...common_config import LABEL_DF, LABEL_REAL, LABEL_F2F, LABEL_FACESWAP

NO_CAPS=10
""" 
Apparently eval mode gets funky
fur a few epochs unless
we change the 'momentum of the BatchNorm
so here goes
"""
MOMENTUM_BATCHNORM = 0.4

class StatsNet(nn.Module):
    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2]*x.data.shape[3])

        mean = torch.mean(x, 2)
        std = torch.std(x, 2)

        return torch.stack((mean, std), dim=1)

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class VggExtractor(nn.Module):
    def __init__(self):
        super(VggExtractor, self).__init__()

        self.vgg_1 = self.Vgg(models.vgg19(pretrained=True), 0, 18)
        self.vgg_1.eval()

    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin:(end+1)])
        return features

    def forward(self, input):
        return self.vgg_1(input)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64, momentum=MOMENTUM_BATCHNORM),
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16, momentum=MOMENTUM_BATCHNORM),
                nn.ReLU(),
                StatsNet(),

                nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(8, momentum=MOMENTUM_BATCHNORM),
                nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(1, momentum=MOMENTUM_BATCHNORM),
                View(-1, 8),
                )
                for _ in range(NO_CAPS)]
        )

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x):
        outputs = [capsule(x.detach()) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)

        return self.squash(output, dim=-1)

class RoutingLayer(nn.Module):
    def __init__(self, gpu_id, num_input_capsules, num_output_capsules, data_in, data_out, num_iterations):
        super(RoutingLayer, self).__init__()

        self.gpu_id = gpu_id
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(num_output_capsules, num_input_capsules, data_out, data_in))


    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x, random, dropout):
        # x[b, data, in_caps]

        x = x.transpose(2, 1)
        # x[b, in_caps, data]

        if random:
            noise = Variable(0.01*torch.randn(*self.route_weights.size()))
            route_weights = self.route_weights + noise
        else:
            route_weights = self.route_weights

        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]

        # route_weights [out_caps , 1 , in_caps , data_out , data_in]
        # x             [   1     , b , in_caps , data_in ,    1    ]
        # priors        [out_caps , b , in_caps , data_out,    1    ]

        priors = priors.transpose(1, 0)
        # priors[b, out_caps, in_caps, data_out, 1]

        if dropout > 0.0:
            drop = Variable(torch.FloatTensor(*priors.size()).bernoulli(1.0- dropout))
            priors = priors * drop
            

        logits = Variable(torch.zeros(*priors.size()))
        # logits[b, out_caps, in_caps, data_out, 1]

        num_iterations = self.num_iterations

        for i in range(num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)

            if i != self.num_iterations - 1:
                delta_logits = priors * outputs
                logits = logits + delta_logits

        # outputs[b, out_caps, 1, data_out, 1]
        outputs = outputs.squeeze()

        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous() 
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        # outputs[b, data_out, out_caps]

        return outputs



class CapsuleNet(nn.Module):
    # added optimiser to class for convenience
    # optimizer : Adam etc.
    # added paths to saved states of model and optimiser
    # __path_model_state
    # __path_optimizer_state
    # added dict and list for class labels
    # __dict_labels
    # __list_labels
    def __init__(self, num_class, path_model_state, path_optimizer_state, dict_labels, list_labels, learning_rate, betas, gpu_id):
        super(CapsuleNet, self).__init__()

        self.num_class = num_class
        self.fea_ext = FeatureExtractor()
        self.fea_ext.apply(self.weights_init)

        self.routing_stats = RoutingLayer(gpu_id=gpu_id,
                                          num_input_capsules=NO_CAPS,
                                          num_output_capsules=num_class,
                                          data_in=8,
                                          data_out=4,
                                          num_iterations=2)

        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=betas)
        self.__dict_labels = dict_labels
        self.__list_labels = list_labels
        self.__path_model_state = path_model_state
        self.__path_optimizer_state = path_optimizer_state


    def get_dict_classes(self):
        return self.__dict_labels

    def get_list_classes(self):
        return self.__list_labels

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def __get_filenames_states(self, epoch):
        filename_model_state = self.__path_model_state + '_{0}.pt'.format(epoch)
        filename_optimizer_state = self.__path_optimizer_state + '_{0}.pt'.format(epoch)
        return filename_model_state, filename_optimizer_state

    def load_states(self, epoch):
        # setting paths to states
        filename_model_state, filename_optimizer_state = self.__get_filenames_states(epoch)
        self.load_state_dict(torch.load(filename_model_state))
        self.optimizer.load_state_dict(torch.load(filename_optimizer_state))
        self.train(mode=True)

    def save_states(self, epoch):
        filename_model_state, filename_optimizer_state = self.__get_filenames_states(epoch)
        dir_model_state = os.path.dirname(filename_model_state)
        dir_optimizer_state = os.path.dirname(filename_optimizer_state)
        # the path to this state might not exist, we create it
        if not os.path.isdir(dir_model_state) :
            os.makedirs(dir_model_state)
        if not os.path.isdir(dir_optimizer_state) :
            os.makedirs(dir_optimizer_state);
        torch.save(self.state_dict(), filename_model_state)
        torch.save(self.optimizer.state_dict(), filename_optimizer_state)


    def forward(self, x, random=False, dropout=0.0):

        z = self.fea_ext(x)
        z = self.routing_stats(z, random, dropout=dropout)
        # z[b, data, out_caps]

        classes = F.softmax(z, dim=-1)

        class_ = classes.detach()
        class_ = class_.mean(dim=1)

        return classes, class_

    def process_batch(self, data_images, data_labels, extractor_vgg, loss_classifier, is_random, perc_dropout):
        labels_images = data_labels.numpy().astype(np.float)

        self.optimizer.zero_grad()
        input_v = Variable(data_images)
        x = extractor_vgg(input_v)
        classes, class_ = self(x, random=is_random, dropout=perc_dropout)

        loss_dis = loss_classifier(classes, Variable(data_labels, requires_grad=False))
        loss_dis.backward()
        self.optimizer.step()

        output_dis = class_.data.cpu().numpy()
        return self.infer_pred(output_dis)


    def infer_pred(self, output_dis):
        _, output_pred_temp = output_dis.max(1)
        output_pred = output_pred_temp.numpy()
        return output_pred


class CapsuleLoss(nn.Module):
    def __init__(self, gpu_id):
        super(CapsuleLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, classes, labels):
        loss_t = self.cross_entropy_loss(classes[:,0,:], labels)

        for i in range(classes.size(1) - 1):
            loss_t = loss_t + self.cross_entropy_loss(classes[:,i+1,:], labels)

        return loss_t
