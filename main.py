
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from data_loader import TrafficSignDataset, get_train_valid_loader
from transforms_helper import *
from pretrained_models import load_model
from save_object import save


parser = argparse.ArgumentParser(description='Pytorch Traffic Sign Classification Example')
parser.add_argument('--batch-size',type=int, default=64, help='training batch size')
parser.add_argument('--val-batch-size', type=int, default=16, help='val batch size')
parser.add_argument('--n-epochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr', type=int, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='batches for logging status')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum. Default=0.9')
parser.add_argument('--trained-model', type=str, default='resnet', help='specify a trained model')
parser.add_argument('--eval', action='store_true', help='specify if the model has to be tested')
parser.add_argument('--no-eval', action='store_false', help='specify if the model has to be tested')
parser.add_argument('--train', action='store_true', help='specify if the model has to be trained')
parser.add_argument('--no-train', action='store_false', help='speficy if the model has to be trained')
parser.add_argument('--feature-extractor', action='store_false', help='use resnet as a feature extractor?')
parser.add_argument('--no-feature-extractor', action='store_true', help='train resnet instead of feature extractor')
parser.add_argument('--image-net-model', type=str, default='resnet', help='which image net pretrained model do you want to use?')

opt = parser.parse_args()

print(opt)

torch.manual_seed(opt.seed)


def test_model(model, test_loader, criterion, auxloss, valid_size=None):

    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    dataloader = test_loader

    for batch_idx, data in tqdm(enumerate(dataloader)):
        if isinstance(data, dict):
            inputs = data['image']
            labels = data['class']
        else:
            inputs = data[0]
            labels = data[1]

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)

        if auxloss:
            softmax_out = outputs[0]
            aux_out = outputs[1]

            _, soft_preds = torch.max(softmax_out.data, 1)
            _, aux_preds = torch.max(aux_out.data, 1)
            soft_loss = criterion(softmax_out, labels)
            aux_loss = criterion(aux_out, labels)
            loss = soft_loss + 0.5 * aux_loss
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(soft_preds == labels.data)

        else:
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    print(' Loss: {:.6f}, Accuracy: {:.6f}'.format(running_loss/len(dataloader.dataset), 100*running_corrects/len(dataloader.dataset)))

    if valid_size is not None:
        return running_loss/valid_size, 100*running_corrects/valid_size
    else:
        return running_loss/len(dataloader.dataset), 100*running_corrects/len(dataloader.dataset)


def train_model(model, train_loader, valid_loader, train_size, valid_size,
                criterion, optimizer, scheduler, num_epochs=opt.n_epochs, auxloss=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = 0

    result = dict()

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0
                dataloader = train_loader
                
                for batch_idx, (inputs, labels) in enumerate(dataloader):
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)

                    if auxloss:
                        softmax_out = outputs[0]
                        aux_out = outputs[1]

                        _, soft_preds = torch.max(softmax_out.data, 1)
                        _, aux_preds = torch.max(aux_out.data, 1)
                        soft_loss = criterion(softmax_out, labels)
                        aux_loss = criterion(aux_out, labels)
                        loss = soft_loss + 0.5* aux_loss
                        running_loss += loss.data[0] * inputs.size(0)
                        running_corrects += torch.sum(soft_preds == labels.data)
                    
                    else:
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        running_loss += loss.data[0] * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                    # statistics

                    if batch_idx % opt.log_interval == 0:
                        print('Train epoch: {} [{}/{}]\t\tLoss: {:.6f}'.format(
                            epoch, batch_idx, len(train_loader), 
                            loss.data[0]))

                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects / train_size

            else:

                epoch_loss, epoch_acc = test_model(model, valid_loader, criterion, auxloss=auxloss, valid_size=valid_size)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            print('{} Loss: {:.6f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc*100))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    result['time_taken'] = time_elapsed
    result['best_val_loss'] = best_val_loss
    result['best_val_acc'] = best_acc
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, result 


if __name__=="__main__":


    result_logs = dict()
    # pretrained = ['squeeze', 'resnet18', 'inception', 'vgg', 'resnet34']
    pretrained = []
    pretrained.append(opt.image_net_model)
    use_gpu = torch.cuda.is_available()
    auxloss = False

    train_data_dir = '/home/cc/Datasets/traffic_sign/GTSRB/Final_Training/Images/'
    test_data_dir = '/home/cc/Datasets/traffic_sign/GTSRB/Final_Test/Images/'

    # if opt.eval:
    #
    #     test_data_set = TrafficSignDataset(csv_file= os.path.join(test_data_dir,'GT-final_test.csv'),
    #                                        root_dir= test_data_dir,
    #                                        transform=test_transform)
    #     test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = opt.val_batch_size)
    #
    #     if not opt.trained_model == 'all':
    #         model = torch.load(opt.trained_model + '_.pkl')
    #         result_logs[opt.trained_model] = dict()
    #     else:
    #         for model_name in pretrained:
    #             result_logs[model_name] = dict()
    #
    #             print("pretrain: {}".format(model_name))
    #             model = torch.load(model_name + '_.pkl')
    #
    #             if use_gpu:
    #                 model = model.cuda()
    #
    #             criterion = nn.CrossEntropyLoss()
    #
    #     test_data_set = TrafficSignDataset(csv_file= os.path.join(test_data_dir,'GT-final_test.csv'),
    #                                        root_dir= test_data_dir,
    #                                        transform=test_transform)
    #     test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = opt.val_batch_size)
    #
    #     result = test_model(model, test_loader, criterion, auxloss)

    if opt.train:
        for model_name in pretrained:
            result_logs[model_name] = dict()

            print("pretrain: {}".format(model_name))
            if not opt.feature_extractor:

                model, train_transform, valid_transform, test_transform, auxloss = load_model(model_name,
                                                                                              feature_extractor=False)
                if use_gpu:
                    model = model.cuda()

                criterion = nn.CrossEntropyLoss()
                # Observe that all parameters are being optimized
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=opt.momentum)

                # Decay LR by a factor of 0.1 every 7 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            else:
                # fixed feature extractor
                model, train_transform, valid_transform, test_transform, auxloss = load_model(model_name,
                                                                                              feature_extractor=True)

                if use_gpu:
                    model = model.cuda()

                criterion = nn.CrossEntropyLoss()

                # Observe that only parameters of final layer are being optimized as
                # opoosed to before.
                optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=opt.momentum)

                # Decay LR by a factor of 0.1 every 7 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            train_loader, valid_loader, train_size, valid_size,\
            classes = get_train_valid_loader(train_data_dir, train_batch_size=opt.batch_size,
                                             val_batch_size=opt.val_batch_size,
                                             train_transform=train_transform,
                                             valid_transform=valid_transform,
                                             random_seed=opt.seed,
                                             valid_size=0.1,
                                             shuffle=True,
                                             num_workers=opt.threads,
                                             pin_memory=True)

            data_iter = iter(train_loader)
            images, labels = data_iter.next()
            print('no of classes : {}'.format(len(classes)))

            model, result_logs[model_name] = train_model(model, train_loader, valid_loader, train_size,
                                                         valid_size, criterion, optimizer, exp_lr_scheduler,
                                                         num_epochs=opt.n_epochs, auxloss=auxloss)
            
            torch.save(model, model_name + '_.pt')

            print("completed_training")

            test_data_set = TrafficSignDataset(csv_file= os.path.join(test_data_dir,'GT-final_test.csv'),
                                               root_dir= test_data_dir,
                                               transform=test_transform)
            test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = opt.val_batch_size)

            result = test_model(model, test_loader, opt.val_batch_size, criterion)
            result_logs[model_name]['test_accuracy'] = result[1]
            result_logs[model_name]['test_loss'] = result[0]

            print(result_logs[model_name])
    
    save(result_logs, 'logs_all.pkl')
    
    print("completed_testing")

