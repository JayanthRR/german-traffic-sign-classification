
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
from test_data_loader import TrafficSignDataset
from transforms_helper import vgg_train_transform, inception_train_transform, resnet_train_transform, squeeze_train_transform
from transforms_helper import vgg_test_transform, inception_test_transform, resnet_test_transform, squeeze_test_transform
from modify_inceptionv3 import get_pretrained_inception
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

def get_train_valid_loader(data_dir,
                           train_batch_size,
                           val_batch_size,
                           train_transform,
                           valid_transform,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transforms
#    valid_transform = valid_transform
#    train_transform = train_transform

#    if augment:
#        train_transform = transforms.Compose([
#        transforms.RandomSizedCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])
#    else:
#        train_transform = transforms.Compose([
#        transforms.Resize((224,224)),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])

    # load the dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
#     train_dataset = datasets.CIFAR10(
#         root=data_dir, train=True,
#         download=True, transform=train_transform,
#     )
    classes = train_dataset.classes
    valid_dataset = datasets.ImageFolder(root=data_dir, transform=valid_transform)
#     valid_dataset = datasets.CIFAR10(
#         root=data_dir, train=True,
#         download=True, transform=valid_transform,
#     )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_data_size = len(train_sampler)
    valid_data_size = len(valid_sampler)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=val_batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, valid_loader, train_data_size, valid_data_size, classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def test_model(model, test_loader, test_size, criterion):

    dataloader = test_loader
    model.eval()
    running_loss = 0
    running_corrects = 0
    for batch_idx, data in tqdm(enumerate(dataloader)):
        # wrap them in Variable
        inputs = data['image']
        labels = data['class']
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.data[0] * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    print(' Loss: {:.6f}, Accuracy: {:.6f}'.format(running_loss/len(dataloader.dataset), 100*running_corrects/len(dataloader.dataset)))

    return running_loss/len(dataloader.dataset), 100*running_corrects/len(dataloader.dataset)


def train_model(model, train_loader, valid_loader, train_size, valid_size,
                criterion, optimizer, scheduler, num_epochs=opt.n_epochs, auxloss=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    #result = dict.fromkeys(['best_val_loss','best_val_acc', 'time_taken', 'test_loss', 'accuracy'],0)
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

                    # print(inputs.shape)
                    # forward
                    outputs = model(inputs)
                    #print(outputs)
                    #print(outputs.data.shape)
                    if auxloss:
                        softmax_out = outputs[0]
                        aux_out = outputs[1]
                        # print('shape of out',softmax_out.shape)
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
                model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                dataloader = valid_loader
                  
                for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader)):
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                epoch_loss = running_loss / valid_size
                epoch_acc = running_corrects / valid_size

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

    if not opt.train:

        criterion = nn.CrossEntropyLoss()
        test_data_set = TrafficSignDataset(csv_file= os.path.join(test_data_dir,'GT-final_test.csv'),
                                           root_dir= test_data_dir,
                                           transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = opt.val_batch_size)
        model = torch.load(opt.trained_model)
        # model = torch.nn.DataParallel(model)

        result = test_model(model, test_loader, opt.val_batch_size, criterion)
    else:
        for pretrain in pretrained:
            result_logs[pretrain] = dict()
            # result_logs[pretrain] = dict.fromkeys(['best_val_loss','best_val_acc', 'time_taken', 'test_loss', 'accuracy'],0)
            print("pretrain: {}".format(pretrain))
            if not opt.feature:
                # training resnet18
                if pretrain == 'inception':

                    model = get_pretrained_inception(num_classes=43, pretrained=True)
                    # model = models.inception_v3(pretrained=True)
                    # num_ftrs = model.fc.in_features
                    # model.fc = nn.Linear(num_ftrs, 42)
                    # model.fc = nn.Linear(num_ftrs, 43)
                    auxloss = True
                    train_transform = inception_train_transform
                    val_transform = inception_test_transform
                    test_transform = inception_test_transform

                elif pretrain == 'squeeze':
                    model = models.squeezenet1_1(pretrained=True)
                    num_ftrs = model.classifier._modules['1'].in_channels
                    # model.classifier._modules['1'] = nn.Linear(num_ftrs, 42)
                    model.classifier._modules['1'] = nn.Conv2d(num_ftrs, 43, 3)
                    model.num_classes = 43
                    train_transform = squeeze_train_transform
                    val_transform = squeeze_test_transform
                    test_transform = squeeze_test_transform

                elif pretrain == 'dense':
                    model = models.densenet121(pretrained=True)
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, 43)
                
                    train_transform = dense_train_transform
                    val_transform = dense_test_transform
                    test_transform = dense_test_transform
                
                elif pretrain == 'vgg':
                    model = models.vgg11(pretrained=True)
                    num_ftrs = model.classifier._modules['6'].in_features
                    model.classifier._modules['6'] = nn.Linear(num_ftrs, 43)

                    train_transform = resnet_train_transform
                    val_transform = resnet_test_transform
                    test_transform = resnet_test_transform

                elif pretrain == 'resnet34':
                    model = models.resnet34(pretrained=True)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 43)

                    train_transform = resnet_train_transform
                    val_transform = resnet_test_transform
                    test_transform = resnet_test_transform

                else:
                    model = models.resnet18(pretrained=True)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 43)

                    train_transform = resnet_train_transform
                    val_transform = resnet_test_transform
                    test_transform = resnet_test_transform

                if use_gpu:
                    model = model.cuda()


                criterion = nn.CrossEntropyLoss()
                # Observe that all parameters are being optimized
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=opt.momentum)

                # Decay LR by a factor of 0.1 every 7 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            else:
                # fixed feature extractor
                model = torchvision.models.resnet18(pretrained=True)
                for param in model.parameters():
                    param.requires_grad = False

                train_transform = resnet_train_transform
                val_transform = resnet_test_transform
                test_transform = resnet_test_transform
                
                # Parameters of newly constructed modules have requires_grad=True by default
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 43)

                if use_gpu:
                    model = model.cuda()

                criterion = nn.CrossEntropyLoss()

                # Observe that only parameters of final layer are being optimized as
                # opoosed to before.
                optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=opt.momentum)

                # Decay LR by a factor of 0.1 every 7 epochs
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


            train_loader, valid_loader, train_size, valid_size, classes = get_train_valid_loader(train_data_dir, 
                                                                train_batch_size=opt.batch_size,
                                                                val_batch_size=opt.val_batch_size,
                                                                train_transform=train_transform,
                                                                valid_transform=val_transform,
                                                                random_seed=opt.seed,
                                                                valid_size=0.1,
                                                                shuffle=True,
                                                                show_sample=False,
                                                                num_workers=opt.threads,
                                                                pin_memory=True)

            data_iter = iter(train_loader)
            images, labels = data_iter.next()
            print('no of classes : {}'.format(len(classes)))



            model, result_logs[pretrain] = train_model(model, train_loader, valid_loader, train_size,
                    valid_size, criterion, optimizer, exp_lr_scheduler, num_epochs=opt.n_epochs, auxloss=auxloss)
            
            torch.save(model, pretrain +'_.pt')

            print("completed_training")

            test_data_set = TrafficSignDataset(csv_file= os.path.join(test_data_dir,'GT-final_test.csv'),
                                               root_dir= test_data_dir,
                                               transform=test_transform)
            test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = opt.val_batch_size)

            result = test_model(model, test_loader, opt.val_batch_size, criterion)
            result_logs[pretrain]['test_accuracy'] = result[1]
            result_logs[pretrain]['test_loss'] = result[0]

            print(result_logs[pretrain])
    
    save(result_logs, 'logs_all.pkl')
    
    print("completed_testing")

