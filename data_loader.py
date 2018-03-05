
from torch.utils.data import Dataset
from transforms_helper import *
import torchvision
from PIL import Image
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch


class TrafficSignDataset(Dataset):
    """Traffic Sign Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.labels_frame = pd.read_csv(csv_file, sep=";")
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        img = self.labels_frame.iloc[idx]

        image = Image.open(os.path.join(self.root_dir, img['Filename']))
        label = img['ClassId']
        
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'class': label}
        
        return sample


def get_train_valid_loader(data_dir,
                           train_batch_size,
                           val_batch_size,
                           train_transform,
                           valid_transform,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
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

    classes = train_dataset.classes
    valid_dataset = datasets.ImageFolder(root=data_dir, transform=valid_transform)

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


def check():

    labels_frame = pd.read_csv('GTSRB/Online-Test/GT-online_test.csv', sep=";")
    img_name = labels_frame.iloc[10]
    print(img_name)
    print(img_name['Filename'])
    # print(labels_frame.iloc[10,:])
    image = io.imread(os.path.join('GTSRB/Online-Test/Images/',img_name['Filename']))
    plt.imshow(image)
    plt.show()

    root_dir = 'GTSRB/Online-Test/'
    traffic_dataset = TrafficSignDataset(csv_file=os.path.join(root_dir, 'GT-online_test.csv'), 
                                         root_dir=os.path.join(root_dir, 'Images/'),
                                         transform=test_transform)

    test_loader = torch.utils.data.DataLoader(traffic_dataset, batch_size=8)                           
    data_iter = iter(test_loader)
    sample = data_iter.next()
    images, labels = sample['image'], sample['class']
    # Make a grid from batch
    out = torchvision.utils.make_grid(images)

    imshow(out, title=[x for x in labels])

