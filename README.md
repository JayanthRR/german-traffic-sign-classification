# Traffic Sign Classification

This project uses the [German Traffic Sign Recognition Benchmarks Dataset(GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This repository is aimed at exploring the utilization of various [torchvision.models](http://pytorch.org/docs/master/torchvision/models.html) for transfer learning.


### Data Loader and Directory Structure

I included a data loader for the GTSRB data set with the help of Pytorch's [data loading tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
I wrote two different data loaders. `get_train_valid_loader` for the training and validation sets (courtesy [kevinzakka](https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb)) and `TrafficSignDataset` for the test data set. `get_train_valid_loader` uses `torchvision.datasets.ImageFolder` that leverages the following directory structure) and makes our life extremely easy to load data. The test data loader requires a `.csv` file with annotations, and is explained in the ipython notebook [here](http://localhost:8888/notebooks/test_data_loader.ipynb)


The directory structure is assumed to look as
- GTSRB/
  - Final_Training/
    - Images/
      - 00000/
      - 00001/

        ...

  - Final_Test/
    - Images/
      - 00000.pm

        ...

      - GT-final_test.csv

The folder names `00000/`, `00001/` correspond to the class labels, and there are 43 such labels.

### Fine tuning `torchvision.models`
- `Resnet`
   takes `224x224` images as input, so we need to apply an appropriate transform (`torchvision.transforms.Resize, torchvision.transforms.RandomResizedCrop`) while calling the data loader. To tune the model for a custom number of classes, we modify the final fully connected layer with the number of classes (here 43).
  ```
  model = models.resnet34(pretrained=True)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 43)

  ```


- `Inceptionv3`
   takes in `299x299` images, so we need to apply an appropriate transform while calling the data loader. Inceptionv3 enables *Auxiliary Classifiers* while training. To modify the model for a custom number of classes (in the output layer)

  ```
  def get_pretrained_inception(num_classes, pretrained=True):
    inception = torchvision.models.inception_v3(pretrained=pretrained)

    fc_in_features = inception.fc.in_features
    inception.fc = nn.Linear(in_features=fc_in_features, out_features=num_classes)
    inception.AuxLogits = InceptionAux(in_channels=768, num_classes=num_classes)

    return inception

  ```
  `inception(inputs)` returns a tuple corresponding to the outputs of the main classifier and the auxiliary classifier. If required, the losses corresponding to both the classifiers can be weighted and passed onto the optimizer. Although, only the predictions of the main classifier are used during test time.

- `Squeezenet`
  takes `255x255` images as input. It is mentioned in some threads in Pytorch forums that it can adaptively take different input shapes, but I haven't tried it so far.
  Training `squeezenet` is extremely fast.  As compared to `inception`, modifying squeezenet is relatively straightforward. We need to modify the `conv2d` layer in the `Classifier`

  ```
  model = models.squeezenet1_1(pretrained=True)
  num_ftrs = model.classifier._modules['1'].in_channels

  nn.Linear(num_ftrs, 42)
  model.classifier._modules['1'] = nn.Conv2d(num_ftrs, 43, 3)
  model.num_classes = 43
  ```
  We just modify the `Conv2d` layer with the required number of classes.

- `VGG`
  We only need to modify the `Classifier` with our required number of classes.

  ```
  model = models.vgg11(pretrained=True)
  num_ftrs = model.classifier._modules['6'].in_features
  model.classifier._modules['6'] = nn.Linear(num_ftrs, 43)

    ```

### Image Augmentation
- Todo

### Hyperparameter optimization
- Todo
