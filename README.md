# Traffic Sign Classification

This project uses the [German Traffic Sign Recognition Benchmarks Dataset(GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This repository is aimed at exploring the utilization of [torchvision.models](http://pytorch.org/docs/master/torchvision/models.html) for transfer learning.

I included a data loader for the GTSRB data set using Pytorch's [data loading tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html). The directory structure is assumed to look as
- GTSRB/
  - Final_Training/
    - Images/
      - 00000/
        - 00000.pm
        - 00001.pm

          ...

        - GT-00000.csv

      - 00001/

        ...

  - Final_Test/
    - Images/
      - 00000.pm

        ...

      - GT-final_test.csv

The folder names `00000/`, `00001/` correspond to the class labels, and there are 43 such labels. 
