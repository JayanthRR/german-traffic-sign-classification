#!/bin/bash


#echo "training" 
#python main.py --n-epochs 10 --batch-size=64 --val-batch-size=16 --image-net-model squeeze --split 0.2
#python main.py --n-epochs 10 --batch-size=64 --val-batch-size=16 --image-net-model resnet18 --split 0.2
#python main.py --n-epochs 10 --batch-size=64 --val-batch-size=16 --image-net-model resnet34 --split 0.2
#python main.py --n-epochs 10 --batch-size=64 --val-batch-size=16 --image-net-model vgg --split 0.2

echo "training full model"
python main.py --n-epochs 15 --batch-size=64 --val-batch-size=16 --image-net-model all --split 0.1

echo "training as feature extractor"
python main.py --n-epochs 15 --batch-size=64 --val-batch-size=16 --image-net-model all --split 0.1 --feature-extractor

echo "testing full model"
python main.py --val-batch-size=16 --trained-model all --image-net-model all --evaluate

echo "testing feature extractor model"
python main.py --val-batch-size=16 --trained-model all --image-net-model all --evaluate --eval-model-type ff



