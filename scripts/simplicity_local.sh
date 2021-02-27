#!/bin/bash

python src/SimplicityRunner.py dataset=mnist_cifar10 runner.exp_name='simplicity' model.num_classes=2 transforms=to_tensor