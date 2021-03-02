#!/bin/bash

python src/VisualClassificationRunner.py \
+optimizer=sgd_cifar10 \
+scheduler=cosine \
+model=resnet18 \
+loss_fn=cross_entropy \
+transforms=cifar \
+dataset=cifar10 \
+dataloader=basic_loader \
+runner.epochs=200