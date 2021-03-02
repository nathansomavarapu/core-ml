#!/bin/bash

python src/VisualClassificationRunner.py \
+optimizer=sgd_cifar10_vit \
+scheduler=cosine_warmup \
+model=vit \
+loss_fn=cross_entropy \
+transforms=cifar \
+dataset=cifar10 \
+dataloader=basic_loader_vit \
+runner.epochs=1000