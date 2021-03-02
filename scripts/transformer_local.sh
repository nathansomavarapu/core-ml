#!/bin/bash

python src/VisualClassificationRunner.py \
+optimizer=sgd_mnist_vit \
+scheduler=cosine_warmup \
+model=vit \
+loss_fn=cross_entropy \
+transforms=mnist \
+dataset=mnist \
+dataloader=basic_loader_vit \
+runner.epochs=25