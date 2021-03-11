#!/bin/bash

for lr in 0.001 0.005
do
    for wd in 0.0 0.0005
    do
        for sp in 0.1 0.3 0.5
        do
            for dom in clipart infograph painting quickdraw real sketch
            do
                srun \
                -p short \
                -c 8 \
                --account=overcap \
                -p overcap \
                --gres=gpu:1 \
                --job-name="dnet" \
                -x calculon,neo,johnny5,c3po,rosie,hal,walle,bender \
                python \
                src/DGRunner.py dataset._name=domainnet \
                dataset.target=$dom \
                +optimizer=sgd_domainnet \
                optimizer.lr="$lr" \
                optimizer.weight_decay="$wd" \
                +model=resnet18_domainnet \
                +loader.p=$sp &
            done
            wait
        done
    done
done