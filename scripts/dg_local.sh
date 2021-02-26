#!/bin/bash

python src/DGRunner.py dataset.name=$1 dataset.target=$2 +optimizer=sgd_$1 +model=$3_$1