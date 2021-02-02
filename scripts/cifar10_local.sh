#!/bin/bash

srun -p short -c 6 --account=overcap --gres=gpu:1 --job-name="test" -x calculon,neo,johnny5,c3po,rosie,hal,walle,bender python src/VisualClassificationRunner.py