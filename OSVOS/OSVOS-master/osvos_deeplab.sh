#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u OSVOS/osvos_deeplab.py | tee OSVOS/traininglogs/0723_deeplab_lr1e-4_Adam_decay10_12_weight.log