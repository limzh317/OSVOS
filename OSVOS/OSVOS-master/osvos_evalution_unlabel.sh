#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,2,3,5 python -u OSVOS/osvos_evalution_unlabel.py | tee OSVOS/unlabel/testlogs/0719_resnet101_lr1e-4_adam_decay30_bs8_post.log