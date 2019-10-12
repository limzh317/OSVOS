#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python -u OSVOS/osvos_psp.py | tee OSVOS/traininglogs/0723_psp_lr1e-3_Adam_decay10_03.log