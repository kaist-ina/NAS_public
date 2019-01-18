#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train_nas_awdnn.py --data_name news --quality ultra --dash_lr 240 --num_epoch 300 --use_cuda --num_update_per_epoch 1000 --load_on_memory
