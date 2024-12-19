#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ../configs/data_generation.yaml ./generate.py

