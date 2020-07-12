#!/bin/bash
nvcc common.cu tensor.cu vector.cu matrix.cu neural.cu -O3 --shared -Xcompiler=-fPIC -o libmatrics.so
sudo cp libmatrics.so /usr/lib/x86_64-linux-gnu
