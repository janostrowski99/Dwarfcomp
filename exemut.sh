#!/bin/bash

module load cuda/9.0
nvcc -arch sm_50 -O3 mutation.cu -o mutation -lcuda -lcufft

./mutation 5 1 1
./mutation 50 1 1
./mutation 500 1 1
./mutation 5000 1 1
./mutation 50000 1 1

./mutation 5 1 1
./mutation 5 10 1
./mutation 5 100 1
./mutation 5 1000 1
./mutation 5 10000 1

./mutation 5 1 1
./mutation 5 1 3
./mutation 5 1 5
./mutation 5 1 7
./mutation 5 1 9
