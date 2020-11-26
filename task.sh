#!/usr/bin/env bash

for t in {0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}
do 
    python main.py --dataset cora --tau $t 2>&1 | tee -a out/cora_tau.out;
done
