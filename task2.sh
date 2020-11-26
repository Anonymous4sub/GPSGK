#!/usr/bin/env bash

python main.py --dataset photo --n_samples 8 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 8 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 8 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;

python main.py --dataset photo --n_samples 25 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 25 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 25 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;

python main.py --dataset photo --n_samples 50 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 50 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 50 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;

python main.py --dataset photo --n_samples 75 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 75 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
python main.py --dataset photo --n_samples 75 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;

for ((s=100; s<=2000; s+=50));
do 
    for i in {1, 2, 3}
    do 
        python main.py --dataset photo --n_samples $s --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
    done
done