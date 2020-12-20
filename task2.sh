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

for l in 0.5 0.8 1.0 2.0 3.0 4.0 5.0 8.0 10.0 15.0 20.0
do 
    for t in 0.0 0.3 0.5 0.6 0.7 0.8 0.9 1.1
    do 
        python main.py --dataset cora --seed 24 --lambda1 $l --tau $t 2>&1 | tee -a out/cora_tau_lambda1.out;
    done
done 

for l in 0.5 0.8 1.0 2.0 3.0 4.0 5.0 8.0 10.0 15.0 20.0
do 
    for t in 0.0 0.3 0.5 0.6 0.7 0.8 0.9 1.1
    do 
        python main.py --dataset citeseer --seed 0 --lambda1 $l --tau $t 2>&1 | tee -a out/citeseer_tau_lambda1.out;
    done
done 

for l in 0.5 0.8 1.0 2.0 3.0 4.0 5.0 8.0 10.0 15.0 20.0
do 
    for t in 0.0 0.3 0.5 0.6 0.7 0.8 0.9 1.1
    do 
        python main.py --dataset pubmed --seed 512 --lambda1 $l --tau $t 2>&1 | tee -a out/pubmed_tau_lambda1.out;
    done
done 

for l in 0.5 0.8 1.0 2.0 3.0 4.0 5.0 8.0 10.0 15.0 20.0
do 
    for t in 0.0 0.3 0.5 0.6 0.7 0.8 0.9 1.1
    do 
        python main.py --dataset photo --seed 24 --pretrain_step 500 --lambda1 $l --tau $t 2>&1 | tee -a out/photo_tau_lambda1.out;
    done
done 

for l in 0.5 0.8 1.0 2.0 3.0 4.0 5.0 8.0 10.0 15.0 20.0
do 
    for t in 0.0 0.3 0.5 0.6 0.7 0.8 0.9 1.1
    do 
        python main.py --dataset computers --seed 24 --pretrain_step 500 --lambda1 $l --tau $t 2>&1 | tee -a out/computers_tau_lambda1.out;
    done
done 