#!/usr/bin/env bash

for n in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000
do 
    python main.py --dataset cora --seed 24 --lambda1 3 --tau 0.8 --n_samples $n 2>&1 | tee -a out/cora_samples_tau.out;
done 

for n in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000
do 
    python main.py --dataset cora --seed 24 --lambda1 3 --tau 0.0 --n_samples $n 2>&1 | tee -a out/cora_samples_withoutau.out;
done

for n in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000
do 
    python main.py --dataset photo --seed 24 --pretrain_step 500 --lambda1 4 --tau 0.7 --n_samples $n 2>&1 | tee -a out/photo_samples_tau.out;
done 

for n in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000
do 
    python main.py --dataset photo --seed 24 --pretrain_step 500 --lambda1 4 --tau 0.0 --n_samples $n 2>&1 | tee -a out/photo_samples_withoutau.out;
done