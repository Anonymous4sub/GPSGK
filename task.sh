#!/usr/bin/env bash

python main.py --dataset cora --lambda2 1e-5 2>&1 | tee -a out/cora_m5.out;
python main.py --dataset cora --lambda2 1e-5 2>&1 | tee -a out/cora_m5.out;
python main.py --dataset cora --lambda2 1e-5 2>&1 | tee -a out/cora_m5.out;
python main.py --dataset cora --lambda2 1e-5 2>&1 | tee -a out/cora_m5.out;
python main.py --dataset cora --lambda2 1e-5 2>&1 | tee -a out/cora_m5.out;

python main.py --dataset cora --lambda2 1e-3 2>&1 | tee -a out/cora_m3.out;
python main.py --dataset cora --lambda2 1e-3 2>&1 | tee -a out/cora_m3.out;
python main.py --dataset cora --lambda2 1e-3 2>&1 | tee -a out/cora_m3.out;
python main.py --dataset cora --lambda2 1e-3 2>&1 | tee -a out/cora_m3.out;
python main.py --dataset cora --lambda2 1e-3 2>&1 | tee -a out/cora_m3.out;

python main.py --dataset cora --lambda2 1e-2 2>&1 | tee -a out/cora_m2.out;
python main.py --dataset cora --lambda2 1e-2 2>&1 | tee -a out/cora_m2.out;
python main.py --dataset cora --lambda2 1e-2 2>&1 | tee -a out/cora_m2.out;
python main.py --dataset cora --lambda2 1e-2 2>&1 | tee -a out/cora_m2.out;
python main.py --dataset cora --lambda2 1e-2 2>&1 | tee -a out/cora_m2.out;

python main.py --dataset cora --lambda2 1e-1 2>&1 | tee -a out/cora_m1.out;
python main.py --dataset cora --lambda2 1e-1 2>&1 | tee -a out/cora_m1.out;
python main.py --dataset cora --lambda2 1e-1 2>&1 | tee -a out/cora_m1.out;
python main.py --dataset cora --lambda2 1e-1 2>&1 | tee -a out/cora_m1.out;
python main.py --dataset cora --lambda2 1e-1 2>&1 | tee -a out/cora_m1.out;

python main.py --dataset cora --lambda2 1.0 2>&1 | tee -a out/cora_1.out;
python main.py --dataset cora --lambda2 1.0 2>&1 | tee -a out/cora_1.out;
python main.py --dataset cora --lambda2 1.0 2>&1 | tee -a out/cora_1.out;
python main.py --dataset cora --lambda2 1.0 2>&1 | tee -a out/cora_1.out;
python main.py --dataset cora --lambda2 1.0 2>&1 | tee -a out/cora_1.out;

python main.py --dataset cora --lambda2 1.2 2>&1 | tee -a out/cora_1p2.out;
python main.py --dataset cora --lambda2 1.2 2>&1 | tee -a out/cora_1p2.out;
python main.py --dataset cora --lambda2 1.2 2>&1 | tee -a out/cora_1p2.out;
python main.py --dataset cora --lambda2 1.2 2>&1 | tee -a out/cora_1p2.out;
python main.py --dataset cora --lambda2 1.2 2>&1 | tee -a out/cora_1p2.out;

python main.py --dataset cora --lambda2 1.5 2>&1 | tee -a out/cora_1p5.out;
python main.py --dataset cora --lambda2 1.5 2>&1 | tee -a out/cora_1p5.out;
python main.py --dataset cora --lambda2 1.5 2>&1 | tee -a out/cora_1p5.out;
python main.py --dataset cora --lambda2 1.5 2>&1 | tee -a out/cora_1p5.out;
python main.py --dataset cora --lambda2 1.5 2>&1 | tee -a out/cora_1p5.out;

python main.py --dataset cora --lambda2 2.0 2>&1 | tee -a out/cora_2.out;
python main.py --dataset cora --lambda2 2.0 2>&1 | tee -a out/cora_2.out;
python main.py --dataset cora --lambda2 2.0 2>&1 | tee -a out/cora_2.out;
python main.py --dataset cora --lambda2 2.0 2>&1 | tee -a out/cora_2.out;
python main.py --dataset cora --lambda2 2.0 2>&1 | tee -a out/cora_2.out;


python main.py --dataset citeseer --n_samples 200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_200samples.out;
python main.py --dataset citeseer --n_samples 200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_200samples.out;
python main.py --dataset citeseer --n_samples 200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_200samples.out;
python main.py --dataset citeseer --n_samples 200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_200samples.out;
python main.py --dataset citeseer --n_samples 200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_200samples.out;

python main.py --dataset citeseer --n_samples 400 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_400samples.out;
python main.py --dataset citeseer --n_samples 400 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_400samples.out;
python main.py --dataset citeseer --n_samples 400 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_400samples.out;
python main.py --dataset citeseer --n_samples 400 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_400samples.out;
python main.py --dataset citeseer --n_samples 400 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_400samples.out;

python main.py --dataset citeseer --n_samples 600 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_600samples.out;
python main.py --dataset citeseer --n_samples 600 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_600samples.out;
python main.py --dataset citeseer --n_samples 600 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_600samples.out;
python main.py --dataset citeseer --n_samples 600 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_600samples.out;
python main.py --dataset citeseer --n_samples 600 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_600samples.out;

python main.py --dataset citeseer --n_samples 800 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_800samples.out;
python main.py --dataset citeseer --n_samples 800 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_800samples.out;
python main.py --dataset citeseer --n_samples 800 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_800samples.out;
python main.py --dataset citeseer --n_samples 800 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_800samples.out;
python main.py --dataset citeseer --n_samples 800 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_800samples.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1000samples.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1000samples.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1000samples.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1000samples.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1000samples.out;

python main.py --dataset citeseer --n_samples 1200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1200samples.out;
python main.py --dataset citeseer --n_samples 1200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1200samples.out;
python main.py --dataset citeseer --n_samples 1200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1200samples.out;
python main.py --dataset citeseer --n_samples 1200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1200samples.out;
python main.py --dataset citeseer --n_samples 1200 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1200samples.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 0.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_withoutstructure.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 0.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_withoutstructure.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 0.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_withoutstructure.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 0.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_withoutstructure.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 0.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_withoutstructure.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m5.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-3 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m3.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-3 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m3.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-3 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m3.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-3 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m3.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-3 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m3.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m2.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1e-1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_m1.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1.2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.2 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p2.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 1.5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p5.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 1.5 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_1p5.out;

python main.py --dataset citeseer --n_samples 1000 --lambda2 2.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 2.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 2.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 2.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_2.out;
python main.py --dataset citeseer --n_samples 1000 --lambda2 2.0 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer/citeseer_2.out;
