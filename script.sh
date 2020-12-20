#!/usr/bin/env bash

for i in $(seq 1 15)
do
    python main.py --dataset cora 2>&1 | tee -a out/cora.out; 
    python main.py --dataset cora --seed 24 --lambda1 3 --tau 0.7; # 0.839
    python main.py --dataset cora --seed 24 --lambda1 3 --tau 0.8; # 0.843

done

for i in $(seq 1 15)
do
    python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer.out;
    python main.py --dataset citeseer --seed 0 --lambda1 2 --tau 0.5; # 0.7287
    python main.py --dataset citeseer --seed 0 --lambda1 2 --tau 0.8; # 0.729
    
done

for i in $(seq 1 15)
do
    python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/pubmed.out;
    python main.py --dataset pubmed --seed 512 --lambda1 1 --tau 0.9; # 0.801
    python main.py --dataset pubmed --seed 512 --lambda1 5 --tau 0.9; # 0.806
done

for i in $(seq 1 15)
do
    python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo.out; 
    python main.py --dataset photo --seed 24 --pretrain_step 500 --lr 0.0005 --lambda1 10 --tau 0.7;  # 0.927
    python main.py --dataset photo --seed 24 --pretrain_step 500 --lambda1 4 --tau 0.7; # 0.929
done

for i in $(seq 1 15)
do
    python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/computers.out;
    python main.py --dataset computers --seed 24 --pretrain_step 500 --lr 0.0005 --lambda1 10 --tau 0.7; # 0.85509
    python main.py --dataset computers --seed 24 --pretrain_step 500 --lambda1 3 --tau 0.9; #0.876
done

# for link prediction
python link.py --dataset cora
python link.py --dataset citeseer --batch_size 256
python link.py --dataset photo --pretrain_step 500
python link.py --dataset computers --pretrain_step 500
python link.py --dataset pubmed --batch_size 2000 --steps 2000 --lambda2 10.0


for ((s=100; s<=2000; s+=50));
do 
    for i in {1, 2, 3}
    do 
        python main.py --dataset photo --n_samples $s --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_samples.out;
    done
done

for ((s=10; s<=2000; s+=10));
do 
    python main.py --dataset cora --n_samples $s 2>&1 | tee -a out/cora_samples.out;
   
done