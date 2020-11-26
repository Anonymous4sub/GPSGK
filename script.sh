#!/usr/bin/env bash

for i in $(seq 1 15)
do
    python main.py --dataset cora 2>&1 | tee -a out/cora.out;
done

for i in $(seq 1 15)
do
    python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer.out;
done

for i in $(seq 1 15)
do
    python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/pubmed.out;
done

for i in $(seq 1 15)
do
    python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo.out;
done

for i in $(seq 1 15)
do
    python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/computers.out;
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