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
python link.py --dataset pubmed --pretrain_step 500 --label_ratio 0.2
