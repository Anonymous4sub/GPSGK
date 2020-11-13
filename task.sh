#!/usr/bin/env bash

python main.py --dataset cora --tau 0.1 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.1 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.1 2>&1 | tee -a out/cora_tau.out;

python main.py --dataset cora --tau 0.3 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.3 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.3 2>&1 | tee -a out/cora_tau.out;

python main.py --dataset cora --tau 0.5 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.5 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.5 2>&1 | tee -a out/cora_tau.out;

python main.py --dataset cora --tau 0.6 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.6 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.6 2>&1 | tee -a out/cora_tau.out;

python main.py --dataset cora --tau 0.8 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.8 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.8 2>&1 | tee -a out/cora_tau.out;

python main.py --dataset cora --tau 0.9 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.9 2>&1 | tee -a out/cora_tau.out;
python main.py --dataset cora --tau 0.9 2>&1 | tee -a out/cora_tau.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.1 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.1 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.1 2>&1 | tee -a out/citeseer_tau.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.3 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.3 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.3 2>&1 | tee -a out/citeseer_tau.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.5 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.5 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.5 2>&1 | tee -a out/citeseer_tau.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/citeseer_tau.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.8 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.8 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.8 2>&1 | tee -a out/citeseer_tau.out;

python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/citeseer_tau.out;
python main.py --dataset citeseer --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/citeseer_tau.out;

python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.1 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.1 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.1 2>&1 | tee -a out/pubmed_tau.out;

python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.3 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.3 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.3 2>&1 | tee -a out/pubmed_tau.out;

python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.5 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.5 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.5 2>&1 | tee -a out/pubmed_tau.out;

python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.6 2>&1 | tee -a out/pubmed_tau.out;

python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.8 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.8 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.8 2>&1 | tee -a out/pubmed_tau.out;

python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/pubmed_tau.out;
python main.py --dataset pubmed --n_samples 1000 --pretrain_step 100 --lr 0.0005 --tau 0.9 2>&1 | tee -a out/pubmed_tau.out;

python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.1 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.1 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.1 2>&1 | tee -a out/photo_tau.out;

python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.3 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.3 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.3 2>&1 | tee -a out/photo_tau.out;

python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/photo_tau.out;

python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.6 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.6 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.6 2>&1 | tee -a out/photo_tau.out;

python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.8 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.8 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.8 2>&1 | tee -a out/photo_tau.out;

python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.9 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.9 2>&1 | tee -a out/photo_tau.out;
python main.py --dataset photo --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.9 2>&1 | tee -a out/photo_tau.out;


python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.1 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.1 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.1 2>&1 | tee -a out/computers_tau.out;

python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.3 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.3 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.3 2>&1 | tee -a out/computers_tau.out;

python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.5 2>&1 | tee -a out/computers_tau.out;

python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.6 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.6 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.6 2>&1 | tee -a out/computers_tau.out;

python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.8 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.8 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.8 2>&1 | tee -a out/computers_tau.out;

python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.9 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.9 2>&1 | tee -a out/computers_tau.out;
python main.py --dataset computers --n_samples 1000 --pretrain_step 500 --lr 0.001 --tau 0.9 2>&1 | tee -a out/computers_tau.out;

