#!/bin/bash
rm costs/bank_cost.txt
cd ~/desktop/programming/c++/neural_net/neural_net/source
g++ -std=c++2a -c neuron.h layer.cpp layer.h dataset_utils.h dataset_utils.cpp neural_network.h neural_network.cpp
g++ -std=c++2a -c main/bank_main.cpp
#python ~/desktop/programming/c++/neural_net/neural_net/scripts/prep_bank_data.py
#sed '$d' train_bank.txt > train_bank.txt 
#sed '$d' test_bank.txt > test_bank.txt 
g++ -o run_bank bank_main.o layer.o dataset_utils.o neural_network.o
./run_bank
