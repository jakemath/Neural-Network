#!/bin/bash
rm ../costs/max_index_cost.txt
python3 prep_bank_data.py
cd ../source
g++ -std=c++2a -c header/neuron.h module/layer.cpp header/layer.h header/dataset_utils.h module/dataset_utils.cpp header/neural_network.h module/neural_network.cpp
g++ -std=c++2a -c main/bank_main.cpp
g++ -o run_bank bank_main.o layer.o dataset_utils.o neural_network.o
./run_bank
rm *.o header/*.o header/*.gch*
