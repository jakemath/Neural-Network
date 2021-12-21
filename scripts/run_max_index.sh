#!/bin/bash
rm ../costs/max_index_cost.txt
cd ../src
g++ -std=c++20 -O3 -c header/neuron.h module/layer.cpp header/layer.h header/dataset_utils.h module/dataset_utils.cpp header/neural_network.h module/neural_network.cpp
g++ -std=c++20 -O3 -c main/max_index_main.cpp
g++ -std=c++20 -O3 -o run_max_index max_index_main.o layer.o dataset_utils.o neural_network.o
./run_max_index
rm header/*.o header/*.gch*
