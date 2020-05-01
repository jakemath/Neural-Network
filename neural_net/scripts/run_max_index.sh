cd .. && rm costs/max_index_cost.txt
cd source
g++ -std=c++2a -c neuron.h layer.cpp layer.h dataset_utils.h dataset_utils.cpp neural_network.h neural_network.cpp
g++ -std=c++2a -c main/max_index_main.cpp
g++ -o run_max_index max_index_main.o layer.o dataset_utils.o neural_network.o
./run_max_index
