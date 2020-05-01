rm costs/bank_cost.txt
cd ~/desktop/programming/c++/neural_net/neural_net/source
g++ -std=c++2a -c neuron.h layer.cpp layer.h dataset_utils.h dataset_utils.cpp neural_network.h neural_network.cpp
g++ -std=c++2a -c main/bank_main.cpp
g++ -o run_bank bank_main.o layer.o dataset_utils.o neural_network.o
./run_bank
