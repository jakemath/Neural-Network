//
//  neural_network.h
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#ifndef neural_network_h
#define neural_network_h

#include "layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork(): layers(vector<Layer>()) {}
    NeuralNetwork(vector<unsigned short int>& layer_sizes,
                   bool random,
                   vector<double>& biases,
                   double weights_mean,
                   double weights_std);
    vector<Layer>& get_layers() { return layers; }
    bool train(list<Point>& dataset,
               string transfer_function,
               double lr,
               string dataset_type,
               bool normalize_lr);
    vector<double> forward_propagate(Point& datapoint, string transfer_function);
    double backpropagate(Point& datapoint, string transfer_function);
    void update_weights(Point& datapoint, double lr, bool normalize_lr);
    double predict(list<Point>& dataset, string transfer_function);
    friend std::ostream& operator <<(std::ostream& out, const NeuralNetwork& n);
private:
    vector<Layer> layers;
};

double relu(double z);
double relu_derivative(double activated_z);
double sigmoid(double z);
double sigmoid_derivative(double activated_z);
double tanh_derivative(double activated_z);
double none_function(double z);
double none_derivative(double z);
void classify(vector<double>& prediction);

std::ostream& operator <<(std::ostream& out, const vector<double>& v);
std::ostream& operator <<(std::ostream& out, const NeuralNetwork& n);
#endif /* neural_network_h */
