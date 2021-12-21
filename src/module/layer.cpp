//
//  layer.cpp
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "../header/layer.h"

Layer::Layer(unsigned short int neurons_,
             bool random,
             double bias_,
             unsigned short int neurons_in_next_layer,
             double weights_mean,
             double weights_std) {
    neurons = vector<Neuron>(neurons_);
    bias = bias_;
    if (neurons_ > 0 && neurons_in_next_layer > 0) {
        unsigned short int neuron;
        if (random) {
            unsigned short int weight;
            unsigned long int seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<double> norm(weights_mean, weights_std);
            for (auto& neuron : neurons) { // Randomize weights from nodes to next layer
                neuron.weights_to_next_layer = vector<double>(neurons_in_next_layer);
                for (const auto& weight : neuron.weights_to_next_layer)
                    neuron.weights_to_next_layer[weight] = norm(generator);
            }
        }
        else {
            for (auto& neuron : neurons) // Randomize weights from nodes to next layer
                neuron.weights_to_next_layer = vector<double>(neurons_in_next_layer, 0.0);
        }
    }
}

std::ostream& operator <<(std::ostream& out, const Layer& l) {
    for (unsigned short int i = 0; i < l.neurons.size(); ++i) {
        out << "NODE " << i + 1 << " Value = " << l.neurons[i].activated_value << " -> "
            << l.neurons[i].transfer_value << ": ";
        for (unsigned int j = 0; j < l.neurons[i].weights_to_next_layer.size(); ++j)
            out << l.neurons[i].weights_to_next_layer[j] << ", ";
        out << endl;
    }
    out << "BIAS = " << l.bias << endl;
    return out;
}
