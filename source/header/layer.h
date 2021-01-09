//
//  layer.h
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#ifndef layer_h
#define layer_h

#include "neuron.h"

struct Layer {
    Layer(): neurons(vector<Neuron>()) {}
    Layer(unsigned short int neurons_,
          bool random,
          double bias_,
          unsigned short int neurons_in_next_layer,
          double weights_mean,
          double weights_std);
    unsigned short int size() const { return neurons.size(); }
    friend std::ostream& operator <<(std::ostream& out, const Layer& l);
    vector<Neuron> neurons;
    double bias;
};

#endif /* layer_h */
