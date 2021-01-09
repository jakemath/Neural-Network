//
//  node.h
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

#include <map>
#include <list>
#include <tuple>
#include <vector>
#include <random>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <iterator>
#include <stdexcept>

using std::map;
using std::cout;
using std::list;
using std::endl;
using std::string;
using std::vector;

struct Neuron {
    Neuron(): activated_value(0.0), transfer_value(0.0), weights_to_next_layer(vector<double>()), error(0.0) {}
    double activated_value;
    double transfer_value;
    double error;
    vector<double> weights_to_next_layer;
};

struct Point {
    Point() {};
    Point(unsigned long int x_size, unsigned long int y_size):
        x(vector<double>(x_size, 0.0)),
        y(vector<double>(y_size, 0.0)) {}
    vector<double> x;
    vector<double> y;
};

#endif /* neuron_h */
