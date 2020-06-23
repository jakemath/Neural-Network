//
//  node.h
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#ifndef node_h
#define node_h

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <iterator>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cassert>
#include <random>

using std::vector;
using std::string;
using std::list;
using std::map;
using std::cout;
using std::endl;

struct Neuron
{
    Neuron(): activated_value(0.0), transfer_value(0.0), weights_to_next_layer(vector<double>()), error(0.0) {}
    double activated_value;
    double transfer_value;
    double error;
    vector<double> weights_to_next_layer;
};

struct Point
{
    Point() {};
    Point(unsigned long int x_size, unsigned long int y_size):
        x(vector<double>(x_size, 0.0)),
        y(vector<double>(y_size, 0.0)) {}
    vector<double> x;
    vector<double> y;
};

#endif /* node_h */
