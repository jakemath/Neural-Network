//
//  dataset_utils.h
//  neural_net
//
//  Created by Jacob Mathai on 4/8/20.
//  Copyright © 2020 Jacob Mathai. All rights reserved.
//

#ifndef dataset_utils_h
#define dataset_utils_h

#include "neuron.h"

list<Point> generate_dataset(unsigned long n, unsigned short int x_shape, unsigned short int y_shape, string type);
void split(const string& s, char c, vector<double>& p);
void load_csv(std::istream& in, list<vector<double>>& frame);

#endif /* dataset_utils_h */
