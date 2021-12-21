//
//  max_index_main.cpp
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "../header/neural_network.h"
#include "../header/dataset_utils.h"

int main() {
    unsigned short int x_shape = 5, y_shape = 1;
    string function = "none";
    string dataset_type = "max_index_const";
    cout << "Making dataset..." << endl;
    list<Point> train_set = generate_dataset(3, x_shape, y_shape, dataset_type);
    cout << endl << "Training..." << endl;
    vector<unsigned short int> layer_sizes = {x_shape, 5, 3, y_shape};
    vector<double> biases(layer_sizes.size() - 1, 0.0);
    cout << "Initializing network..." << endl;
    NeuralNetwork net(layer_sizes, true, biases, 0.0, .1);
    cout << net;
    bool trained = net.train(train_set, function, .001, dataset_type, false);
    if (trained) {
        cout << "Testing..." << endl;
        double avg_cost = net.predict(train_set, function);
        cout << "AVERAGE COST: " << avg_cost << endl;
    }
}
