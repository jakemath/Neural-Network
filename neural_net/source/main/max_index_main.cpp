//
//  max_index_main.cpp
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "/Users/Jacob/Desktop/Programming/C++/neural_net/neural_net/source/neural_network.h"
#include "/Users/Jacob/Desktop/Programming/C++/neural_net/neural_net/source/dataset_utils.h"

int main()
{
    unsigned short int x_shape = 2, y_shape = 1;
    cout << "Making dataset..." << endl;
    list<Point> train_set = generate_dataset(500000, x_shape, y_shape, "max_index_type_1");
    cout << endl << "Training..." << endl;
    vector<unsigned short int> layer_sizes = {x_shape, y_shape};
    vector<double> biases(layer_sizes.size() - 1, 0.0);
    cout << "Initializing network..." << endl;
    NeuralNetwork net(layer_sizes, true, biases, 0.0, .001);
    cout << net;
    bool trained = net.train(train_set, "sigmoid", .01, "max_index_type_1");
    if (trained)
    {
        list<Point> test_set = generate_dataset(50000, x_shape, y_shape, "max_index_type_1");
        cout << "Testing..." << endl;
        double avg_cost = net.predict(test_set, "sigmoid");
        cout << "AVERAGE COST: " << avg_cost << endl;
    }
}
