//
//  main.cpp
//  neural_net
//
//  Created by Jacob Mathai on 11/9/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "<YOUR PATH>/neural_net/source/neural_network.h"
#include "<YOUR PATH>/neural_net/neural_net/source/dataset_utils.h"

int main()
{
    std::ifstream train("<YOUR PATH>/data/test_bank.txt"),
                  test("<YOUR PATH>/data/train_bank.txt");
    cout << "Loading csvs..." << endl;
    list<vector<double>> trainframe, testframe;
    list<vector<double>>::iterator i;
    load_csv(train, trainframe);
    load_csv(test, testframe);
    list<Point> train_set, test_set;
    cout << "Parsing train set..." << endl;
    for (i = trainframe.begin(); i != trainframe.end(); ++i)
    {
        Point p(i -> size() - 1, 1);
        p.x = vector<double>(i -> begin(), i -> end() - 2);
        p.y[0] = i -> back();
        train_set.push_back(p);
    }
    cout << "Parsing test set..." << endl;
    for (i = testframe.begin(); i != testframe.end(); ++i)
    {
        Point p;
        p.x = vector<double>(i -> begin(), i -> end() - 2);
        p.y = vector<double>(1, i -> back());
        test_set.push_back(p);
    }
    cout << endl << "Training..." << endl;
    unsigned short int x_shape = train_set.front().x.size(), y_shape = train_set.front().y.size();
    vector<unsigned short int> layer_sizes = {x_shape, 3, y_shape};
    vector<double> biases(layer_sizes.size() - 1, 0.0);
    cout << "Initializing network..." << endl;
    NeuralNetwork net(layer_sizes, false, biases, 0.0, .1);
    cout << net;
    bool trained = net.train(train_set, "sigmoid", .015, "none");
    if (trained)
    {
        cout << "Testing..." << endl;
        double avg_cost = net.predict(test_set, "sigmoid");
        cout << "AVERAGE COST: " << avg_cost << endl;
    }
}
