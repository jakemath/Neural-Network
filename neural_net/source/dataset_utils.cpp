//
//  dataset_utils.cpp
//  neural_net
//
//  Created by Jacob Mathai on 4/8/20.
//  Copyright © 2020 Jacob Mathai. All rights reserved.
//

#include "dataset_utils.h"

list<Point> generate_dataset(unsigned long n, unsigned short int x_shape, unsigned short int y_shape, string type)
{
    list<Point> dataset(n, Point(x_shape, y_shape));
    list<Point>::iterator datapoint;
    unsigned int i, j;
    std::random_device rd;
    if (type == "sum")
    {
        for (datapoint = dataset.begin(), i = 1; datapoint != dataset.end() && i <= n; ++datapoint, ++i)
        {
            for (j = 0; j < x_shape; ++j)
                datapoint -> x[j] = 1.0 * (rd() % 10000 + 1) / 10000.0;
            datapoint -> y[0] = std::accumulate(datapoint -> x.begin(), datapoint -> x.end(), 0.0);
        }
    }
    else if (type == "max_regression")
    {
        for (datapoint = dataset.begin(), i = 1; datapoint != dataset.end() && i <= n; ++datapoint, ++i)
        {
            float max = -10000.0;
            for (j = 0; j < x_shape; ++j)
            {
                datapoint -> x[j] = 1.0 * (rd() % 10000 + 1)/ 10000.0;
                max = std::fmax(datapoint -> x[j], max);
            }
            datapoint -> y[0] = max;
        }
    }
    else if (type == "max_index")
    {
        for (datapoint = dataset.begin(), i = 1; datapoint != dataset.end() && i <= n; ++datapoint, ++i)
        {
            double max_index = 0.0, max = -INT_MAX;
            for (j = 0; j < x_shape; ++j)
            {
                datapoint -> x[j] = 1.0 * (rd() % 10000 + 1) / 10000.0;
                if (datapoint -> x[j] > max)
                {
                    max = datapoint -> x[j];
                    max_index = 1.0*j;
                }
            }
            datapoint -> y[0] = max_index;
        }
    }
    else if (type == "max_index_type_1")
    {
        Point p(x_shape, y_shape);
        p.x[0] = rd() % 5 + 1;
        for (j = 1; j < x_shape; ++j)
        {
            p.x[j] = 1.0 * (rd() % 1000 + 1) / 1000.0;
            while (p.x[j] >= datapoint -> x[0])
                p.x[j] = 1.0 * (rd() % 1000 + 1) / 1000.0;
        }
        p.y[0] = 0;
        for (datapoint = dataset.begin(); datapoint != dataset.end(); ++datapoint)
        {
            datapoint -> x = p.x;
            datapoint -> y = p.y;
//            datapoint -> x[0] = rd() % 5 + 1;
//            for (j = 1; j < x_shape; ++j)
//            {
//                datapoint -> x[j] = 1.0 * (rd() % 1000 + 1) / 1000.0;
//                while (datapoint -> x[j] >= datapoint -> x[0])
//                {
////                    cout << datapoint -> x[0] << ", " << datapoint -> x[j] << endl;
//                    datapoint -> x[j] = 1.0 * (rd() % 1000 + 1) / 1000.0;
//                }
//            }
//            datapoint -> y[0] = 0;
        }
    }
    return dataset;
}

void split(const string& s, char c, vector<double>& p)
{
    string str;
    std::istringstream tokenStream (s);
    while (getline(tokenStream, str, ','))
        p.push_back(stod(str));
}

void load_csv(std::istream& in, list<vector<double>>& frame)
{
    string str;
    while (!in.eof())
    {
        vector<double> p;
        getline(in, str, '\n');
        split(str, ',', p);
        frame.push_back(p);
    }
}

