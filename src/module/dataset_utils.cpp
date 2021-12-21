//
//  dataset_utils.cpp
//  neural_net
//
//  Created by Jacob Mathai on 4/8/20.
//  Copyright © 2020 Jacob Mathai. All rights reserved.
//

#include "../header/dataset_utils.h"


std::ostream& operator <<(std::ostream& out, const vector<double>& v) {
    unsigned int i = 0;
    out << "[";
    for (; i < v.size() - 1; ++i)
        out << v[i] << ", ";
    out << v[i] << "]";
    return out;
}


list<Point> generate_dataset(unsigned long n,
                             unsigned short int x_shape,
                             unsigned short int y_shape,
                             string type) {
    list<Point> dataset(n, Point(x_shape, y_shape));
    list<Point>::iterator datapoint;
    unsigned int i, j;
    std::random_device rd;
    if (type == "sum") {
        for (datapoint = dataset.begin(), i = 1; datapoint != dataset.end() && i <= n; ++datapoint, ++i) {
            for (j = 0; j < x_shape; ++j)
                datapoint->x[j] = 1.0 * (rd() % 10000 + 1) / 10000.0;
            datapoint->y[0] = std::accumulate(datapoint->x.begin(), datapoint->x.end(), 0.0);
        }
    }
    else if (type == "max_regression") {
        for (datapoint = dataset.begin(), i = 1; datapoint != dataset.end() && i <= n; ++datapoint, ++i) {
            float max = -10000.0;
            for (j = 0; j < x_shape; ++j) {
                datapoint->x[j] = 1.0 * (rd() % 10000 + 1)/ 10000.0;
                max = std::fmax(datapoint->x[j], max);
            }
            datapoint->y[0] = max;
        }
    }
    else if (type == "max_index") {
        for (datapoint = dataset.begin(), i = 1; datapoint != dataset.end() && i <= n; ++datapoint, ++i) {
            double max_index = 0.0, max = -INT_MAX;
            for (j = 0; j < x_shape; ++j) {
                datapoint->x[j] = 1.0 * (rd() % 10000 + 1) / 10000.0;
                if (datapoint->x[j] > max) {
                    max = datapoint->x[j];
                    max_index = 1.0*j;
                }
            }
            datapoint->y[0] = max_index;
        }
    }
    else if (type == "max_index_const") {
        dataset = list<Point>(50000, Point(x_shape, y_shape));
        vector<Point> const_points(n, Point(x_shape, y_shape));
        for (i = 0; i < n; ++i) { // Generate const points
            Point p(x_shape, y_shape);
            p.x[i] = 1.0 * (rd() % 1000 + 1) / 1000.0;
            for (j = 0; j < x_shape; ++j) {
                if (j != i) {
                    p.x[j] = 1.0 * (rd() % 1000 + 1) / 1000.0;
                    while (p.x[j] >= p.x[i])
                        p.x[j] = 1.0 * (rd() % 1000 + 1) / 1000.0;
                }
            }
            p.y[0] = i;
            cout << p.x << p.y << endl;
            const_points[i] = p;
        }
        list<Point>::iterator offset = dataset.end();
        if (n > 1) {
            for (i = 0; i < n - 1; ++i)
                --offset;
        }
        for (datapoint = dataset.begin(); datapoint != offset;) {
            for (i = 0; i < n; ++i, ++datapoint) {
                datapoint->x = const_points[i].x;
                datapoint->y = const_points[i].y;
            }
        }
    }
    return dataset;
}


void split(const string& s, char c, vector<double>& point) {
    string str;
    std::istringstream tokenStream(s);
    while (getline(tokenStream, str, ','))
        point.push_back(stod(str));
}

void load_csv(std::istream& in, list<vector<double>>& frame) {
    string str;
    while (!in.eof()) {
        getline(in, str, '\n');
        if (!str.empty()) {
            vector<double> point;
            split(str, ',', point);
            frame.push_back(point);
        }
    }
}

