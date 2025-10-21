#pragma once
#include <iostream>
#include <utility>
#include <vector>
#include <numeric>

class Neuron {
    public:
        Neuron(double bias)
            : Bias(std::move(bias))
        {}
        
        double pass(std::vector<double> values) {
            return std::accumulate(values.begin(), values.end(), Bias);
        }
        
    private:
        double Bias;
};