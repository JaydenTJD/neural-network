#pragma once
#include <iostream>
#include <vector>
#include "./neuron.hpp"

class Layer {
    public:
        Layer(int height) {
            for (int i = 0; i < height; i++) {
                neurons.push_back(Neuron{0.0});
            }
        }

        int getHeight() {
            return neurons.size();
        }

        std::vector<double> pass(std::vector<std::vector<double>> inValues) {
            std::vector<double> outValues;
            for (int i = 0; i < getHeight(); i++) {
                outValues.push_back(neurons.at(i).pass(inValues.at(i)));
            }
            return outValues;
        }

    private:
        std::vector<Neuron> neurons;
};