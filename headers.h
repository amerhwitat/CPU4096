
#ifndef HEADERS_H
#define HEADERS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <map>



double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
};

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
};

double random_weight() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(gen);
};

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;

    Neuron(int inputs) {
        for (int i = 0; i < inputs; i++)
            weights.push_back(random_weight());
        bias = random_weight();
    }

    double activate(const std::vector<double>& inputs) {
        double sum = bias;
        for (size_t i = 0; i < inputs.size(); i++)
            sum += inputs[i] * weights[i];

        output = sigmoid(sum);
        return output;
    }
};



class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int neuron_count, int inputs_per_neuron) {
        for (int i = 0; i < neuron_count; i++)
            neurons.emplace_back(inputs_per_neuron);
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& n : neurons)
            outputs.push_back(n.activate(inputs));
        return outputs;
    }
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    double learning_rate = 0.1;

    NeuralNetwork(const std::vector<int>& topology) {
        for (size_t i = 1; i < topology.size(); i++)
            layers.emplace_back(topology[i], topology[i - 1]);
    }

    std::vector<double> forward(std::vector<double> input) {
        for (auto& layer : layers)
            input = layer.forward(input);
        return input;
    }

    void backpropagate(const std::vector<double>& input,
                       const std::vector<double>& target) {

        // Forward
        std::vector<std::vector<double>> layer_inputs;
        std::vector<double> outputs = input;
        layer_inputs.push_back(input);

        for (auto& layer : layers) {
            outputs = layer.forward(outputs);
            layer_inputs.push_back(outputs);
        }

        // Output layer deltas
        for (size_t i = 0; i < layers.back().neurons.size(); i++) {
            auto& n = layers.back().neurons[i];
            n.delta = (target[i] - n.output) * sigmoid_derivative(n.output);
        }

        // Hidden layers
        for (int l = layers.size() - 2; l >= 0; l--) {
            for (size_t i = 0; i < layers[l].neurons.size(); i++) {
                double error = 0.0;
                for (auto& next : layers[l + 1].neurons)
                    error += next.weights[i] * next.delta;

                layers[l].neurons[i].delta =
                    error * sigmoid_derivative(layers[l].neurons[i].output);
            }
        }

        // Update weights
        for (size_t l = 0; l < layers.size(); l++) {
            for (auto& n : layers[l].neurons) {
                for (size_t w = 0; w < n.weights.size(); w++)
                    n.weights[w] += learning_rate * n.delta * layer_inputs[l][w];
                n.bias += learning_rate * n.delta;
            }
        }
    }
};


std::vector<double> encode_word(const std::string& word, int max_len = 10) {
    std::vector<double> v(max_len * 26, 0.0);

    for (size_t i = 0; i < word.size() && i < max_len; i++) {
        char c = std::tolower(word[i]);
        if (c >= 'a' && c <= 'z')
            v[i * 26 + (c - 'a')] = 1.0;
    }
    return v;
};

//#include "core.cpp"
//#include "Neuron.cpp"
//#include "Layer.cpp"
//#include "NeuralNetwork.cpp"
//#include "encode_word.cpp"

#endif
