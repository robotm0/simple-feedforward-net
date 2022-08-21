#ifndef NETWORK_H
#define NETWORK_H

#include "matrices.h"

// implement neural network layer structure

enum layerType {LAYER_INPUT, LAYER_HIDDEN, LAYER_OUTPUT};
typedef struct Layer {
  Matrix* input;
  Matrix* weights;
  Matrix* biases;
  Matrix* multiplied;
  Matrix* output;
  enum layerType layer_type;
} Layer;

typedef struct Network {
  Matrix* input;
  Matrix* output;
  Matrix* target_output;
  unsigned int num_layers;
  unsigned int* num_nodes;
  Layer* layers;
  Matrix* cost;
  double total_cost;
} Network;

// implement neural network calculations

void initialise_network(
  Network* network, unsigned int num_layers, unsigned int* num_nodes,
  unsigned int clearNetwork
);

void randomise_network(Network* net);

double activate_hidden(double n);

double activate_hidden_derivative(double n);

double activate_output(double n);

double activate_output_derivative(double n);

void cost(Matrix* output, Matrix* target_output, Matrix* cost_matrix);

double total_cost(Matrix* cost);

void forward_pass(Network* net, double* input, double* target_output);

void backpropagate(
  Network* net, double bias_learning_rate, double weight_learning_rate
);

#endif