// Artificial Neural Network
// Version 1
// Robot_M0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// commment out if not to print verbose
// #define PRINT_VERBOSE 1
#include "network.h"

// implement neural network calculations

void initialise_network(
  Network* network, unsigned int num_layers, unsigned int* num_nodes,
  unsigned int clearNetwork
) {
  network->num_layers = num_layers;
  network->num_nodes = num_nodes;
  // create network input
  if (clearNetwork) {
    free_matrix(network->input);
  } else {
    network->input = create_empty_matrix(network->num_nodes[0], 1);
  }
  // create network output
  if (clearNetwork) {
    free_matrix(network->output);
  } else {
    network->output = create_empty_matrix(
      network->num_nodes[network->num_layers-1], 1
    );
  }
  // create network target output
  if (clearNetwork) {
    free_matrix(network->target_output);
  } else {
    network->target_output = create_empty_matrix(
      network->num_nodes[network->num_layers-1], 1
    );
  }
  // create network cost
  if (clearNetwork) {
    free_matrix(network->cost);
  } else {
    network->cost = create_empty_matrix(
      network->output->rows, 1
    );
  }
  if (!clearNetwork) {
    // allocate memory to layers
    network->layers = (Layer*)malloc(sizeof(Layer) * network->num_layers);
  }
  for (unsigned int l = 0; l < network->num_layers; l++) {
    // create network input
    if (clearNetwork) {
      free_matrix(network->layers[l].input);
    } else {
      network->layers[l].input = create_empty_matrix(
        network->num_nodes[l], 1
      );
    }
    // set layer type
    if (l == network->num_layers - 1) {
      // if output layer
      (network->layers[l]).layer_type = LAYER_OUTPUT;
    } else if (l == 0) {
      // if input layer
      (network->layers[l]).layer_type = LAYER_INPUT;
    } else {
      // if hidden layer
      (network->layers[l]).layer_type = LAYER_HIDDEN;
    }
    // output layer
    if ((network->layers[l]).layer_type != LAYER_OUTPUT) {
      // create layer output
      if (clearNetwork) {
        free_matrix(network->layers[l].output);
      } else {
        network->layers[l].output = create_empty_matrix(
          network->num_nodes[l+1], 1
        );
      }
      // create multiplied matrix
      if (clearNetwork) {
        free_matrix(network->layers[l].multiplied);
      } else {
        network->layers[l].multiplied = create_empty_matrix(
          network->layers[l].output->rows, 1
        );
      }
    } else {
      // create layer output
      if (clearNetwork) {
        free_matrix(network->layers[l].output);
      } else {
        network->layers[l].output = create_empty_matrix(
          network->layers[l].input->rows, 1
        );
      }
      // create layer multiplied matrix
      if (clearNetwork) {
        free_matrix(network->layers[l].multiplied);
      } else {
        network->layers[l].multiplied = create_empty_matrix(
          network->layers[l].output->rows, 1
        );
      }
    }
    if ((network->layers[l]).layer_type != LAYER_OUTPUT) {
      // if input or hidden layer
      // create layer weights
      if (clearNetwork) {
        free_matrix(network->layers[l].weights);
      } else {
        network->layers[l].weights = create_empty_matrix(
          network->layers[l].output->rows,
          network->layers[l].input->rows
        );
      }
    }
    // create layer biases
    if (clearNetwork) {
      free_matrix(network->layers[l].biases);
    } else {
      network->layers[l].biases = create_empty_matrix(
        network->layers[l].output->rows, 1
      );
    }
  }
  // clear the network layers
  if (clearNetwork) {
    free(network->layers);
  }
}

void randomise_network(Network* net) {
  for (unsigned int l = 0; l < net->num_layers; l++) {
    if ((net->layers[l]).layer_type != LAYER_OUTPUT) {
      // if input or hidden layer
      unsigned int weights_size = (net->layers[l]).weights->rows;
      weights_size *= (net->layers[l]).weights->columns;
      // randomise layer weights
      for (unsigned int i = 0; i < weights_size; i++) {
        (net->layers[l]).weights->matrix_data[i] = random_normal();
      }
    }
    // randomise layer biases
    for (unsigned int j = 0; j < (net->layers[l]).biases->rows; j++) {
      (net->layers[l]).biases->matrix_data[j] = random_normal();
    }
  }
}

double activate_hidden(double n) {
  // activation function of hidden layer
  // return (1 / ( 1 + (float)exp(-1*(double)n) ));
  return (double)atan(n);
}

double activate_hidden_derivative(double n) {
  // derivative of activation function of hidden layer
  // return activate_hidden(n) * (1.0 - activate_hidden(n));
  return (double)(1 / ((n*n) - 1));
}

double activate_output(double n) {
  // activation function of output layer
  return ((double)atan(n));
}

double activate_output_derivative(double n) {
  // derivative of activation function of output layer
  return (double)(1 / ((n*n) - 1));
}

void cost(Matrix* output, Matrix* target_output, Matrix* cost_matrix) {
  // Euclidean distance from output to target output, squared
  unsigned int output_size = output->columns * output->rows;
  unsigned int targ_output_size = target_output->columns * target_output->rows;
  if (output_size != targ_output_size) {
    printf("Error: Cost: matrices not the same size: %d x %d, %d x %d\n",
    output->rows, output->columns, target_output->rows, target_output->columns);
    exit(1);
  } else {
    free(cost_matrix->matrix_data);
    double* cost_matrix_data = (double*)malloc(sizeof(double) * output_size);
    for (unsigned int i = 0; i < output_size; i++) {
      // cost_matrix_data[i] = pow(
      //   target_output->matrix_data[i] - output->matrix_data[i],
      //   2.0);
      // return absolute value
      cost_matrix_data[i] = fabs(
        target_output->matrix_data[i]
        - output->matrix_data[i]
      );
    }
    cost_matrix->matrix_data = cost_matrix_data;
  }
}

double total_cost(Matrix* cost) {
  unsigned int matrix_size = cost->rows;
  double total_cost = 0;
  for (unsigned int i = 0; i < matrix_size; i++) {
    total_cost += fabs(cost->matrix_data[i]);
  }
  return total_cost;
}

void forward_pass(Network* net, double* input, double* target_output) {
  #ifdef PRINT_VERBOSE
  printf("=================================\n");
  printf("Network:\n");
  printf("Network input:\n");
  #endif
  for (unsigned int i = 0; i < net->input->rows; i++) {
    net->input->matrix_data[i] = input[i];
  }
  for (unsigned int i = 0; i < net->target_output->rows; i++) {
    net->target_output->matrix_data[i] = target_output[i];
  }
  #ifdef PRINT_VERBOSE
  print_matrix(net->input);
  #endif
  for (unsigned int l = 0; l < net->num_layers; l++) {
    #ifdef PRINT_VERBOSE
    printf("=== Layer %d ===\n", l);
    #endif
    Layer* cur_layer = &net->layers[l];
    // set layer inputs
    // free(cur_layer->input);
    if (cur_layer->layer_type == LAYER_INPUT) {
      copy_matrix(net->input, cur_layer->input);
      // cur_layer->input = net->input;
    } else {
      copy_matrix(net->layers[l-1].output, cur_layer->input);
      // cur_layer->input = net->layers[l-1].output;
    }
    #ifdef PRINT_VERBOSE
    printf("Layer input:\n");
    print_matrix(cur_layer->input);
    #endif
    // forward pass layer
    if (cur_layer->layer_type != LAYER_OUTPUT) {
      // is input or hidden layer
      Matrix* multiplied;
      multiplied = multiply(cur_layer->weights, cur_layer->input);
      #ifdef PRINT_VERBOSE
      printf("Multiplied:\n");
      print_matrix(multiplied);
      #endif
      copy_matrix(multiplied, cur_layer->multiplied);
      add(multiplied, cur_layer->biases, multiplied);
      #ifdef PRINT_VERBOSE
      printf("Multiplied + biases:\n");
      print_matrix(multiplied);
      #endif
      // activate output
      unsigned int matrix_size = multiplied->rows * multiplied->columns;
      for (unsigned int i = 0; i < matrix_size; i++) {
        multiplied->matrix_data[i] = activate_hidden(
          multiplied->matrix_data[i]);
      }
      // set output
      free_matrix(cur_layer->output);
      cur_layer->output = multiplied;
    } else {
      // is output layer
      // free_matrix(cur_layer->output);
      add(cur_layer->input, cur_layer->biases, cur_layer->multiplied);
      // activate output
      Matrix* multiplied = cur_layer->multiplied;
      unsigned int matrix_size = multiplied->rows * multiplied->columns;
      for (unsigned int i = 0; i < matrix_size; i++) {
        cur_layer->output->matrix_data[i] = activate_hidden(
          multiplied->matrix_data[i]);
      }
      // free_matrix(net->output);
      copy_matrix(cur_layer->output, net->output);
    }
    #ifdef PRINT_VERBOSE
    printf("Layer output:\n");
    print_matrix(cur_layer->output);
    #endif
  }
  cost(net->output, net->target_output, net->cost);
  net->total_cost = total_cost(net->cost);
  #ifdef PRINT_VERBOSE
  printf("Network output:\n");
  print_matrix(net->output);
  printf("Target output:\n");
  print_matrix(net->target_output);
  printf("Cost:\n");
  print_matrix(net->cost);
  printf("Total network cost: %f\n", net->total_cost);
  #endif
}

void backpropagate(
  Network* net, double bias_learning_rate, double weight_learning_rate
) {
  // perform backpropagation
  Matrix* delta;
  for (unsigned int l = net->num_layers - 1; l > 0; l--) {
    #ifdef PRINT_VERBOSE
    printf("=== Layer %d ===\n", l);
    #endif
    Layer* cur_layer = &net->layers[l];
    if (cur_layer->layer_type == LAYER_OUTPUT) {
      delta = create_empty_matrix(
        cur_layer->output->rows,
        1
      );
      unsigned int matrix_size = delta->rows;
      for (unsigned int i = 0; i < matrix_size; i++) {
        // delta->matrix_data[i] = (
        //   net->cost->matrix_data[i]
        // );
        // Layer* final_layer = &(net->layers[net->num_layers-1]);
        delta->matrix_data[i] = (
          // activate_output_derivative(
          //   final_layer->multiplied->matrix_data[i]
          // )
          // *
          -(
            net->target_output->matrix_data[i]
            - net->output->matrix_data[i]
          )
        );
      }
      // copy_matrix(net->cost, delta);
      #ifdef PRINT_VERBOSE
      printf("Delta:\n");
      print_matrix(delta);
      #endif
      // update biases
      for (unsigned int i = 0; i < cur_layer->biases->rows; i++) {
        cur_layer->biases->matrix_data[i] -= (
          delta->matrix_data[i] * bias_learning_rate
        );
      }
    } else {
      // is hidden or input layer
      // update weights
      Matrix* weight_delta = create_empty_matrix(
        cur_layer->weights->rows, cur_layer->weights->columns
      );
      outer_product(delta, net->layers[l-1].output, weight_delta);
      #ifdef PRINT_VERBOSE
      printf("Weights:\n"); print_matrix(cur_layer->weights);
      printf("Delta:\n"); print_matrix(delta);
      printf("Previous output:\n"); print_matrix(net->layers[l-1].output);
      printf("Weight delta:\n"); print_matrix(weight_delta);
      #endif
      unsigned int weights_size = cur_layer->weights->rows;
      weights_size *= cur_layer->weights->columns;
      for (unsigned int i = 0; i < weights_size; i++) {
        cur_layer->weights->matrix_data[i] -= (
          weight_delta->matrix_data[i] * weight_learning_rate
        );
      }
      // update biases
      for (unsigned int i = 0; i < cur_layer->biases->rows; i++) {
        cur_layer->biases->matrix_data[i] -= (
          delta->matrix_data[i] * bias_learning_rate
        );
      }
      #ifdef PRINT_VERBOSE
      printf("Delta:\n");
      print_matrix(delta);
      #endif
      // free weight_delta
      free_matrix(weight_delta);
      // compute delta for next layer
      if (l > 0) {
        Matrix* transposed = create_empty_matrix(
          cur_layer->weights->rows,
          cur_layer->weights->columns
        );
        transpose(cur_layer->weights, transposed);
        Matrix* multiplied = multiply(
          transposed,
          delta
        );
        Matrix* activation_gradient = create_empty_matrix(
          net->layers[l-1].output->rows,
          1
        );
        unsigned int matrix_size = activation_gradient->rows;
        for (unsigned int i = 0; i < matrix_size; i++) {
          activation_gradient->matrix_data[i] = activate_output_derivative(
            net->layers[l-1].output->matrix_data[i]
          );
        }
        hadamard_product(activation_gradient, multiplied, multiplied);
        free_matrix(delta);
        delta = multiplied;
        free_matrix(transposed);
        free_matrix(activation_gradient);
        #ifdef PRINT_VERBOSE
        printf("New delta:\n");
        print_matrix(delta);
        #endif
      }
    }
  }
  free_matrix(delta);
}