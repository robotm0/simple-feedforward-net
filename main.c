// number of epochs the program will train for
#define NUM_EPOCHS 1000000
// print information every increment
#define PRINT_INCREMENT 1000
// // reset the display of the average cost every so many epochs
// #define RESET_COST 1000

#include "network.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  printf("Hello Saqib\n");
  Network net;
  const unsigned int num_layers = 2;
  unsigned int num_nodes[] = {10, 10};
  double* output = (double*)calloc(num_nodes[num_layers-1], sizeof(double));
  double* input = (double*)calloc(num_nodes[0], sizeof(double));
  // stop training once threshold reached
  const double cost_threshold = 0.001;
  printf("Initialising network\n");
  initialise_network(&net, num_layers, num_nodes, 0);
  printf("Randomising network\n");
  randomise_network(&net);
  double average_cost = 0;
  unsigned int i = 0;
  for (; i < NUM_EPOCHS; i++) {
    // autoencoder
    // unsigned int index = floor((rand()*num_nodes[0])/RAND_MAX);
    unsigned int index;
    index = 1;
    input[index] = 1;
    output[index] = 1;
    // printf("Forward pass\n");
    forward_pass(&net, input, output);
    average_cost += net.total_cost;
    if (!(i%PRINT_INCREMENT))
      printf("i: %d Total cost: %f, Average: %f\n", i, net.total_cost, average_cost/i);
    // printf("Backpropagating\n");
    backpropagate(&net, 0.001, 0.001);
    // reset values
    input[index] = 0;
    output[index] = 0;
    if (net.total_cost <= cost_threshold) break;
  }
  // free all network data
  free(output);
  free(input);
  initialise_network(&net, num_layers, num_nodes, 1);
  printf("Done in %d epochs! Cost: %f :D\n", i, net.total_cost);
  return 0;
}
