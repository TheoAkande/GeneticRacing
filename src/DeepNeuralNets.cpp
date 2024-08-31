#include "DeepNeuralNets.h"
#include "Utils.h"

// Compute shader variables
GLuint DeepNeuralNets::neuralNetComputeShader;
GLuint DeepNeuralNets::evolutionComputeShader;
GLuint DeepNeuralNets::nnCBOs[NUM_NN_CBS];

// Network random seeds
float DeepNeuralNets::seeds[NUM_NEURAL_NETS];

// Network inputs
float *DeepNeuralNets::carData;
float *DeepNeuralNets::computerVisionData;

// Neural network weights
float DeepNeuralNets::layer1Weights[(NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_NEURAL_NETS];
float DeepNeuralNets::layer2Weights[(NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_NEURAL_NETS];
float DeepNeuralNets::layer3Weights[(NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_NEURAL_NETS];
float DeepNeuralNets::outputWeights[(NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_NEURAL_NETS];

// Neural network outputs
float DeepNeuralNets::layer1Outputs[NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer2Outputs[NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::layer3Outputs[NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
float DeepNeuralNets::outputOutputs[NUM_OUTPUTS * NUM_NEURAL_NETS];

// Neural network fitness
float DeepNeuralNets::fitness[NUM_NEURAL_NETS];

// Generation leaders
float DeepNeuralNets::genLeadersLayer1Weights[(NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersLayer2Weights[(NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersLayer3Weights[(NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersOutputWeights[(NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_GENERATION_LEADERS];
float DeepNeuralNets::genLeadersFitness[NUM_GENERATION_LEADERS];