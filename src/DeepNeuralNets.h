#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SOIL2/soil2.h>
#include <cstdlib>

#include "Defs.h"

using namespace std;

#ifndef DEEPNEURALNETS_H
#define DEEPNEURALNETS_H

#define MAX_FLOATS 268435456
#define MAX_SSBO_SIZE 1073741824
#define MAX_SSBO 16

#define NUM_NEURAL_NETS numCars

// Currently not an RNN -> if rnn then add numInputs to NUM_INPUTS
#define NUM_INPUTS (numComputerVisionAngles + numCarFloats)
#define NUM_OUTPUTS numInputs

#define NUM_HIDDEN_LAYER_1_NODES 32
#define NUM_HIDDEN_LAYER_2_NODES 16
#define NUM_HIDDEN_LAYER_3_NODES 8

#define NUM_GENERATION_LEADERS 10

#define NUM_NN_CBS 12

class DeepNeuralNets
{
    private:
        // Compute shader variables
        static GLuint neuralNetComputeShader;
        static GLuint evolutionComputeShader;
        static GLuint nnCBOs[NUM_NN_CBS];

        // Network random seeds
        static float seeds[NUM_NEURAL_NETS];

        // Network inputs
        static float *carData;
        static float *computerVisionData;

        // Neural network weights
        static float layer1Weights[(NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_NEURAL_NETS];
        static float layer2Weights[(NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_NEURAL_NETS];
        static float layer3Weights[(NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_NEURAL_NETS];
        static float outputWeights[(NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_NEURAL_NETS];

        // Neural network outputs
        static float layer1Outputs[NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
        static float layer2Outputs[NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
        static float layer3Outputs[NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
        static float outputOutputs[NUM_OUTPUTS * NUM_NEURAL_NETS];

        // Neural network fitness
        static float fitness[NUM_NEURAL_NETS];

        // Generation leaders
        static float genLeadersLayer1Weights[(NUM_INPUTS * NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_GENERATION_LEADERS];
        static float genLeadersLayer2Weights[(NUM_HIDDEN_LAYER_1_NODES * NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_GENERATION_LEADERS];
        static float genLeadersLayer3Weights[(NUM_HIDDEN_LAYER_2_NODES * NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_GENERATION_LEADERS];
        static float genLeadersOutputWeights[(NUM_HIDDEN_LAYER_3_NODES * NUM_OUTPUTS + 1) * NUM_GENERATION_LEADERS];
        static float genLeadersFitness[NUM_GENERATION_LEADERS];

        // Setup
        static void createRandomPopulation(void);

        // Evolution
        static void gatherGenerationLeaders(void);
        static void evolveGenerationLeaders(void);

        // Eport utility
        static void exportModel(string filename, float *layer1Weights, float *layer2Weights, float *layer3Weights, float *outputWeights);
        static void exportPopulationModel(string filename, int index);  // Index in whole population
        static void exportTopModel(string filename, int index);         // Index in generation leaders
    public:
        DeepNeuralNets();

        static void initNeuralNets(float *carData, float *computerVisionData); // pointers to buffers that will be used as inputs
        static float *invokeNeuralNets(void); // return pointer to outputOutputs
        static void setFitness(float *fitnessData); // inform neural nets of their fitness
        static void evolveNeuralNets(void); // evolve neural nets

        // Persistence
        static void exportBestModel(string filename);
        static void importModel(string filename, int index);
};

#endif