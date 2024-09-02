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
#include <chrono>

#include "Defs.h"

using namespace std;

#ifndef DEEPNEURALNETS_H
#define DEEPNEURALNETS_H

#define NUM_NEURAL_NETS numCars

// Currently not an RNN -> if rnn then add numInputs to NUM_INPUTS
#define NUM_INPUTS (numComputerVisionAngles + numCarFloats + 4) // 4 for start line
#define NUM_OUTPUTS numInputs

#define NUM_HIDDEN_LAYER_1_NODES 32
#define NUM_HIDDEN_LAYER_2_NODES 16
#define NUM_HIDDEN_LAYER_3_NODES 8

// Definitely evolve the top 3, then the rest are random
#define NUM_GENERATION_LEADERS 7
#define NUM_WHEEL_CHOICES 3

#define LEARNING_RATE 0.5f

#define NUM_NN_CBS 8
/*
    0: seeds
    1: layer1Weights
    2: layer2Weights
    3: layer3Weights
    4: outputWeights
    5: layer1Outputs
    6: layer2Outputs
    7: layer3Outputs
*/

class DeepNeuralNets
{
    private:
        // General
        static int epoch;
        static int lastCalculatedLeaders;

        // Compute shader variables
        static GLuint Layer1ComputeShader, Layer2ComputeShader, Layer3ComputeShader, OutputComputeShader;
        static GLuint evolutionComputeShader;
        static GLuint randomPopulationComputeShader;
        static GLuint nnCBOs[NUM_NN_CBS];

        // Network random seeds
        static float seeds[NUM_NEURAL_NETS];

        // Network inputs
        static GLuint carData;
        static GLuint computerVisionData;
        static GLuint inputs;

        // Neural network weights
        static float layer1Weights[(NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
        static float layer2Weights[(NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
        static float layer3Weights[(NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
        static float outputWeights[(NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * NUM_NEURAL_NETS];

        // Neural network outputs
        static float layer1Outputs[NUM_HIDDEN_LAYER_1_NODES * NUM_NEURAL_NETS];
        static float layer2Outputs[NUM_HIDDEN_LAYER_2_NODES * NUM_NEURAL_NETS];
        static float layer3Outputs[NUM_HIDDEN_LAYER_3_NODES * NUM_NEURAL_NETS];
        static float outputOutputs[NUM_OUTPUTS * NUM_NEURAL_NETS];

        // Neural network fitness
        static GLuint fitnessSSBO;
        static float fitness[NUM_NEURAL_NETS * numCarFitnessFloats];
        static int topIndices[NUM_GENERATION_LEADERS];
        static int wheelChoices[NUM_WHEEL_CHOICES];

        // Generation leaders
        static float genLeadersLayer1Weights[(NUM_INPUTS + 1) * NUM_HIDDEN_LAYER_1_NODES * NUM_GENERATION_LEADERS];
        static float genLeadersLayer2Weights[(NUM_HIDDEN_LAYER_1_NODES + 1) * NUM_HIDDEN_LAYER_2_NODES * NUM_GENERATION_LEADERS];
        static float genLeadersLayer3Weights[(NUM_HIDDEN_LAYER_2_NODES + 1) * NUM_HIDDEN_LAYER_3_NODES * NUM_GENERATION_LEADERS];
        static float genLeadersOutputWeights[(NUM_HIDDEN_LAYER_3_NODES + 1) * NUM_OUTPUTS * NUM_GENERATION_LEADERS];
        static float genLeadersFitness[NUM_GENERATION_LEADERS];

        // Setup
        static void createRandomPopulation(void);

        // Evolution
        static void calculateGenerationLeaderIndices(void);
        static void spinWheel(void);

        // Eport utility
        static void exportModel(string filename, float *layer1Weights, float *layer2Weights, float *layer3Weights, float *outputWeights);
        static void exportPopulationModel(string filename, int index);  // Index in whole population
        static void exportTopModel(string filename, int index);         // Index in generation leaders
    public:
        DeepNeuralNets();

        static void initNeuralNets(GLuint carData, GLuint computerVisionData, GLuint inputs, GLuint fitness); // pointers to SSBOs
        static void invokeNeuralNets(glm::vec4 startLine);
        static void gatherGenerationLeaders(void);
        static void evolveNeuralNets(void); // evolve neural nets

        // Persistence
        static void exportBestModel(void);
        static void exportGenerationLeaders(void);
        static void importModel(string filename, int index);
};

#endif