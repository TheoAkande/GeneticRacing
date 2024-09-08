#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;

// Private





// Public

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, uint64_t seed)
{
    this->architecture = architecture;
    this->seed = seed;
    this->initialized = false;

    setupArchitecture();
    createRandomWeights();

    this->initialized = true;
}