#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;

// Private





// Public

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, string weightPath)
{
    this->architecture = architecture;
    this->initialized = false;

    setupArchitecture();
    loadWeights(weightPath);

    this->initialized = true;
}

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, uint64_t seed)
{
    this->architecture = architecture;
    this->seed = seed;
    this->initialized = false;

    setupArchitecture();
    createRandomWeights();

    this->initialized = true;
}

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture)
{
    FeedForwardNeuralNet::FeedForwardNeuralNet(architecture, (uint64_t)time(NULL));
}

void FeedForwardNeuralNet::invoke(vector<float> *inputs, vector<float> *outputs)
{
    if (!initialized)
    {
        cout << "Neural net not initialized" << endl;
        return;
    }

    // Invoke
}

void FeedForwardNeuralNet::destroy(void)
{
    for (int i = 0; i < weights.size(); i++)
    {
        delete weights[i];
        delete outputs[i];
    }
    glDeleteBuffers(numCbs, cbs.data());
}