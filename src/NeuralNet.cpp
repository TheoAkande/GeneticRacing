#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;
GLuint FeedForwardNeuralNet::invocationShader;
bool FeedForwardNeuralNet::initialized = false;

// Private





// Public

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, string weightPath)
{
    if (!initialized) setupClass();

    this->architecture = architecture;

    setupArchitecture();
    loadWeights(weightPath);
}

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, uint64_t seed)
{
    if (!initialized) setupClass();

    this->architecture = architecture;
    this->seed = seed;

    setupArchitecture();
    createRandomWeights();
}

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture)
{
    FeedForwardNeuralNet::FeedForwardNeuralNet(architecture, (uint64_t)time(NULL));
}

void FeedForwardNeuralNet::invoke(vector<float> *inputs, vector<float> *outputs)
{
    if (!initialized)
    {
        cout << "Neural nets not initialized" << endl;
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