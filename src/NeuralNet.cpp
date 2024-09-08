#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;
GLuint FeedForwardNeuralNet::invocationShader;
bool FeedForwardNeuralNet::initialized = false;

// Private

void FeedForwardNeuralNet::setupArchitecture(void)
{
    // Setup compute buffer objects
    numCbs = architecture.size() * 2;   // 1 for weights, 1 for outputs per layer
    cbs.resize(numCbs);
    glGenBuffers(numCbs, cbs.data());

    // Setup weights and outputs
    outputs.push_back(new vector<float>(architecture[0]));  // Inputs count as outputs of first layer
    for (int i = 0; i < architecture.size() - 1; i++)
    {
        weights.push_back(new vector<float>((architecture[i] + 1) * architecture[i + 1]));  // num weights = num inputs + 1 for bias
        outputs.push_back(new vector<float>(architecture[i + 1]));
    }
}

void FeedForwardNeuralNet::createRandomWeights(void)
{
    // Note: parallelize this later when I make better glsl random number generator
    // Generate weights
    for (int i = 0; i < weights.size(); i++)
    {
        for (int j = 0; j < weights[i]->size(); j++)
        {
            (*weights[i])[j] = (float)rand() / (float)RAND_MAX * randomWeightRange * 2 - randomWeightRange;
        }
    }

    // Load weights into compute buffer objects
    for (int i = 0; i < weights.size(); i++)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbs[i * 2]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * weights[i]->size(), weights[i]->data(), GL_STATIC_DRAW);
    }
}



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