#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;
GLuint FeedForwardNeuralNet::invocationShader;
bool FeedForwardNeuralNet::initialized = false;

// Private

void FeedForwardNeuralNet::setupArchitecture(void) {
    // Setup compute buffer objects
    this->numCbs = this->architecture.size() * 2;   // 1 for weights, 1 for outputs per layer
    this->cbs.resize(this->numCbs);
    glGenBuffers(this->numCbs, this->cbs.data());

    // Setup weights and outputs
    this->outputs.push_back(new vector<float>(this->architecture[0]));  // Inputs count as outputs of first layer
    this->outputs[0]->resize(this->architecture[0]);
    for (int i = 0; i < this->architecture.size() - 1; i++) {
        this->weights.push_back(new vector<float>((this->architecture[i] + 1) * this->architecture[i + 1]));  // num weights = num inputs + 1 for bias
        this->outputs.push_back(new vector<float>(this->architecture[i + 1]));
        this->outputs[i + 1]->resize(this->architecture[i + 1]);
    }
}

void FeedForwardNeuralNet::createRandomWeights(void) {
    // Note: parallelize this later when I make better glsl random number generator
    // Generate weights
    for (int i = 0; i < this->weights.size(); i++) {
        for (int j = 0; j < this->weights[i]->size(); j++) {
            (*this->weights[i])[j] = (float)rand() / (float)RAND_MAX * randomWeightRange * 2 - randomWeightRange;
        }
    }

    // Load weights into compute buffer objects
    for (int i = 0; i < this->weights.size(); i++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cbs[i * 2]);
        // Note: we use DYNAMIC_COPY because at the moment the outputs are going to be stored back to CPU each time
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->weights[i]->size(), this->weights[i]->data(), GL_DYNAMIC_COPY);
    }
}

void FeedForwardNeuralNet::feedForward(int layer) {
    // Bind the shader
    glUseProgram(invocationShader);

    // Bind the weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->cbs[layer * 2]);      // Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->cbs[layer * 2 + 1]);  // Inputs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->cbs[layer * 2 + 3]);  // Outputs    

    // Dispatch the compute shader
    glDispatchCompute(this->architecture[layer + 1], 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void FeedForwardNeuralNet::loadWeights(string path) {
    // for now do nothing
}

void FeedForwardNeuralNet::setupClass(void) {
    // Load the invocation shader
    invocationShader = Utils::createShaderProgram("shaders/neuralNet/invocation.glsl");

    initialized = true;
}



// Public

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, string weightPath) {
    if (!initialized) setupClass();

    this->architecture = architecture;

    setupArchitecture();
    loadWeights(weightPath);
}

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, uint64_t seed) {
    if (!initialized) setupClass();

    this->architecture = architecture;
    this->seed = seed;

    setupArchitecture();
    createRandomWeights();
}

FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture) {
    FeedForwardNeuralNet::FeedForwardNeuralNet(architecture, (uint64_t)time(NULL));
}

void FeedForwardNeuralNet::invoke(vector<float> *inputs, vector<float> *outputs) {
    if (!initialized) {
        cout << "Neural nets not initialized" << endl;
        return;
    }

    // Load input data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->cbs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * inputs->size(), inputs->data(), GL_DYNAMIC_DRAW);
    memcpy(this->outputs[0]->data(), inputs->data(), sizeof(float) * inputs->size());

    for (int i = 0; i < this->weights.size(); i++) {
        // Feed forward through the network
        feedForward(i); 
        // Get output
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->cbs[(i + 1) * 2 + 1]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * this->outputs[i + 1]->size(), this->outputs[i + 1]->data());
    }

    // Copy outputs to output vector
    outputs->resize(this->architecture.back());
    memcpy(outputs->data(), this->outputs.back()->data(), sizeof(float) * this->architecture.back());
}

void FeedForwardNeuralNet::destroy(void) {
    for (int i = 0; i < this->weights.size(); i++) {
        delete this->weights[i];
        delete this->outputs[i];
    }
    glDeleteBuffers(this->numCbs, this->cbs.data());
}