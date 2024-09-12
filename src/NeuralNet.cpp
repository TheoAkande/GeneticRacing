#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;
GLuint FeedForwardNeuralNet::invocationShader;
bool FeedForwardNeuralNet::initialized = false;

// Private

// Setup the architecture of the neural net (buffers and arrays for weights and outputs)
void FeedForwardNeuralNet::setupArchitecture(void) {
    // Setup compute buffer objects
    this->numCbs = this->architecture.size() * 2;   // 1 for weights, 1 for outputs per layer
    this->cbs.resize(this->numCbs);
    glGenBuffers(this->numCbs, this->cbs.data());

    // Setup weights and outputs
    this->outputs.push_back(new vector<float>(this->architecture[0]));  // Inputs count as outputs of first layer
    for (int i = 0; i < this->architecture.size() - 1; i++) {
        this->weights.push_back(new vector<float>((this->architecture[i] + 1) * this->architecture[i + 1]));  // num weights = num inputs + 1 for bias
        this->outputs.push_back(new vector<float>(this->architecture[i + 1]));

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->cbs[i * 2 + 3]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * this->architecture[i + 1], NULL, GL_DYNAMIC_DRAW);
    }
}

// Create random weights for the neural net
// Note: parallelize this later when I make better glsl random number generator
void FeedForwardNeuralNet::createRandomWeights(void) {
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

// Feed data from layer to layer + 1 (layer 0 is inputs)
// Note: for now we only use ReLU activation
void FeedForwardNeuralNet::feedForward(int layer) {
    // Bind the shader
    glUseProgram(invocationShader);

    // Set the uniforms
    this->uLoc = glGetUniformLocation(invocationShader, "numInputs");
    glUniform1i(this->uLoc, this->architecture[layer]);
    this->uLoc = glGetUniformLocation(invocationShader, "numOutputs");
    glUniform1i(this->uLoc, this->architecture[layer + 1]);

    // Bind the weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->cbs[layer * 2]);      // Weights
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->cbs[layer * 2 + 1]);  // Inputs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->cbs[layer * 2 + 3]);  // Outputs    

    // Dispatch the compute shader
    glDispatchCompute(this->architecture[layer + 1], 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

// Load weights of the neural net from a file
void FeedForwardNeuralNet::loadWeights(string path) {
    // for now do nothing
}

// Create shader programs and sort out static variables
void FeedForwardNeuralNet::setupClass(void) {
    // Load the invocation shader
    invocationShader = Utils::createShaderProgram("shaders/neuralNet/feedforward.glsl");

    initialized = true;
}



// Public

// Initialize with weights from file
FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, string weightPath, bool softmax) {
    if (!initialized) setupClass();

    this->architecture = architecture;
    this->softmax = softmax;

    setupArchitecture();
    loadWeights(weightPath);
}

// Initialize with random weights and seed
FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, uint64_t seed, bool softmax) {
    if (!initialized) setupClass();

    this->architecture = architecture;
    this->seed = seed;
    this->softmax = softmax;

    setupArchitecture();
    createRandomWeights();
}

// Initialize with random weights and seed from time
FeedForwardNeuralNet::FeedForwardNeuralNet(vector<int> architecture, bool softmax) 
    : FeedForwardNeuralNet(architecture, (uint64_t)time(NULL), softmax) {}

// Feed inputs through the network and store the outputs in the outputs vector
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

    outputs->clear();
    if (softmax) {
        // Apply softmax to the output
        float sum = 0.0f;
        for (int i = 0; i < this->outputs.back()->size(); i++) {
            sum += exp(this->outputs.back()->at(i));
        }
        for (int i = 0; i < this->outputs.back()->size(); i++) {
            outputs->push_back(exp(this->outputs.back()->at(i)) / sum);
        }
    } else {
        // Copy outputs to output vector
        outputs->resize(this->architecture.back());
        memcpy(outputs->data(), this->outputs.back()->data(), sizeof(float) * this->architecture.back());
    }
}

// Free the memory used by the neural net and release its buffers
void FeedForwardNeuralNet::destroy(void) {
    for (int i = 0; i < this->weights.size(); i++) {
        delete this->weights[i];
    }
    for (int i = 0; i < this->outputs.size(); i++) {
        delete this->outputs[i];
    }
    glDeleteBuffers(this->numCbs, this->cbs.data());
}

void FeedForwardNeuralNet::backPropagate(vector<float> *expected) {
    // Do nothing for now
}
