#include "NeuralNet.h"

float FeedForwardNeuralNet::randomWeightRange = 1.0f;
GLuint FeedForwardNeuralNet::RElUshader, FeedForwardNeuralNet::ReLUder;
bool FeedForwardNeuralNet::initialized = false;

// Private

// Setup the architecture of the neural net (buffers and arrays for weights and outputs)
void FeedForwardNeuralNet::setupArchitecture(void) {
    // Setup weights and outputs
    this->outputs.push_back(new vector<float>(this->architecture[0]));  // Inputs count as outputs of first layer
    this->ffOutputs = new Matrix(1, this->architecture[0]);
    for (int i = 0; i < this->architecture.size() - 1; i++) {
        this->weights.push_back(new Matrix((this->architecture[i] + 1), this->architecture[i + 1]));  // num weights = num inputs + 1 for bias
        this->outputs.push_back(new vector<float>(this->architecture[i + 1]));
    }
}

// Create random weights for the neural net
// Note: parallelize this later when I make better glsl random number generator
void FeedForwardNeuralNet::createRandomWeights(void) {
    // Generate weights
    for (int i = 0; i < this->weights.size(); i++) {
        weights[i]->randomData(randomWeightRange);
    }
}

// Feed data from layer to layer + 1 (layer 0 is inputs)
// Note: for now we only use ReLU activation
void FeedForwardNeuralNet::feedForward(int layer) {

    cout << "Weights:" << endl;
    weights[layer]->show();

    ffOutputs->addCol(1.0f); // Add bias

    cout << "Before:" << endl;
    ffOutputs->show();

    (*ffOutputs) *= (*weights[layer]);

    cout << "After weights:" << endl;
    ffOutputs->show();

    ffOutputs->map(RElUshader);

    cout << "After ReLU:" << endl;
    ffOutputs->show();

    // Write back to output vector
    delete outputs[layer + 1];
    outputs[layer + 1] = (*ffOutputs)[0];
}

// Load weights of the neural net from a file
void FeedForwardNeuralNet::loadWeights(string path) {
    // for now do nothing
}

// Create shader programs and sort out static variables
void FeedForwardNeuralNet::setupClass(void) {
    // Load the invocation shader
    RElUshader = Utils::createShaderProgram("shaders/neuralNet/RElU.glsl");
    ReLUder = Utils::createShaderProgram("shaders/neuralNet/ReLUderriv.glsl");

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
GLuint FeedForwardNeuralNet::invoke(vector<float> inputs) {
    assert(initialized);

    // Load input data
    this->ffOutputs->resize(1, this->architecture[0]);
    this->ffOutputs->setData(inputs);
    delete this->outputs[0];
    this->outputs[0] = new vector<float>(inputs);

    for (int i = 0; i < this->weights.size(); i++) {
        // Feed forward through the network
        feedForward(i);
    }

    if (softmax) {
        // Vector to store softmax outputs
        vector<float> os;
        // Apply softmax to the output
        float sum = 0.0f;
        for (int i = 0; i < this->outputs.back()->size(); i++) {
            sum += exp(this->outputs.back()->at(i));
        }
        for (int i = 0; i < this->outputs.back()->size(); i++) {
            os.push_back(exp(this->outputs.back()->at(i)) / sum);
        }
        ffOutputs->setData(os);
    }

    return ffOutputs->matCBOs[0]; // SSBO that holds the outputs
}

// Free the memory used by the neural net and release its buffers
void FeedForwardNeuralNet::destroy(void) {
    for (int i = 0; i < this->weights.size(); i++) {
        delete (this->weights[i]);
    }
    for (int i = 0; i < this->outputs.size(); i++) {
        delete this->outputs[i];
    }
}

void FeedForwardNeuralNet::backPropagate(Matrix& expected) {
    // Do nothing for now
}
