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
    // Setup learning gradients
    for (int i = 0; i < this->weights.size(); i++) {
        this->gradients.push_back(new Matrix(this->weights[i]->rows, this->weights[i]->cols));
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
    // Add bias
    ffOutputs->addCol(1.0f); 

    // Multiply by weights
    (*ffOutputs) *= (*weights[layer]);

    // Apply ReLU
    ffOutputs->map(RElUshader); 

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

void FeedForwardNeuralNet::backPropagate(Matrix& expected, bool clear) {
    // Refresh gradients if needed
    if (clear) {
        for (int i = 0; i < this->weights.size(); i++) {
            delete this->gradients[i];
            this->gradients[i] = new Matrix(this->weights[i]->rows, this->weights[i]->cols);
        }
    }

    // Calculate the error
    Matrix& errorDelta = expected - *ffOutputs;

    // Apply the derivative of the ReLU function to last outputs
    ffOutputs->map(ReLUder);

    // Calculate the delta for the last layer
    errorDelta *= *ffOutputs;

    // Store the delta
    vector<Matrix *> deltas;
    deltas.push_back(&errorDelta);

    Matrix *delta;

    // Calculate the deltas for the remaining layers
    for (int i = this->weights.size() - 1; i > 0; i--) {
        // Calculate the delta
        Matrix *weightTranspose = weights[i]->transpose();

        // We only want to add the bias row if a bias is expected in the next layer (i.e. not the output layer)
        if (i < this->weights.size() - 1) {
            // Add bias row
            weightTranspose->addRow(1.0f);
        }
        delta = &(*deltas.back() * *weightTranspose);

        delete weightTranspose;

        // Apply the derivative of the ReLU function
        Matrix outputDerriv = Matrix(*this->outputs[i], 1, this->outputs[i]->size());
        outputDerriv.addCol(1.0f); // Add bias
        outputDerriv.map(ReLUder);  

        // Calculate the delta
        delta->dotInplace(outputDerriv);

        // Store the delta
        deltas.push_back(delta);
    }

    int deltaLen = deltas.size();
    // Calculate the gradients
    for (int i = 0; i < this->weights.size(); i++) {
        // Get the activations for the layer
        Matrix gradient = Matrix(*this->outputs[i], this->outputs[i]->size(), 1);
        gradient.addRow(1.0f); // Add bias

        // Drop the bias row from the delta if it is not the output layer
        if (i < this->weights.size() - 1) {
            deltas[deltaLen - 1 - i]->deleteCol();
        }

        gradient *= *deltas[deltaLen - 1 - i];

        // Add the gradient to the total
        *gradients[i] += gradient;
    }

    // Increment the number of iterations
    iterations++;

    // Clean up the deltas
    for (int i = 0; i < deltas.size(); i++) {
        delete deltas[i];
    }
}
