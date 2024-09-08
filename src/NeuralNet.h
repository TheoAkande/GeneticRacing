// As opposed to "DeepNeuralNets", which is a set of multiple evolved brains, 
// NeuralNet is a single brain that will be taught via conventional methods (back propogation)
// As such, it will be parallelized and constructed differently
// For example, each neural net will be its own object, rather than static
// This also allows for polymorphism for different net architectures (rnn, convolutional, lstm, etc)
// We will also use vectors almost exclusively for flexibility

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
#include <time.h>

#include "Defs.h"

using namespace std;

class FeedForwardNeuralNet
{
    private:
        uint64_t seed;

        vector<int> architecture;           // Number of nodes in each layer
        vector<vector<float> *> weights;    // Pointers to each layer's weights
        vector<vector<float> *> outputs;    // Outputs of each layer. Note: it is not time efficient to store all outputs, but could be useful for backprop?
        vector<GLuint> cbs;                 // Compute buffer objects: 2 * i = weights, 2 * i + 1 = outputs
        int numCbs;                         // Number of compute buffer objects

        void setupArchitecture(void);       // Setup the architecture
        void createRandomWeights(void);     // Initialize random weights
        void feedForward(int layer);        // Feed data from layer to layer + 1
        void loadWeights(string path);      // Load weights from file
        
        static bool initialized;            // Whether the neural net class is initialized
        static float randomWeightRange;     // Range of random weights
        static GLuint invocationShader;     // Shader for invoking the neural net

        static void setupClass(void);       // Setup the class
    public:
        FeedForwardNeuralNet(vector<int> architecture, string weightPath);  // Construct from persisted weights
        FeedForwardNeuralNet(vector<int> architecture, uint64_t seed);      // Construct with random weights
        FeedForwardNeuralNet(vector<int> architecture);                     // Construct with random weights and seed from time

        void invoke(vector<float> *inputs, vector<float> *outputs); // Feed inputs through the network
        void destroy(void); // Free memory
};